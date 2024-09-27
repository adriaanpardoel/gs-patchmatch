import logging
import random
from argparse import ArgumentParser
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_apply, quaternion_multiply, quaternion_to_matrix
from scipy.stats import cosine
from torch import nn
from torchvision.transforms.functional import resize, to_pil_image

from gaussian_splatting.arguments import OptimizationParams
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.loss_utils import l1_loss
from gaussian_splatting.utils.sh_utils import RGB2SH
from .util import (Box, DummyPipeline, current_timestamp_filename, filter_gaussians, normal_rotation, normalize,
                   points_inside_box, points_inside_mask, transform_shs, up_vec)


nested_dict = lambda: defaultdict(nested_dict)


def fix_render_pkg(render_pkg, n_gaussians):
    viewspace_point_tensor, visibility_filter, radii, rendered_gaussians = render_pkg['viewspace_points'], render_pkg[
        'visibility_filter'], render_pkg['radii'], render_pkg['rendered_gaussians']

    res_viewspace_point_tensor = torch.zeros((n_gaussians, 3), dtype=viewspace_point_tensor.dtype)
    res_visibility_filter = torch.zeros(n_gaussians, dtype=visibility_filter.dtype)
    res_radii = torch.zeros(n_gaussians, dtype=radii.dtype)

    res_viewspace_point_tensor[rendered_gaussians] = viewspace_point_tensor
    res_visibility_filter[rendered_gaussians] = visibility_filter
    res_radii[rendered_gaussians] = radii

    if viewspace_point_tensor.grad is not None:
        res_viewspace_point_tensor.grad = torch.zeros(res_viewspace_point_tensor.shape)
        res_viewspace_point_tensor.grad[rendered_gaussians] = viewspace_point_tensor.grad

    render_pkg['viewspace_points'] = res_viewspace_point_tensor
    render_pkg['visibility_filter'] = res_visibility_filter
    render_pkg['radii'] = res_radii


def set_requires_grad(tensor, requires_grad):
    """Returns a new tensor with the specified requires_grad setting."""
    return tensor.detach().clone().requires_grad_(requires_grad)


def copy_neighbour_gaussians(nnf, gaussians):
    new_gaussians = GaussianModel(3)
    new_gaussians.active_sh_degree = 3

    for p_i in range(nnf.level_repr.k_inside):
        own_pos = nnf.level_repr.points_inside[p_i]
        own_normal = nnf.level_repr.normals_inside[p_i]

        other_pos = nnf.level_repr.points_outside[nnf[p_i]]
        other_normal = nnf.level_repr.normals_outside[nnf[p_i]]

        box_rotation = normal_rotation(torch.tensor([0, 0, -1], dtype=torch.float), other_normal)
        box_axis = quaternion_apply(box_rotation, torch.eye(3))
        copy_box = Box(other_pos, box_axis, torch.ones(3) * nnf.level_repr.cell_size)
        q = normal_rotation(other_normal, own_normal)

        copy_gaussians = points_inside_box(copy_box, gaussians._xyz)
        logging.debug(f'Copying {torch.sum(copy_gaussians)} Gaussians for point {p_i}...')

        xyz_translated = gaussians._xyz[copy_gaussians] - other_pos
        xyz_rotated = quaternion_apply(q, xyz_translated)
        xyz_new = xyz_rotated + own_pos
        new_gaussians._xyz = torch.cat([new_gaussians._xyz, xyz_new.float()])

        rotation = gaussians._rotation[copy_gaussians]
        new_rotation = quaternion_multiply(q, rotation)
        new_gaussians._rotation = torch.cat([new_gaussians._rotation, new_rotation])

        features_rest = gaussians._features_rest[copy_gaussians]
        new_features_rest = transform_shs(features_rest, quaternion_to_matrix(q))
        new_gaussians._features_rest = torch.cat([new_gaussians._features_rest, new_features_rest])

        new_gaussians._features_dc = torch.cat([new_gaussians._features_dc, gaussians._features_dc[copy_gaussians]])
        new_gaussians._opacity = torch.cat([new_gaussians._opacity, gaussians._opacity[copy_gaussians]])
        new_gaussians._scaling = torch.cat([new_gaussians._scaling, gaussians._scaling[copy_gaussians]])

    return new_gaussians


def initialise_patch_voting(nnf, original_gaussians, inpainting_mask, renderer, k_gaussians_per_pixel=2):
    new_gaussians = GaussianModel(3)
    new_gaussians.active_sh_degree = 3

    for p_i in range(nnf.level_repr.k_inside):
        target_patch = renderer.render_target_patch(nnf[p_i], original_gaussians)['render']
        extended_target_patch = torch.repeat_interleave(
            torch.repeat_interleave(target_patch, k_gaussians_per_pixel, dim=1), k_gaussians_per_pixel, dim=2)

        own_pos = nnf.level_repr.points_inside[p_i]
        own_normal = nnf.level_repr.normals_inside[p_i]

        view_dir = -own_normal
        up = up_vec(view_dir)

        z_axis = view_dir
        x_axis = normalize(torch.cross(up, z_axis))
        y_axis = torch.cross(z_axis, x_axis)

        gaussians_res = renderer.render_res * k_gaussians_per_pixel
        n_new_gaussians = gaussians_res ** 2

        r = torch.arange(gaussians_res)
        pixel_indices = torch.stack([
            r.repeat_interleave(gaussians_res),
            r.repeat(gaussians_res)
        ]).transpose(0, 1)
        pixel_size = nnf.level_repr.cell_size * renderer.patch_size / gaussians_res
        offsets = (pixel_indices - gaussians_res / 2 + 0.5) * pixel_size
        new_xyz = (own_pos +
                   offsets[:, 0].unsqueeze(1).expand((n_new_gaussians, 3)) * y_axis.expand((n_new_gaussians, -1)) +
                   offsets[:, 1].unsqueeze(1).expand((n_new_gaussians, 3)) * x_axis.expand((n_new_gaussians, -1)))
        xyz_jitter = (torch.rand_like(new_xyz) - 0.5) * 2 * 0.25 * pixel_size
        new_gaussians._xyz = torch.cat([new_gaussians._xyz, new_xyz + xyz_jitter])
        # new_gaussians._xyz = torch.cat([new_gaussians._xyz, new_xyz])

        new_scaling = torch.tensor([pixel_size / 2, pixel_size / 2, pixel_size / 10], dtype=torch.float).expand(
            (n_new_gaussians, -1))
        new_scaling = new_scaling * (0.75 + torch.rand_like(new_scaling) * 0.5)
        new_scaling = new_gaussians.scaling_inverse_activation(new_scaling)
        new_gaussians._scaling = torch.cat([new_gaussians._scaling, new_scaling])

        q = normal_rotation(torch.tensor([0, 0, -1], dtype=torch.float), own_normal)
        new_rotation = q.expand((n_new_gaussians, -1))
        new_gaussians._rotation = torch.cat([new_gaussians._rotation, new_rotation])

        new_features_dc = RGB2SH(torch.reshape(extended_target_patch.permute(1, 2, 0), (n_new_gaussians, 1, 3)))
        new_gaussians._features_dc = torch.cat([new_gaussians._features_dc, new_features_dc])

        new_features_rest = torch.zeros((n_new_gaussians, 15, 3))
        new_gaussians._features_rest = torch.cat([new_gaussians._features_rest, new_features_rest])

        new_opacity = new_gaussians.inverse_opacity_activation(torch.ones((n_new_gaussians, 1), dtype=torch.float) * 0.5)
        new_gaussians._opacity = torch.cat([new_gaussians._opacity, new_opacity])

    # remove any new gaussians outside the inpainting mask
    gaussians_inside = points_inside_mask(inpainting_mask, new_gaussians.get_xyz)
    filter_gaussians(new_gaussians, gaussians_inside, inplace=True)

    return new_gaussians


def perform_optimisation(n_batch_iterations, nnf, optimisation_renderer, original_gaussians, new_gaussians, final_level, cameras_extent, background, np_random_generator, config=None):
    random_rotations = _random_rotations(10, np_random_generator=np_random_generator)

    gt_outside, masks_outside = _render_gt_outside(optimisation_renderer, original_gaussians, random_rotations)
    gt_inside = _render_gt_inside(optimisation_renderer, original_gaussians, random_rotations, nnf, final_level)

    losses_inside = torch.zeros(n_batch_iterations)
    losses_outside = torch.zeros(n_batch_iterations)
    losses_total = torch.zeros(n_batch_iterations)

    k_points = optimisation_renderer.level_repr.k_inside

    n_original_gaussians = original_gaussians.get_xyz.shape[0]

    with torch.enable_grad():
        gaussians = _initialise_optimisation_gaussians(original_gaussians, new_gaussians)

        for param_group in gaussians.optimizer.param_groups:
            logging.debug(f'Learning rate {param_group["name"]}: {param_group["lr"]}')

        if config and config.debug_mode:
            nnf.plot_renders(renderer=optimisation_renderer, gaussians=gaussians, config=config)

            im = to_pil_image(torch.clamp(render(
                config.debug_cam, gaussians, DummyPipeline(), background)['render'], min=0, max=1))
            plt.imshow(im)
            plt.show()
            im.save(config.output_path(f'optimisation_renders/{current_timestamp_filename()}_0.png'))

        for i in range(n_batch_iterations):
            logging.debug(f'Optimisation batch iteration {i + 1}/{n_batch_iterations}')
            logging.debug(f'{gaussians.get_xyz.shape[0]} Gaussians')

            # Go over the whole batch of points inside
            for p_i in range(k_points):
                q_i = random.randint(0, random_rotations.shape[0] - 1)
                random_q = random_rotations[q_i]

                render_pkg = optimisation_renderer.render_source_patch(p_i, gaussians, viewpoint_rotation=random_q)
                image = render_pkg['render']
                gt_image = gt_inside[p_i][q_i]
                patch_dist = nnf.dist[p_i]

                losses_inside[i] += set_requires_grad(_handle_loss(gaussians, image, gt_image, patch_dist, render_pkg,
                                                                   n_original_gaussians), False)

                # Now a point outside
                p_j = int(optimisation_renderer.level_repr.selected_points_outside[random.randint(
                    0, optimisation_renderer.level_repr.selected_points_outside.shape[0] - 1)])

                render_pkg = optimisation_renderer.render_target_patch(p_j, gaussians, mask=masks_outside[p_j][q_i],
                                                                       viewpoint_rotation=random_q)
                image = torch.nan_to_num(render_pkg['render'], nan=0)
                gt_image = torch.nan_to_num(gt_outside[p_j][q_i], nan=0)
                patch_dist = 0

                losses_outside[i] += set_requires_grad(_handle_loss(gaussians, image, gt_image, patch_dist, render_pkg,
                                                                    n_original_gaussians), False)

            losses_inside[i] /= k_points
            losses_outside[i] /= k_points
            losses_total[i] = (losses_inside[i] + losses_outside[i]) / 2

            logging.debug(f'Loss inside: {losses_inside[i]}')
            logging.debug(f'Loss outside: {losses_outside[i]}')
            logging.debug(f'Loss total: {losses_total[i]}')

            # Do optimizer step for entire batch at once
            with torch.no_grad():
                # Zero out the gradients of the original Gaussians
                for t in [gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._opacity,
                          gaussians._scaling, gaussians._rotation]:
                    t.grad[:n_original_gaussians] = 0

                # Optimizer step
                gaussians.optimizer.step()

                if i % 10 == 0 and i > 0:
                    # Pruning sets the gradients to None, but the gradients are still needed to decide which Gaussians
                    # to densify/prune
                    size_threshold = optimisation_renderer.render_res
                    gaussians.densify_and_prune(0.0002 * k_points, 0.005, cameras_extent, size_threshold,
                                                prune_from=n_original_gaussians)

                gaussians.optimizer.zero_grad(set_to_none=True)

                if ((i + 1) % 5 == 0 or i == n_batch_iterations - 1) and config and config.debug_mode:
                    im = to_pil_image(torch.clamp(render(
                        config.debug_cam, gaussians, DummyPipeline(), background)['render'], min=0, max=1))
                    plt.imshow(im)
                    plt.show()
                    im.save(config.output_path(f'optimisation_renders/{current_timestamp_filename()}_{i + 1}.png'))

    if config and config.debug_mode:
        nnf.plot_renders(renderer=optimisation_renderer, gaussians=gaussians, config=config, hue_order=nnf.hue_order)

        plt.plot(losses_inside.detach().cpu().numpy())
        plt.plot(losses_outside.detach().cpu().numpy())
        plt.plot(losses_total.detach().cpu().numpy())
        plt.title('Loss')
        plt.legend(['Inside', 'Outside', 'Total'])
        plt.show()
        plt.savefig(config.output_path(f'plots/{current_timestamp_filename()}.png'))

    return gaussians


def _random_rotations(n, np_random_generator=None):
    res = torch.empty((n, 4))

    # always include no rotation
    res[0] = torch.tensor([1, 0, 0, 0])

    for i in range(1, n):
        # generate random quaternion that represents a rotation uniformly distributed across the sphere
        # TODO: vectorise computation instead of for-loop
        random_axis = normalize(torch.randn(3))
        random_angle = cosine.rvs(scale=1 / 6, random_state=np_random_generator)  # limit to 30 deg
        axis_angle = random_axis * random_angle
        res[i] = axis_angle_to_quaternion(axis_angle)

    return res


def _render_gt_outside(renderer, gaussians, rotations):
    gt_outside = nested_dict()
    masks_outside = nested_dict()

    for p_i in renderer.level_repr.selected_points_outside:
        for q_i, q in enumerate(rotations):
            render_pkg = renderer.render_target_patch(p_i, gaussians, mask=True, viewpoint_rotation=q)
            gt_outside[int(p_i)][q_i] = render_pkg['render']
            masks_outside[int(p_i)][q_i] = render_pkg['mask']

    return gt_outside, masks_outside


def _render_gt_inside(renderer, gaussians, rotations, nnf, final_level=False):
    gt_inside = nested_dict()

    for p_i in range(renderer.level_repr.k_inside):
        for q_i, q in enumerate(rotations):
            if final_level:
                gt_inside[int(p_i)][q_i] = renderer.render_target_patch(
                    nnf[p_i], gaussians, viewpoint_rotation=q)['render']
            else:
                gt_image = nnf.renderer.render_target_patch(nnf[p_i], gaussians, viewpoint_rotation=q)['render']
                gt_inside[int(p_i)][q_i] = resize(gt_image, renderer.render_res)

    return gt_inside


def _initialise_optimisation_gaussians(original_gaussians, new_gaussians):
    gaussians = GaussianModel(3)
    gaussians.active_sh_degree = 3

    gaussians._xyz = nn.Parameter(torch.cat([set_requires_grad(original_gaussians._xyz, False),
                                             set_requires_grad(new_gaussians._xyz, True)]))
    gaussians._features_dc = nn.Parameter(torch.cat([set_requires_grad(original_gaussians._features_dc, False),
                                                     set_requires_grad(new_gaussians._features_dc, True)]))
    gaussians._features_rest = nn.Parameter(torch.cat([set_requires_grad(original_gaussians._features_rest, False),
                                                       set_requires_grad(new_gaussians._features_rest, True)]))
    gaussians._opacity = nn.Parameter(torch.cat([set_requires_grad(original_gaussians._opacity, False),
                                                 set_requires_grad(new_gaussians._opacity, True)]))
    gaussians._scaling = nn.Parameter(torch.cat([set_requires_grad(original_gaussians._scaling, False),
                                                 set_requires_grad(new_gaussians._scaling, True)]))
    gaussians._rotation = nn.Parameter(torch.cat([set_requires_grad(original_gaussians._rotation, False),
                                                  set_requires_grad(new_gaussians._rotation, True)]))

    gaussians.max_radii2D = set_requires_grad(torch.zeros(gaussians._xyz.shape[0]), False)

    gaussians.training_setup(OptimizationParams(ArgumentParser()))

    return gaussians


def _handle_loss(gaussians, image, gt_image, patch_dist, render_pkg, n_original_gaussians):
    loss = l1_loss(image, gt_image)

    patch_similarity = 1 - patch_dist  # [0, 1]
    loss = loss * (patch_similarity ** 2)

    try:  # quick-fix for rare case where no gaussians are rendered because our surface estimate is incorrect
        loss.backward()
    except:
        pass

    fix_render_pkg(render_pkg, gaussians.get_xyz.shape[0])
    viewspace_point_tensor, visibility_filter, radii = render_pkg['viewspace_points'], render_pkg[
        'visibility_filter'], render_pkg['radii']

    with torch.no_grad():
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                             radii[visibility_filter])
        visibility_filter[:n_original_gaussians] = False

        try:  # quick-fix for rare case where no gaussians are rendered because our surface estimate is incorrect
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        except:
            pass

    return loss
