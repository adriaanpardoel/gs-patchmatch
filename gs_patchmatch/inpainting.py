from __future__ import annotations

import sys
sys.path.append('./gaussian_splatting')

import logging
import random
from copy import copy

import math
import matplotlib.pyplot as plt
import open3d as o3d
import torch
from numpy.random import Generator, PCG64
from torchvision.transforms.functional import to_pil_image

from gaussian_splatting.gaussian_renderer import render
from .config import Config
from .level_representation import LevelRepresentation
from .nnf import NNF
from .patch_renderer import PatchRenderer
from .patch_voting import copy_neighbour_gaussians, initialise_patch_voting, perform_optimisation
from .scene_parser import parse_scene_info
from .scene_representation import SceneRepresentation
from .util import DummyPipeline, current_timestamp_filename, filter_gaussians, points_inside_mask


def inpaint(scene, n_em_iterations=2, n_nnf_iterations=25, copy_gaussians_per_point=50, patch_size=3,
            insert_gaussians_per_pixel=2, nnf_cell_res=2, n_batch_optimisation_iterations=25,
            nnf_dist_func='L2 Delta E', debug=False):
    if patch_size % 2 != 1:
        raise ValueError('patch_size must be an odd number')

    torch.set_default_device('cuda')

    plt.rcParams['figure.dpi'] = 200

    random.seed = 0
    torch.manual_seed(1)
    o3d.utility.random.seed(2)
    np_random_generator = Generator(PCG64(3))

    with torch.no_grad():
        scene_info = parse_scene_info(scene)
        gaussians = scene_info.gaussians

        background = torch.zeros(3)

        config = Config(debug_mode=debug, debug_cam=scene_info.render_cam)

        if debug:
            logging.basicConfig(filename=config.output_path('debug.log'), level=logging.DEBUG)

        logging.debug(f'scene: {scene}')
        logging.debug(f'n_em_iterations: {n_em_iterations}')
        logging.debug(f'n_nnf_iterations: {n_nnf_iterations}')
        logging.debug(f'copy_gaussians_per_point: {copy_gaussians_per_point}')
        logging.debug(f'patch_size: {patch_size}')
        logging.debug(f'insert_gaussians_per_pixel: {insert_gaussians_per_pixel}')
        logging.debug(f'nnf_cell_res: {nnf_cell_res}')
        logging.debug(f'n_batch_optimisation_iterations: {n_batch_optimisation_iterations}')
        logging.debug(f'nnf_dist_func: {nnf_dist_func}')

        scene_repr = SceneRepresentation(
            scene_info.mesh, scene_info.inpainting_mask, use_inpainting_prior=scene_info.use_inpainting_prior,
            global_mask=scene_info.global_mask, config=config)

        gaussians_outside = ~points_inside_mask(scene_info.inpainting_mask, gaussians.get_xyz)
        original_gaussians = filter_gaussians(gaussians, gaussians_outside)

        if config.debug_mode:
            im_original = to_pil_image(torch.clamp(render(
                config.debug_cam, gaussians, DummyPipeline(), background)['render'], min=0, max=1))
            plt.imshow(im_original)
            plt.show()
            im_original.save(config.output_path(f'renders/{current_timestamp_filename()}_original.png'))

            im_masked = to_pil_image(torch.clamp(render(
                config.debug_cam, original_gaussians, DummyPipeline(), background)['render'], min=0, max=1))
            plt.imshow(im_masked)
            plt.show()
            im_masked.save(config.output_path(f'renders/{current_timestamp_filename()}_masked.png'))

            original_gaussians.save_ply(config.output_path(f'point_cloud/{current_timestamp_filename()}.ply'))

        if not scene_info.use_inpainting_prior:
            gaussians = copy(original_gaussians)

        nnf_render_res = nnf_cell_res * patch_size
        upscale_render_res = 2 * nnf_render_res
        final_optimisation_render_res = 8 * nnf_render_res

        n_gaussians_outside = original_gaussians.get_xyz.shape[0]
        points_per_area = n_gaussians_outside / scene_repr.masked_mesh.get_surface_area() / copy_gaussians_per_point
        max_points_inside = int(points_per_area * scene_repr.area_inside)
        max_points_outside = int(points_per_area * scene_repr.area_outside)
        n_levels = int(math.log(max_points_inside, 4))

        nnf_prior = None

        # for every resolution
        for level in range(n_levels - 1, -1, -1):
            logging.debug(f'Level {level}')

            final_level = (level == 0)

            factor = 0.25 ** level
            k_inside = int(factor * max_points_inside)
            k_outside = int(factor * max_points_outside)

            level_repr = LevelRepresentation(scene_repr, k_inside, k_outside)

            if config.debug_mode:
                o3d.io.write_point_cloud(str(config.output_path(f'levels/level{level}_inside.ply')),
                                         level_repr.pc_inside)
                o3d.io.write_point_cloud(str(config.output_path(f'levels/level{level}_outside.ply')),
                                         level_repr.pc_outside)

            nnf_renderer = PatchRenderer(level_repr, patch_size=patch_size, render_res=nnf_render_res,
                                         background=background)
            upscale_renderer = PatchRenderer(level_repr, patch_size=patch_size, render_res=(
                final_optimisation_render_res if final_level else upscale_render_res), background=background)

            for i_em in range(n_em_iterations):
                logging.debug(f'EM iteration {i_em + 1}/{n_em_iterations}')

                nnf = NNF(nnf_renderer, gaussians, dist_func=nnf_dist_func)
                nnf.minimise(n_nnf_iterations, prior=nnf_prior, config=config)
                nnf_prior = nnf
                logging.debug('Generated NNF')

                optimisation_renderer = upscale_renderer if i_em == n_em_iterations - 1 else nnf_renderer

                # so now we have an NNF, let's do patch voting
                if final_level:
                    new_gaussians = copy_neighbour_gaussians(nnf, original_gaussians)
                else:
                    new_gaussians = initialise_patch_voting(
                        nnf, original_gaussians, scene_info.inpainting_mask, optimisation_renderer,
                        k_gaussians_per_pixel=insert_gaussians_per_pixel)

                # combine gaussians and new_gaussians and perform optimisation
                gaussians = perform_optimisation(
                    n_batch_optimisation_iterations, nnf, optimisation_renderer, original_gaussians, new_gaussians,
                    final_level, scene_info.cameras_extent, background, np_random_generator, config=config)

                # plot NNF after optimisation
                nnf.gaussians = gaussians
                nnf.plot(plot_arrows=False, config=config)

                # free some memory
                nnf.gaussians = None
                del new_gaussians

                if final_level:
                    gaussians.save_ply(config.output_path(f'point_cloud/{current_timestamp_filename()}.ply'))

    logging.debug('Done')
