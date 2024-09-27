import colorsys
import logging
import random
from collections import defaultdict
from typing import Literal, get_args

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from colour import delta_E
from kornia.color import rgb_to_hls, rgb_to_lab
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.transforms.functional import to_pil_image

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix
from .util import DummyCamera, DummyPipeline, current_timestamp_filename, look_at, normalize, up_vec


_max_delta_E = delta_E([0, -128, -128], [100, 127, 127])


def _mse(im_a, im_b):
    return torch.mean(torch.square(im_b - im_a))


def _mse_normalised_delta_E(im_a, im_b):
    im_a_lab = rgb_to_lab(im_a)
    im_b_lab = rgb_to_lab(im_b)

    d = delta_E(im_a_lab.permute(0, 2, 3, 1).flatten(end_dim=2).detach().cpu().numpy(),
                im_b_lab.permute(0, 2, 3, 1).flatten(end_dim=2).detach().cpu().numpy())

    return np.mean(np.square(np.nan_to_num(d, nan=_max_delta_E) / _max_delta_E))


_DistFuncs = Literal['L2 RGB', 'L2 Delta E']


class NNF:
    def __init__(self, patch_renderer, gaussians, dist_func='L2 Delta E'):
        self.renderer = patch_renderer
        self.level_repr = patch_renderer.level_repr
        self.gaussians = gaussians

        if dist_func not in get_args(_DistFuncs):
            raise ValueError(f'Invalid distance function: {dist_func}')

        if dist_func == 'L2 RGB':
            self._dist_func = _mse
        elif dist_func == 'L2 Delta E':
            self._dist_func = _mse_normalised_delta_E

        self.source_patch_cache = {}
        self.target_patch_cache = {}
        self.distance_cache = defaultdict(dict)
        # TODO: maybe it's faster to just render all the patches on initialisation (so we can skip the cache hit check)

        # initialize nnf with random vertices outside the mask
        self.nnf = torch.randint(self.level_repr.k_outside, (self.level_repr.k_inside,))
        self.dist = torch.empty(self.level_repr.k_inside)

        for p_i, neighbour in enumerate(self.nnf):
            self.dist[p_i] = self._distance(p_i, neighbour)

        self.hue_order = None

    def _source_patch(self, p_i):
        if p_i not in self.source_patch_cache:
            self.source_patch_cache[p_i] = (self.renderer.render_source_patch(p_i, self.gaussians)['render']
                                            .unsqueeze(0))  # [1, C, H, W]

        return self.source_patch_cache[p_i]

    def _target_patch(self, p_i):
        if p_i not in self.target_patch_cache:
            self.target_patch_cache[p_i] = (self.renderer.render_target_patch(p_i, self.gaussians, mask=True)['render']
                                            .unsqueeze(0))  # [1, C, H, W]

        return self.target_patch_cache[p_i]

    def _distance(self, index_inside, index_outside):
        if index_outside not in self.distance_cache[index_inside]:
            im_a = self._source_patch(index_inside)
            im_b = self._target_patch(index_outside)

            self.distance_cache[index_inside][index_outside] = self._dist_func(im_a, im_b)

        return self.distance_cache[index_inside][index_outside]

    def minimise(self, n_iterations, prior=None, merge_after_k_iterations=-1, config=None):
        if merge_after_k_iterations == -1:
            merge_after_k_iterations = int(n_iterations / 2)

        # for every nnf iteration
        # do neighbour propagation and random search
        for i in range(n_iterations):
            logging.debug(f'NNF iteration {i + 1}/{n_iterations}')

            if i == merge_after_k_iterations and prior:
                logging.debug('Merging with prior')
                before = self.nnf.clone()
                self._merge_with_prior(prior)
                logging.debug(f'{torch.sum(before != self.nnf)}/{self.nnf.shape[0]} changed')

            reverse_order = (i % 2 == 1)
            point_indices = (range(self.level_repr.k_inside - 1, -1, -1) if reverse_order else
                             range(self.level_repr.k_inside))

            for p_i in point_indices:
                logging.debug(
                    f'{self.level_repr.k_inside - p_i if reverse_order else p_i + 1}/{self.level_repr.k_inside}')

                self._propagation(p_i)
                self._random_search(p_i)

        if config.debug_mode:
            self.plot_renders(config=config)
            self.plot(config=config)
            self.plot(show_targets=False, config=config)

    def _merge_with_prior(self, prior):
        # for every nnf point, find the closest (e.g. Euclidean distance) corresponding point from the prior
        closest_prior_points = torch.cdist(
            self.level_repr.points_inside, prior.level_repr.points_inside).min(dim=1).indices

        # vectors from nearest points to their nnf match
        offsets = (prior.level_repr.points_outside[prior.nnf[closest_prior_points]] -
                   prior.level_repr.points_inside[closest_prior_points])

        # the positions we consider for propagation
        displaced_positions = self.level_repr.points_inside + offsets

        # candidate indices to propagate
        candidates = torch.cdist(displaced_positions, self.level_repr.points_outside).min(dim=1).indices

        # use candidates for updating nnf
        for p_i, candidate in enumerate(candidates):
            self._update_for_candidate(p_i, candidate)

    def _propagation(self, p_i, reverse_order=False):
        # for propagation, simply use the offset between the vertex and its neighbour (point to point)
        # add that offset to the current vertex and just take the closest point
        neighbours = self.level_repr.nearest_points[p_i]

        # limit to the ones we already visited
        visited_neighbours = neighbours[(neighbours > p_i) if reverse_order else (neighbours < p_i)]

        # vectors from nearest points to their nnf match
        offsets = (self.level_repr.points_outside[self.nnf[visited_neighbours]] -
                   self.level_repr.points_inside[visited_neighbours])

        # the positions we consider for propagation
        displaced_positions = self.level_repr.points_inside[p_i] + offsets

        # candidate indices to propagate
        candidates = torch.cdist(displaced_positions, self.level_repr.points_outside).min(dim=1).indices

        # use candidates for updating nnf
        for candidate in candidates:
            self._update_for_candidate(p_i, candidate)

    def _random_search(self, p_i, random_search_windows_ratio=0.5):
        v0 = int(self.nnf[p_i])
        min_search_dist = self.level_repr.min_search_dists[v0]

        search_i = 0
        while True:
            # the max search radius is the max distance between any two points in the point cloud
            # we decrease the search window with every search iteration
            search_radius = self.level_repr.max_search_dist * (random_search_windows_ratio ** search_i)

            if search_radius < min_search_dist:
                break

            # limit the candidates to all points within the search radius from v0
            within_radius = torch.nonzero(
                (self.level_repr.distances_outside[v0] < search_radius) &
                (self.level_repr.distances_outside[v0] >= min_search_dist))

            # select a random candidate from the possible candidates
            candidate = within_radius[random.randint(0, within_radius.shape[0] - 1)]

            # use candidate for updating nnf
            self._update_for_candidate(p_i, candidate)

            search_i += 1

    def _update_for_candidate(self, p_i, candidate):
        p_i = int(p_i)
        candidate = int(candidate)

        d = self._distance(p_i, candidate)

        if d < self.dist[p_i]:
            self.dist[p_i] = d
            self.nnf[p_i] = candidate

    def __getitem__(self, item):
        return self.nnf[item]

    def plot_renders(self, renderer=None, gaussians=None, config=None, hue_order=None):
        if renderer is None:
            renderer = self.renderer

        if gaussians is None:
            gaussians = self.gaussians

        k_inside = self.level_repr.k_inside

        images = torch.ones((k_inside, 3, renderer.render_res, 2 * renderer.render_res + 1), device='cpu')

        for p_i in range(k_inside):
            images[p_i, :, :, :renderer.render_res] = (
                renderer.render_source_patch(p_i, gaussians)['render'].detach().cpu())
            images[p_i, :, :, (renderer.render_res + 1):] = (
                renderer.render_target_patch(self.nnf[p_i], gaussians)['render'].detach().cpu())

        if hue_order is None:
            hls_images = rgb_to_hls(images)
            hues = torch.mean(hls_images[:, 0, ...], dim=[1, 2])
            hue_order = torch.sort(hues).indices
            self.hue_order = hue_order

        t = current_timestamp_filename()
        for i, image in enumerate(images[hue_order]):
            to_pil_image(image).save(config.output_path(f'nnf_{t}/{(i+1):03d}.png'))

        images = images[hue_order].numpy().transpose((0, 2, 3, 1))

        n_cols = math.ceil(math.sqrt(k_inside * 2) / 2)
        n_rows = math.ceil(k_inside / n_cols)

        fig = plt.figure()
        grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=(1 / n_cols))

        for i, ax in enumerate(grid):
            row = i // n_cols
            col = i % n_cols
            if col * n_rows + row < images.shape[0]:
                ax.imshow(images[col * n_rows + row])

        for ax in grid:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        if config:
            fig.savefig(config.output_path(f'plots/{current_timestamp_filename()}.png'))

    def plot(self, render_res=1024, plot_arrows=True, show_targets=True, background=None, config=None):
        if background is None:
            background = torch.zeros(3)

        view_dir = -normalize(torch.mean(self.level_repr.normals_inside, dim=0))
        up = up_vec(view_dir)

        z_axis = view_dir
        x_axis = normalize(torch.cross(up, z_axis))
        y_axis = torch.cross(z_axis, x_axis)

        axis = torch.stack([x_axis, y_axis, z_axis])

        nnf_points = torch.cat([self.level_repr.points_inside, self.level_repr.points_outside])
        transformed_nnf_points = torch.matmul(nnf_points, axis.t())
        transformed_max = torch.max(transformed_nnf_points, dim=0).values
        transformed_min = torch.min(transformed_nnf_points, dim=0).values
        cube_extent = torch.max(transformed_max - transformed_min) / 2

        transformed_center = (transformed_min + transformed_max) / 2
        center = torch.matmul(torch.inverse(axis), transformed_center)

        fov = math.pi / 180  # 1 deg

        cam_dist = cube_extent / math.tan(fov / 2)
        cam_pos = center - view_dir * cam_dist

        znear = cam_dist - cube_extent
        zfar = cam_dist + cube_extent

        projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov, fovY=fov).transpose(0, 1)

        world_view_transform = look_at(cam_pos, center.float(), up=up)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

        cam = DummyCamera(render_res, render_res, fov, fov, znear, zfar, world_view_transform, full_proj_transform,
                          cam_pos)

        im = torch.clamp(render(cam, self.gaussians, DummyPipeline(), background)['render'], min=0, max=1).permute(
            1, 2, 0).detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax.imshow(im)

        half_res = render_res / 2

        if plot_arrows:
            vecs_h = torch.cat([self.level_repr.points_inside, torch.ones(self.level_repr.k_inside, 1)],
                               dim=1)
            inside_proj = torch.matmul(vecs_h, full_proj_transform)
            inside_proj = (inside_proj / inside_proj[:, 3].unsqueeze(-1))[:, :2] * half_res + half_res

            vecs_h = torch.cat([self.level_repr.points_outside[self.nnf],
                                torch.ones(self.level_repr.k_inside, 1)], dim=1)
            outside_proj = torch.matmul(vecs_h, full_proj_transform)
            outside_proj = (outside_proj / outside_proj[:, 3].unsqueeze(-1))[:, :2] * half_res + half_res

            arrow_width = max(1, 10 / math.sqrt(self.level_repr.k_inside))
            arrow_head_width = 5 * arrow_width

            for i in range(self.level_repr.k_inside):
                from_x = float(inside_proj[i][0])
                from_y = float(inside_proj[i][1])

                dx = float(outside_proj[i][0]) - from_x
                dy = float(outside_proj[i][1]) - from_y

                if not show_targets:
                    arrow_length = math.sqrt(dx ** 2 + dy ** 2)
                    scale_factor = 10 * arrow_width / arrow_length
                    dx *= scale_factor
                    dy *= scale_factor

                angle = math.atan(dy / dx) if dx != 0 else (math.pi / 2)
                if angle < 0:
                    angle = math.pi + angle
                if dy < 0 or (dy == 0 and dx < 0):
                    angle += math.pi

                color = (*colorsys.hsv_to_rgb(angle / (2 * math.pi), 1, 1), 0.5)

                ax.arrow(from_x, from_y, dx, dy, width=arrow_width, head_width=arrow_head_width, color=color)

        plt.show()

        if config:
            fig.savefig(config.output_path(f'plots/{current_timestamp_filename()}.png'))
