import numpy as np
import torch


class LevelRepresentation:
    def __init__(self, scene_repr, k_inside, k_outside, k_propagation_points=4):
        self.scene_repr = scene_repr

        self.k_inside = k_inside
        self.k_outside = k_outside

        # sample points on the surface
        self.pc_inside = scene_repr.mesh_inside.sample_points_poisson_disk(number_of_points=self.k_inside,
                                                                           init_factor=10)
        self.pc_outside = scene_repr.mesh_outside.sample_points_poisson_disk(number_of_points=self.k_outside,
                                                                             init_factor=10)

        self.points_inside = torch.from_numpy(np.asarray(self.pc_inside.points)).float().cuda()
        self.normals_inside = torch.from_numpy(np.asarray(self.pc_inside.normals)).float().cuda()
        self.points_outside = torch.from_numpy(np.asarray(self.pc_outside.points)).float().cuda()
        self.normals_outside = torch.from_numpy(np.asarray(self.pc_outside.normals)).float().cuda()

        # order points inside to improve propagation (points are sorted by x-coordinate)
        order = torch.sort(self.points_inside[:, 0])[1]
        self.points_inside = self.points_inside[order]
        self.normals_inside = self.normals_inside[order]

        # compute k nearest points for every point (for propagation)
        distances_inside = torch.cdist(self.points_inside, self.points_inside)
        smallest_distances_inside = torch.topk(distances_inside, min(self.k_inside, k_propagation_points + 1), dim=1,
                                               largest=False)
        self.nearest_points = smallest_distances_inside.indices[:, 1:]  # [n x k]
        # we search for k+1 points and then get rid of the first column because we don't want a point to have itself as
        # one if its nearest points

        # compute min/max distances for random search
        self.distances_outside = torch.cdist(self.points_outside.cpu(), self.points_outside.cpu())
        self.max_search_dist = torch.max(self.distances_outside)
        self.min_search_dists = torch.topk(self.distances_outside, 2, dim=1, largest=False).values[:, -1]

        self.cell_size = torch.mean(smallest_distances_inside.values[:, 1:])

        # select outside points for optimisation phase
        distances_out_to_in = torch.cdist(self.points_outside, self.points_inside)
        min_distances_to_mask = torch.min(distances_out_to_in, dim=1).values
        select_k_outside = min(k_outside, max(10, int(k_inside / 2)))
        self.selected_points_outside = torch.topk(min_distances_to_mask, select_k_outside, largest=False).indices
