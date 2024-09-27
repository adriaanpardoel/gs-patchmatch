import numpy as np
import open3d as o3d
import torch
from pytorch3d.structures import Meshes

from .util import points_inside_mask


class SceneRepresentation:
    def __init__(self, mesh, inpainting_mask, use_inpainting_prior=False, global_mask=None, config=None):
        self.original_mesh = mesh
        self.inpainting_mask = inpainting_mask
        self.global_mask = global_mask

        self.masked_mesh = self._generate_masked_mesh()
        self.mesh_outside = self._generate_target_mesh()
        self.mesh_inside = self._generate_source_mesh(use_inpainting_prior=use_inpainting_prior, config=config)
        self.p3d_meshes = self._generate_pytorch3d_meshes()

        self.area_outside = self.mesh_outside.get_surface_area()
        self.area_inside = self.mesh_inside.get_surface_area()

    def _generate_masked_mesh(self):
        masked_mesh_vertices = ~points_inside_mask(
            self.inpainting_mask, torch.from_numpy(np.asarray(self.original_mesh.vertices)).cuda())

        vertex_indices = torch.nonzero(masked_mesh_vertices).squeeze(-1)
        return self.original_mesh.select_by_index(vertex_indices.detach().cpu().numpy())

    def _generate_target_mesh(self):
        if not self.global_mask:
            return self.masked_mesh

        global_mask_vertices = points_inside_mask(
            self.global_mask, torch.from_numpy(np.asarray(self.masked_mesh.vertices)).cuda())

        vertex_indices = torch.nonzero(global_mask_vertices).squeeze(-1)
        return self.masked_mesh.select_by_index(vertex_indices.detach().cpu().numpy())

    def _generate_source_mesh(self, use_inpainting_prior=False, config=None):
        if use_inpainting_prior:
            mask_vertices = points_inside_mask(
                self.inpainting_mask, torch.from_numpy(np.asarray(self.original_mesh.vertices)).cuda())

            vertex_indices = torch.nonzero(mask_vertices).squeeze(-1)
            res = self.original_mesh.select_by_index(vertex_indices.detach().cpu().numpy())
        else:
            point_cloud = self.masked_mesh.sample_points_uniformly(number_of_points=1_000_000)
            poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)

            inside_mask = points_inside_mask(self.inpainting_mask,
                                             torch.from_numpy(np.asarray(poisson_mesh.vertices)).cuda())
            res = poisson_mesh.select_by_index(torch.arange(inside_mask.shape[0])[inside_mask].detach().cpu().numpy())

            if config and config.debug_mode:
                o3d.io.write_triangle_mesh(str(config.output_path('meshes/mesh_poisson.ply')), poisson_mesh)

        if config and config.debug_mode:
            o3d.io.write_triangle_mesh(str(config.output_path('meshes/mesh_masked.ply')), self.masked_mesh)
            o3d.io.write_triangle_mesh(str(config.output_path('meshes/mesh_inside.ply')), res)
            o3d.io.write_triangle_mesh(str(config.output_path('meshes/mesh_outside.ply')), self.mesh_outside)

        return res

    def _generate_pytorch3d_meshes(self):
        return Meshes(verts=[torch.from_numpy(np.asarray(self.mesh_outside.vertices)).float().cuda(),
                             torch.from_numpy(np.asarray(self.mesh_inside.vertices)).float().cuda()],
                      faces=[torch.from_numpy(np.asarray(self.mesh_outside.triangles)).float().cuda(),
                             torch.from_numpy(np.asarray(self.mesh_inside.triangles)).float().cuda()])
