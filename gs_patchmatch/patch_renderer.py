import math
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, MeshRasterizer, RasterizationSettings, look_at_view_transform
# from pytorch3d.renderer.opengl import MeshRasterizerOpenGL
from pytorch3d.transforms import quaternion_apply

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix
from .util import DummyCamera, DummyPipeline, filter_gaussians, look_at, normalize, up_vec


class PatchRenderer:
    def __init__(self, level_repr, patch_size=3, render_res=8, background=None):
        self.level_repr = level_repr
        self.patch_size = patch_size
        self.render_res = render_res
        self.background = torch.zeros(3) if background is None else background

        self.fov = math.pi / 180  # 1 deg

        cube_extent = 0.5 * level_repr.cell_size * patch_size
        self.cam_dist = cube_extent / math.tan(self.fov / 2)

        self.znear = self.cam_dist - cube_extent
        self.zfar = self.cam_dist + cube_extent

        self.projection_matrix = getProjectionMatrix(
            znear=self.znear, zfar=self.zfar, fovX=self.fov, fovY=self.fov).transpose(0, 1)

        raster_settings = RasterizationSettings(
            image_size=self.render_res,
            faces_per_pixel=1,
            bin_size=0,
        )
        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)

        self.pipeline = DummyPipeline()

    def render_source_patch(self, p_i, gaussians, mask=False, viewpoint_rotation=None):
        point = self.level_repr.points_inside[p_i]
        normal = self.level_repr.normals_inside[p_i]

        return self._render_patch(point, normal, gaussians, mask=mask, viewpoint_rotation=viewpoint_rotation)

    def render_target_patch(self, p_i, gaussians, mask=False, viewpoint_rotation=None):
        point = self.level_repr.points_outside[p_i]
        normal = self.level_repr.normals_outside[p_i]

        return self._render_patch(point, normal, gaussians, mask=mask, viewpoint_rotation=viewpoint_rotation)

    def _render_patch(self, point, normal, gaussians, mask=False, viewpoint_rotation=None):
        rotated_normal = quaternion_apply(viewpoint_rotation, normal) if viewpoint_rotation is not None else normal

        view_dir = normalize(-rotated_normal)
        cam_pos = point - view_dir * self.cam_dist
        up = up_vec(view_dir)

        res = self._render_gaussians(cam_pos, point, up, gaussians)

        if mask is True:
            mask = self._render_mask(cam_pos, point, up)

        if torch.is_tensor(mask):
            image = res['render'].clone()
            image[:, mask] = float('nan')
            res['render'] = image
            res['mask'] = mask

        return res

    def _render_gaussians(self, cam_pos, target, up, gaussians):
        distances = torch.sqrt(torch.sum(torch.square(gaussians.get_xyz - cam_pos), dim=1))
        behind_znear = distances > self.znear

        rendering_gaussians = filter_gaussians(gaussians, behind_znear)

        world_view_transform = look_at(cam_pos, target, up=up)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)

        cam = DummyCamera(self.render_res, self.render_res, self.fov, self.fov, self.znear, self.zfar,
                          world_view_transform, full_proj_transform, cam_pos)

        render_pkg = render(cam, rendering_gaussians, self.pipeline, self.background)
        render_pkg['rendered_gaussians'] = behind_znear
        return render_pkg

    def _render_mask(self, cam_pos, target, up):
        """
        This function assumes that the view direction is orthogonal to a side of the box and that the box is a cube.
        """
        R, T = look_at_view_transform(eye=cam_pos.unsqueeze(0),
                                      at=target.unsqueeze(0),
                                      up=up.unsqueeze(0),
                                      device='cuda')

        cameras = FoVPerspectiveCameras(R=R, T=T, znear=self.znear, zfar=self.zfar, fov=self.fov, degrees=False,
                                        device='cuda')

        fragments = self.rasterizer(self.level_repr.scene_repr.p3d_meshes, cameras=cameras)
        zbuf = fragments.zbuf.squeeze(3)
        zbuf[zbuf < 0] = self.zfar

        return torch.flip(zbuf[1] <= zbuf[0], [0, 1])
