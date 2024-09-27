from copy import copy
from dataclasses import dataclass
from datetime import datetime

import cv2
import einops
import numpy as np
import open3d as o3d
import torch
from e3nn import o3
from einops import einsum
from pytorch3d.transforms import matrix_to_quaternion

from gaussian_splatting.scene.cameras import Camera


def points_inside_mask(mask, points):
    if isinstance(mask, Sphere):
        return points_inside_sphere(mask, points.double())
    elif isinstance(mask, Box):
        return points_inside_box(mask, points.double())
    elif isinstance(mask, list) and all([isinstance(x, ImageMask) for x in mask]):
        per_mask = torch.stack([points_inside_image_mask(x, points) for x in mask])
        return torch.all(per_mask != 0, dim=0) & ~torch.all(per_mask == -1, dim=0)


def points_inside_sphere(sphere, points):
    return torch.cdist(sphere.center.unsqueeze(0), points).squeeze(0) < sphere.radius


def points_inside_box(box, points):
    # get four corners of the box (the minimum point and then one in every axis direction)
    c0 = box.center - torch.sum(box.extent * box.axis.t(), dim=1)
    c1 = c0 + 2 * box.extent[0] * box.axis[0]
    c2 = c0 + 2 * box.extent[1] * box.axis[1]
    c3 = c0 + 2 * box.extent[2] * box.axis[2]

    # get edges
    u = c1 - c0
    v = c2 - c0
    w = c3 - c0

    batched_dot = torch.vmap(torch.dot)
    points_double = points.double()

    ux = batched_dot(u.expand((points.shape[0], -1)), points_double)
    vx = batched_dot(v.expand((points.shape[0], -1)), points_double)
    wx = batched_dot(w.expand((points.shape[0], -1)), points_double)

    return ((torch.dot(u, c0) <= ux) & (ux <= torch.dot(u, c1)) &
            (torch.dot(v, c0) <= vx) & (vx <= torch.dot(v, c2)) &
            (torch.dot(w, c0) <= wx) & (wx <= torch.dot(w, c3)))


def points_inside_image_mask(mask, points):
    points_homogeneous = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1).cuda()  # [X, Y, Z, W]
    projection_homogeneous = torch.matmul(mask.cam.full_proj_transform.t(), points_homogeneous.float().unsqueeze(2)).squeeze(2)  # [X, Y, Z, W]
    projection = projection_homogeneous[:, :3] / projection_homogeneous[:, 3].unsqueeze(1)  # [X, Y, Z]

    eps = 0.000001
    inside_image = torch.all((projection[:, :2] >= -1 + eps) & (projection[:, :2] < 1 - eps), dim=1)

    im_dim = torch.tensor([[mask.image.shape[1], mask.image.shape[2]]]).cuda()  # [Y, X]
    px = ((projection[inside_image][:, [1, 0]] + 1) / 2 * im_dim).int()  # [Y, X]

    np_mask = mask.image.squeeze(0).detach().numpy()
    dilated_mask = torch.from_numpy(cv2.dilate(np_mask, np.ones((15, 15), dtype=np.float32))).type(torch.int8).cuda()

    res = -torch.ones(points.shape[0], dtype=torch.int8)
    res[inside_image] = dilated_mask[px[:, 0], px[:, 1]]
    return res


def normal_rotation(from_normal, to_normal, up=torch.tensor([0, 1, 0], dtype=torch.float, device='cuda')):
    # based on https://math.stackexchange.com/questions/624348/finding-rotation-axis-and-angle-to-align-two-oriented-vectors

    v1 = normalize(from_normal)
    v2 = normalize(torch.cross(up, v1))
    v3 = torch.cross(v1, v2)
    V = torch.stack([v1, v2, v3]).t()

    w1 = normalize(to_normal)
    w2 = normalize(torch.cross(up, w1))
    w3 = torch.cross(w1, w2)
    W = torch.stack([w1, w2, w3]).t()

    R = torch.matmul(W, V.t())
    return matrix_to_quaternion(R)


def normalize(v):
    return v / torch.linalg.norm(v)


def look_at(eye, target, up=torch.tensor([0, 1, 0], device='cuda', dtype=torch.float)):
    z_axis = normalize(target - eye)
    x_axis = normalize(torch.cross(up, z_axis))
    y_axis = torch.cross(z_axis, x_axis)

    axes = torch.stack([x_axis, y_axis, z_axis])
    t = -torch.vmap(torch.dot)(axes, eye.unsqueeze(0).expand(3, 3))
    v = torch.cat([axes.transpose(0, 1), t.unsqueeze(0)])
    return torch.cat([v, torch.tensor([0, 0, 0, 1]).unsqueeze(-1)], dim=1)


@dataclass
class Sphere:
    center: torch.Tensor
    radius: float

    def __post_init__(self):
        self.center = self.center.double()


@dataclass
class Box:
    center: torch.Tensor
    axis: torch.Tensor
    extent: torch.Tensor

    def __post_init__(self):
        self.center = self.center.double()
        self.axis = self.axis.double()
        self.extent = self.extent.double()


@dataclass
class ImageMask:
    image: torch.Tensor
    cam: Camera


class DummyCamera:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, camera_center):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center


class DummyPipeline:
    convert_SHs_python = False
    compute_cov3D_python = False
    debug = False


def filter_gaussians(gaussians, select, inplace=False):
    res = gaussians if inplace else copy(gaussians)

    res._xyz = res._xyz[select]
    res._features_dc = res._features_dc[select]
    res._features_rest = res._features_rest[select]
    res._opacity = res._opacity[select]
    res._scaling = res._scaling[select]
    res._rotation = res._rotation[select]

    return res


def current_timestamp_filename():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def up_vec(view_dir):
    return torch.tensor([0, 0, 1] if view_dir[1].abs() == 1 else [0, 1, 0], dtype=torch.float)


def transform_shs(shs_feat, rotation_matrix):
    """
    From: https://github.com/graphdeco-inria/gaussian-splatting/issues/176
    """
    ## rotate shs
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=torch.float)  # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.inverse(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)

    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], - rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], - rot_angles[1], rot_angles[2])

    # rotation of the shs features
    one_degree_shs = shs_feat[:, 0:3]
    one_degree_shs = einops.rearrange(one_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    one_degree_shs = einsum(
        D_1,
        one_degree_shs,
        '... i j, ... j -> ... i',
    )
    one_degree_shs = einops.rearrange(one_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 0:3] = one_degree_shs

    two_degree_shs = shs_feat[:, 3:8]
    two_degree_shs = einops.rearrange(two_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    two_degree_shs = einsum(
        D_2,
        two_degree_shs,
        '... i j, ... j -> ... i',
    )
    two_degree_shs = einops.rearrange(two_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 3:8] = two_degree_shs

    three_degree_shs = shs_feat[:, 8:15]
    three_degree_shs = einops.rearrange(three_degree_shs, 'n shs_num rgb -> n rgb shs_num')
    three_degree_shs = einsum(
        D_3,
        three_degree_shs,
        '... i j, ... j -> ... i',
    )
    three_degree_shs = einops.rearrange(three_degree_shs, 'n rgb shs_num -> n shs_num rgb')
    shs_feat[:, 8:15] = three_degree_shs

    return shs_feat


def flip_normals(mesh):
    flipped = -np.asarray(mesh.vertex_normals)
    mesh.vertex_normals = o3d.utility.Vector3dVector(flipped)
