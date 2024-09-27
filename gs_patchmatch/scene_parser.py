from __future__ import annotations

import sys
sys.path.append('./gaussian_splatting')

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import open3d as o3d
import torch
from torchvision.io import read_image

from gaussian_splatting.arguments import ModelParams
from gaussian_splatting.scene import Scene
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.scene.gaussian_model import GaussianModel
from .util import Box, ImageMask, Sphere, flip_normals


@dataclass
class SceneInfo:
    source_path: Path
    gaussians: GaussianModel
    mesh: o3d.geometry.TriangleMesh
    use_inpainting_prior: bool
    cameras: list[Camera]
    render_cam: Camera
    cameras_extent: float
    inpainting_mask: Box | list[ImageMask]
    global_mask: Optional[Box | list[ImageMask]] = None


def parse_scene_info(json_file, include_original_images=False):
    with open(json_file, 'r') as f:
        scene_info = json.load(f)

    parser = ArgumentParser()
    model_params = ModelParams(parser)
    args = parser.parse_args(args=[])
    args.source_path = scene_info['training_dataset']
    args.model_path = scene_info['point_cloud']
    dataset = model_params.extract(args)

    gaussians = GaussianModel(3)
    scene = Scene(dataset, gaussians, load_iteration=-1, shuffle=False)
    cameras = scene.getTrainCameras()

    if not include_original_images:
        for cam in cameras:
            cam.original_image = None

    mesh = o3d.io.read_triangle_mesh(scene_info['mesh'])
    if 'flip_normals' in scene_info and scene_info['flip_normals']:
        flip_normals(mesh)

    return SceneInfo(
        Path(args.source_path),
        gaussians,
        mesh,
        scene_info['use_inpainting_prior'],
        cameras,
        cameras[scene_info['render_cam']],
        scene.cameras_extent,
        _parse_json_mask(scene_info['inpainting_mask'], cameras),
        _parse_json_mask(scene_info['global_mask'], cameras) if ('global_mask' in scene_info and
                                                                 scene_info['global_mask'] is not None) else None,
    )


def _parse_json_mask(json_mask, cameras):
    if json_mask['type'] == 'sphere':
        return _parse_json_sphere(json_mask)
    elif json_mask['type'] == 'box':
        return _parse_json_box(json_mask)
    elif json_mask['type'] == 'images':
        return _read_image_masks(Path(json_mask['path']), cameras)


def _parse_json_sphere(json_sphere):
    return Sphere(torch.tensor(json_sphere['center']), json_sphere['radius'])


def _parse_json_box(json_box):
    return Box(torch.tensor(json_box['center']), torch.tensor(json_box['axis']), torch.tensor(json_box['extent']))


def _read_image_masks(im_path, cameras):
    res = []

    for cam in cameras:
        mask_file = [f for f in im_path.iterdir() if f.stem == cam.image_name][0]
        res.append(ImageMask(read_image(str(mask_file)), cam))

    return res
