# Patch-Based Inpainting of 3D Gaussian Splatting

This repository contains the code accompanying my master thesis "Patch-Based Inpainting of 3D Gaussian Splatting" done
at the Computer Graphics and Visualisation research group at TU Delft.
The thesis report can be found in the
[TU Delft repository](https://resolver.tudelft.nl/uuid:cf7e4835-1cbe-4964-83be-c9201c513f08).

## Abstract

Image inpainting is a problem that has been well studied over the last decades.
In contrast, for 3D reconstructions such as neural radiance fields (NeRFs), work in this area is still limited.
Most existing 3D inpainting methods follow a similar approach: they perform image inpainting on the training images and
use the inpainted images for further training of the 3D model.
Due to inconsistencies in the different inpaintings of the images, the 3D inpainting often becomes blurry.
With the advent of 3D Gaussian Splatting (3DGS), we identify a new opportunity for 3D inpainting.
As 3DGS is more explicit in nature than NeRF, we can manipulate the 3D Gaussians directly rather than relying on image
inpainting.
Based on that key idea, we propose a method that works similar to the PatchMatch image inpainting algorithm.
We first construct a nearest-neighbour field (NNF) by searching for nearest-neighbour patches throughout the scene that
look similar to the area we want to inpaint.
After constructing the NNF we copy the contents of the nearest-neighbour patches to the inpainting region and blend them
together to obtain the inpainting result.
In our experiments we found that our method performs well in terms of texture synthesis but struggles with structure
synthesis, similar to the original PatchMatch algorithm.
In cases where only texture synthesis is required to inpaint the area our method is able to provide good results,
although in some cases pre-processing of the scene is necessary, as we found that better quality inputs (e.g. the scene
itself, the surface mesh underlying the scene, and precise masks) drastically improve the results of our method.
Moreover, some parameters of the algorithm are highly scene-dependent and by tailoring them to the scene we can further
enhance the performance of the algorithm.
Besides introducing a 3D inpainting method that directly manipulates the scene contents, our work offers valuable new
insights into 3DGS editing in general.

## Overview

The entire algorithm is implemented in Python, using the PyTorch framework.
All the code can be found inside the `gs_patchmatch` package.

### Prerequisites

- CUDA 11.8
- Python 3.9
- Conda (recommended for easy setup)

Depending on the scene to inpaint, different amounts of VRAM may be necessary.
One can adjust the code to keep some large tensors in RAM instead of VRAM to decrease VRAM usage.

### Setup

Run the following command to set up the conda environment:

```conda env create --file environment.yaml```

### Running

We provide the `inpaint.py` script in the root directory to run the algorithm through a command-line interface.
The script can be run using:

```python inpaint.py -s <path to scene.json file>```

The script takes the following arguments:

**--scene / -s**\
Path to the `scene.json` file, which contains all the information about the scene to inpaint (see the JSON files in the
`scenes` directory).

**--em-iter**\
The number of EM iterations to run for every level of the multi-scale hierarchy (`2` by default).

**--nnf-iter**\
The number of NNF iterations to run to construct the NNF inside a single EM iteration (`25` by default).

**--gaussians-per-point**\
The approximate average number of Gaussians that will be copied per point in the final iteration (`50` by default).
Corresponds to the parameter &gamma; used in the report.

**--patch-size**\
The patch size N to use throughout the algorithm (patches will be N&times;N).
`3` by default.
Needs to be an odd number.
Corresponds to the parameter N used in the report.

**--gaussians-per-pixel**\
The number of Gaussians to insert per rendered pixel in one direction at the coarser levels (`2` by default).
Corresponds to the parameter &psi; used in the report.

**--nnf-cell-res**\
The number of pixels to render in one direction per cell of the patch (`2` by default).
Corresponds to the parameter &phi; used in the report.

**--optimisation-iter**\
The number of optimisation iterations to run (`25` by default).

**--nnf-dist-func**\
The distance function used to compare patches.
Can be either `L2 RGB` or `L2 Delta E` (default).

**--debug**\
Flag to turn on debug mode. More output will be generated in debug mode.
If not in debug mode, only the inpainted scene will be saved as output.
