from argparse import ArgumentParser
from pathlib import Path

from gs_patchmatch.inpainting import inpaint


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--scene', type=Path, required=True, help='Scene info JSON file')
    parser.add_argument('--em-iter', dest='n_em_iterations', type=int, help='Number of EM iterations')
    parser.add_argument('--nnf-iter', dest='n_nnf_iterations', type=int, help='Number of NNF iterations')
    parser.add_argument('--gaussians-per-point', dest='copy_gaussians_per_point', type=int, help='Approximate number of Gaussians that will be copied per point after constructing the finest-level NNF')
    parser.add_argument('--patch-size', type=int, help='Patch size (e.g. for patch size N, the patch will be NxN cells)')
    parser.add_argument('--gaussians-per-pixel', dest='insert_gaussians_per_pixel', type=int, help='Number of Gaussians that will be inserted in every dimension per pixel of the patch render when initialising the optimisation phase')
    parser.add_argument('--nnf-cell-res', type=int, help='Number of pixels to render in every dimension per cell. Used to render patches during NNF construction')
    parser.add_argument('--optimisation-iter', dest='n_batch_optimisation_iterations', type=int, help='Number of batch optimisation iterations')
    parser.add_argument('--nnf-dist-func', type=str, choices=['L2 RGB', 'L2 Delta E'], help='Distance function used during NNF construction')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    inpaint(**{k: v for k, v in args.__dict__.items() if v is not None})
