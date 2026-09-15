"""복셀을 동일한 축 비율로 표시하고 PNG로 저장한다."""
import argparse
import os
from pathlib import Path
import numpy as np
from common import read_json


def render(voxels, bounds, output=None, show=False):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.runtime' / 'matplotlib'))
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if voxels.ndim != 3 or voxels.dtype != np.bool_:
        raise ValueError('3차원 bool 복셀 배열이 필요합니다.')
    edges = np.meshgrid(*[np.linspace(lo, hi, size+1) for (lo, hi), size in zip(bounds, voxels.shape)], indexing='ij')
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection='3d')
    axis.voxels(*edges, voxels, facecolors='steelblue', edgecolor=None)
    axis.set(xlabel='X', ylabel='Y (front)', zlabel='Z', title=f'Manual orthographic Visual Hull | {voxels.sum()} voxels')
    axis.set_xlim(*bounds[0]); axis.set_ylim(*bounds[1]); axis.set_zlim(*bounds[2])
    axis.set_box_aspect([hi-lo for lo, hi in bounds])
    if output:
        figure.savefig(output, dpi=140, bbox_inches='tight')
    if show:
        plt.show()
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--save')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    path = Path(args.input)
    metadata = path.with_name('metadata.json')
    bounds = read_json(metadata)['config']['bounds'] if metadata.exists() else [[-1, 1]]*3
    if args.save and Path(args.save).exists():
        raise FileExistsError(args.save)
    render(np.load(path, allow_pickle=False), bounds, args.save, not args.no_show)
