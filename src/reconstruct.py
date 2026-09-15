"""수동 직교투영으로 세 실루엣의 복셀 교집합을 구한다."""
import argparse
from pathlib import Path
import numpy as np
from common import read_image, read_json, write_json


def voxel_centers(grid, bounds):
    bounds = np.asarray(bounds, dtype=float)
    if not 1 <= grid <= 160 or bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError('grid는 1~160, bounds는 축별 증가하는 유한 범위여야 합니다.')
    axes = [lo + (np.arange(grid) + .5) * (hi-lo)/grid for lo, hi in bounds]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)


def project(points, camera):
    values = np.array([camera[key] for key in ('angle', 'cx', 'cy', 'scale')], dtype=float)
    if not np.isfinite(values).all() or values[3] <= 0:
        raise ValueError('투영값은 유한하며 scale은 양수여야 합니다.')
    angle, cx, cy, scale = values
    theta = np.deg2rad(angle)
    u = cx + scale * (points[..., 0]*np.cos(theta) - points[..., 1]*np.sin(theta))
    v = cy - scale * points[..., 2]
    return np.rint(u), np.rint(v)


def carve(masks, cameras, grid=40, bounds=((-1, 1),)*3):
    if not masks or len(masks) != len(cameras):
        raise ValueError('마스크와 카메라 설정 개수가 일치해야 합니다.')
    points = voxel_centers(grid, bounds)
    occupied = np.ones(points.shape[:-1], dtype=bool)
    counts = []
    for mask, camera in zip(masks, cameras):
        if mask.ndim != 2 or not np.isin(mask, [0, 255]).all():
            raise ValueError('마스크는 0/255 이진 단일 채널이어야 합니다.')
        u, v = project(points, camera)
        valid = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        inside = np.zeros_like(occupied)
        # 영상 범위를 검사한 점에 한해서 정수 변환 및 조회한다.
        inside[valid] = mask[v[valid].astype(int), u[valid].astype(int)] == 255
        occupied &= inside
        counts.append(int(occupied.sum()))
    return occupied, counts


def reconstruct(input_path, grid, output_path):
    source, output = Path(input_path), Path(output_path)
    config = read_json(source / 'config.json')
    if config.get('method') != 'manual_orthographic' or len(config['cameras']) != 3:
        raise ValueError('수동 직교투영 카메라 3개 설정이 필요합니다.')
    masks = []
    for i, camera in enumerate(config['cameras'], 1):
        mask = read_image(source / f'cam{i}_mask.png')
        if mask.shape != (camera['height'], camera['width']):
            raise ValueError('설정과 마스크 크기 불일치')
        masks.append(mask)
    voxels, counts = carve(masks, config['cameras'], grid, config['bounds'])
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / 'voxels.npy', voxels, allow_pickle=False)
    write_json(output / 'metadata.json', dict(config=config, grid=grid, counts=counts, input=str(source.resolve())))
    from visualize import render
    render(voxels, config['bounds'], output / 'result.png')
    print('카메라별 잔여 복셀:', counts)
    if not voxels.any():
        print('주의: 빈 결과입니다. 마스크와 중심/배율/각도를 확인하세요.')
    return voxels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--grid', type=int, default=40)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reconstruct(args.input, args.grid, args.output)
