"""합성 상자 실루엣을 생성한다. 실제 촬영 결과가 아니다."""
import argparse
import numpy as np
from common import default_config
from capture import save_capture


def box_inputs():
    config = default_config([0, 1, 2], 320, 240)
    config['input_type'] = 'synthetic_box'
    masks = []
    for camera in config['cameras']:
        theta = np.deg2rad(camera['angle'])
        half_width = .5*abs(np.cos(theta)) + .3*abs(np.sin(theta))
        v, u = np.indices((240, 320))
        mask = (abs(u-camera['cx']) <= camera['scale']*half_width) & (abs(v-camera['cy']) <= camera['scale']*.6)
        masks.append(mask.astype(np.uint8)*255)
    frames = [np.repeat((255-mask)[..., None], 3, axis=2) for mask in masks]
    return frames, masks, config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='captures/synthetic_box')
    args = parser.parse_args()
    print(save_capture(args.output, *box_inputs()))
