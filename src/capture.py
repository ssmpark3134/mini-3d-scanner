"""카메라 3대 미리보기와 원본/마스크/설정 저장."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from common import default_config, open_camera, read_json, write_image, write_json
from silhouette import make_mask


def save_capture(base, frames, masks, config):
    if len(frames) != 3 or len(masks) != 3 or any(not np.any(mask) for mask in masks):
        raise ValueError('유효한 원본과 비어 있지 않은 마스크 3개가 필요합니다.')
    target = Path(base)
    if target.exists():
        target = target.with_name(target.name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    target.mkdir(parents=True, exist_ok=False)
    saved = deepcopy(config)
    saved['captured_at'] = datetime.now().astimezone().isoformat()
    saved['synchronization'] = 'sequential USB reads; stationary object only'
    for i, (frame, mask, camera) in enumerate(zip(frames, masks, saved['cameras']), 1):
        if frame.shape[:2] != mask.shape:
            raise ValueError('원본과 마스크 크기 불일치')
        camera.update(width=frame.shape[1], height=frame.shape[0])
        write_image(target / f'cam{i}.png', frame)
        write_image(target / f'cam{i}_mask.png', mask)
    # 설정 파일이 마지막에 있어야 완성된 촬영 묶음으로 처리한다.
    write_json(target / 'config.json', saved)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--output', default='captures/scan_001')
    parser.add_argument('--config', help='카메라별 각도/중심/배율/ROI/초점 설정 JSON')
    parser.add_argument('--write-config', help='기본 설정만 생성하고 종료')
    args = parser.parse_args()
    config = read_json(args.config) if args.config else default_config(args.cameras, args.width, args.height)
    if len(config['cameras']) != 3 or len(set(args.cameras)) != 3:
        raise ValueError('서로 다른 장치 3개와 설정 3개가 필요합니다.')
    if args.write_config:
        if Path(args.write_config).exists():
            raise FileExistsError(args.write_config)
        write_json(args.write_config, config)
        return
    cameras = []
    try:
        for device, setting in zip(args.cameras, config['cameras']):
            camera = open_camera(device, args.width, args.height)
            cameras.append(camera)
            setting['device'] = device
            mode = setting.get('focus_mode', 'unknown')
            accepted = None
            if mode in ('manual', 'auto'):
                accepted = bool(camera.set(cv2.CAP_PROP_AUTOFOCUS, int(mode == 'auto')))
                if mode == 'manual' and setting.get('focus_value') is not None:
                    accepted = bool(camera.set(cv2.CAP_PROP_FOCUS, setting['focus_value'])) and accepted
            setting['focus_request_accepted'] = accepted
            setting['focus_readback_raw'] = camera.get(cv2.CAP_PROP_FOCUS)
            print(f"장치 {device}: 초점 설정 요청 결과={accepted}; 실제 선명도는 화면에서 확인하세요.")
        cv2.namedWindow('Scanner')
        for i, setting in enumerate(config['cameras'], 1):
            cv2.createTrackbar(f'Threshold CAM{i}', 'Scanner', setting['threshold'], 255, lambda value: None)
        while True:
            frames, masks, panels = [], [], []
            for i, (camera, setting) in enumerate(zip(cameras, config['cameras']), 1):
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError(f'CAM{i} 프레임 읽기 실패')
                height, width = frame.shape[:2]
                if 'width' in setting and (setting['width'], setting['height']) != (width, height):
                    raise ValueError('설정 영상 크기와 실제 크기가 다릅니다. 중심/배율/ROI를 다시 맞추세요.')
                setting.update(width=width, height=height)
                setting['threshold'] = cv2.getTrackbarPos(f'Threshold CAM{i}', 'Scanner')
                mask = make_mask(frame, setting['threshold'], setting.get('kernel', 3), setting.get('roi'))
                frames.append(frame)
                masks.append(mask)
                preview = frame.copy()
                cv2.drawMarker(preview, (round(setting['cx']), round(setting['cy'])), (0, 0, 255))
                label = f"CAM{i} device={setting['device']} {width}x{height}"
                if not mask.any():
                    label += ' EMPTY'
                elif np.any(mask[0]) or np.any(mask[-1]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
                    label += ' BORDER'
                cv2.putText(preview, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                size = (400, max(1, round(height * 400 / width)))
                panels.append(np.vstack([cv2.resize(preview, size), cv2.cvtColor(cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)]))
            panel_height = max(panel.shape[0] for panel in panels)
            panels = [cv2.copyMakeBorder(panel, 0, panel_height-panel.shape[0], 0, 0, cv2.BORDER_CONSTANT) for panel in panels]
            cv2.imshow('Scanner', np.hstack(panels))
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty('Scanner', cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('s'), ord('S')):
                try:
                    print('저장 완료:', save_capture(args.output, frames, masks, config))
                except (OSError, ValueError) as error:
                    print('저장 실패:', error)
    finally:
        for camera in cameras:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
