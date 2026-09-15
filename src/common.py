"""파일 입출력과 공통 설정. 원본 좌표를 유지한다."""
import json
from pathlib import Path
import cv2
import numpy as np


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"이미지 읽기 실패: {path}")
    return image


def write_image(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise OSError(f"이미지 인코딩 실패: {path}")
    encoded.tofile(path)
    if not np.array_equal(read_image(path), image):
        raise OSError(f"저장 검증 실패: {path}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def default_config(devices, width, height):
    return {'method': 'manual_orthographic', 'input_type': 'real',
            'bounds': [[-1, 1], [-1, 1], [-1, 1]],
            'cameras': [dict(name=f'cam{i+1}', device=device, angle=angle,
                             cx=width/2, cy=height/2, scale=min(width, height)/3,
                             threshold=120, kernel=3, roi=None,
                             focus_mode='unknown', focus_value=None,
                             sharpness_confirmed=False)
                        for i, (device, angle) in enumerate(zip(devices, [-45, 0, 45]))]}


def open_camera(device, width=640, height=480):
    camera = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        camera = cv2.VideoCapture(device, backend)
        if camera.isOpened():
            break
        camera.release()
    if not camera.isOpened():
        raise RuntimeError(f'카메라 {device} 열기 실패')
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera
