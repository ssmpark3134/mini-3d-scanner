"""장치 번호별 프레임 읽기 확인. 실제 위치는 렌즈를 가려 확인한다."""
import argparse
import cv2
from common import open_camera, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', nargs='+', type=int, default=list(range(6)))
    parser.add_argument('--report', default='camera_report.json')
    args = parser.parse_args()
    results = []
    for device in args.devices:
        camera = None
        entry = {'device': device, 'read_ok': False}
        try:
            camera = open_camera(device)
            for _ in range(15):
                ok, frame = camera.read()
                if ok and frame is not None:
                    entry.update(read_ok=True, width=frame.shape[1], height=frame.shape[0])
                    break
        except RuntimeError as error:
            entry['error'] = str(error)
        finally:
            if camera is not None:
                camera.release()
        results.append(entry)
        print(entry)
    write_json(args.report, results)
    return 0 if sum(item['read_ok'] for item in results) >= 3 else 1


if __name__ == '__main__':
    raise SystemExit(main())
