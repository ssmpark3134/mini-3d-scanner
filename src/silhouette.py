"""역이진화와 연결 성분으로 실루엣을 추출한다."""
import cv2
import numpy as np


def make_mask(frame, threshold=120, kernel=3, roi=None):
    if not 0 <= threshold <= 255 or kernel < 1 or kernel % 2 == 0:
        raise ValueError('임계값은 0~255, 커널은 양의 홀수여야 합니다.')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    if roi is not None:
        x, y, width, height = map(int, roi)
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x+width > mask.shape[1] or y+height > mask.shape[0]:
            raise ValueError('ROI가 원본 영상 범위를 벗어났습니다.')
        allowed = np.zeros_like(mask)
        allowed[y:y+height, x:x+width] = 255
        mask = cv2.bitwise_and(mask, allowed)
    element = np.ones((kernel, kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, element)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return np.zeros_like(mask)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == largest, 255, 0).astype(np.uint8)
