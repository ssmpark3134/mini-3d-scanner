# 카메라 3대 간이 3D 스캐너

Python, OpenCV, NumPy, Matplotlib으로 정지 물체의 실루엣을 추출하고 수동 직교투영으로 근사 Visual Hull을 만든다.

현재 코드 구현과 합성 테스트는 완료했다. 실제 카메라 3대 연결·초점·촬영·복원 검증은 남아 있다.

## 집에서 시작하기 (Windows PowerShell)

1. Python 3.13과 Git을 설치한다.
2. 이 저장소를 `git clone 저장소주소`로 내려받거나 GitHub의 Download ZIP으로 받고 압축을 푼다.
3. 내려받은 프로젝트 폴더에서 PowerShell을 열고 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

테스트 결과 `Ran 7 tests`와 `OK`를 확인한다. 기존 PC의 `.runtime` 실행 환경은 Git에 포함하지 않는다.

## 테스트 순서

[테스트 방법과 순서](docs/테스트방법.md)의 A(카메라 없음) → B(카메라 연결 후) 순서로 진행한다. **집에서는 문서의 `.\.runtime\python\python.exe`를 `.\.venv\Scripts\python.exe`로 바꾸고, 작업 경로도 내려받은 폴더로 바꾼다.**

- [사용법과 설정](docs/사용법.md)
- [진행 방법과 구현 상태](docs/진행방법.md)
- [프로젝트 계획](PROJECT_PLAN.md)

촬영 데이터 `captures/`, 복원 결과 `outputs/`, Python 환경은 Git에서 제외된다. 합성 데이터는 `src/synthetic_demo.py`로 다시 만들 수 있다. 실제 촬영 데이터를 옮기려면 별도로 복사한다. 별도 실험 폴더 `tumblertest/`는 이 스캐너 저장 범위에 포함하지 않는다.

수동 근사 결과이며 정밀 치수 측정이나 내부·오목면 복원용이 아니다.
