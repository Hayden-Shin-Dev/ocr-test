# ocr-test

OCR 기능을 붙이기 위한 최소 Python 프로젝트입니다.

## 설치 및 실행

```powershell
python -m pip install -e .
python -m ocr_test
```

현재는 프로젝트 초기화 상태를 확인하는 CLI만 포함되어 있습니다. OCR 엔진과 이미지 입력 처리는 다음 단계에서 추가하면 됩니다.

## 테스트

```powershell
python -m unittest discover -s tests -v
```
