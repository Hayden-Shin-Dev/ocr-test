# ocr-test

물류 문서 이미지에서 텍스트를 추출하고 결과를 확인하는 로컬 OCR 웹 앱입니다.

현재 앱은 특정 문서 양식이나 고정 좌표를 사용하지 않습니다. 이미지 전체에서 텍스트 영역을 검출하고, 인식된 라인을 좌표·신뢰도와 함께 표시합니다.

## 지원 입력

- PNG, JPG, JPEG, WEBP, BMP, TIFF
- 로컬 샘플: `data/01.원천데이터` 아래의 모든 이미지
- 업로드 이미지: 기본 최대 20MB

## 설치

Windows PowerShell 기준:

```powershell
cd "C:\Users\shinm\Desktop\새 폴더 (2)"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

처음 PaddleOCR를 실행하면 공식 OCR 모델이 로컬 캐시에 다운로드됩니다. CPU 환경은 기본값이고, NVIDIA GPU 환경에서는 `.env` 또는 환경 변수로 `PADDLEOCR_DEVICE=gpu:0`을 지정할 수 있습니다.

## 실행

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn ocr_test.app:app --host 127.0.0.1 --port 8000
```

브라우저에서 <http://127.0.0.1:8000>을 엽니다. 샘플을 고르거나 이미지를 업로드한 뒤 `텍스트 추출`을 누르면 원본·전체 텍스트·라인별 신뢰도를 한 화면에서 확인할 수 있습니다.

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 구성

- `src/ocr_test/ocr_engine.py`: PaddleOCR 지연 초기화 및 결과 표준화
- `src/ocr_test/app.py`: 샘플 목록, 이미지 제공, 업로드·OCR API
- `src/ocr_test/web/`: 로컬 결과 대시보드
- `data/`: 로컬 데이터셋이며 Git에는 포함하지 않음

