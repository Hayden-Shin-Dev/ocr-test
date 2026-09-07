from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Runtime settings for local OCR inference."""

    project_root: Path = PROJECT_ROOT
    source_root: Path = PROJECT_ROOT / "data" / "01.원천데이터"
    output_dir: Path = Path(os.getenv("OCR_OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
    paddle_lang: str = os.getenv("PADDLEOCR_LANG", "korean")
    paddle_device: str = os.getenv("PADDLEOCR_DEVICE", "cpu")
    max_upload_mb: int = int(os.getenv("OCR_MAX_UPLOAD_MB", "20"))
    host: str = os.getenv("OCR_HOST", "127.0.0.1")
    port: int = int(os.getenv("OCR_PORT", "8000"))


settings = Settings()
