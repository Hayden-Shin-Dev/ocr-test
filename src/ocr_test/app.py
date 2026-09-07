from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .ocr_engine import OCRDocument, engine


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
WEB_ROOT = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Logistics OCR Lab", version="0.1.0")
if (WEB_ROOT / "static").is_dir():
    app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")


class SampleRequest(BaseModel):
    path: str


def _sample_root() -> Path:
    return settings.source_root.resolve()


def _resolve_sample(relative_path: str) -> Path:
    root = _sample_root()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents or candidate.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Invalid sample path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Sample image not found")
    return candidate


def _sample_items() -> list[dict[str, str]]:
    root = _sample_root()
    if not root.is_dir():
        return []
    items: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            relative = path.relative_to(root).as_posix()
            items.append({
                "path": relative,
                "name": path.name,
                "document_type": path.parent.name,
                "url": f"/api/sample-image?path={relative}",
            })
    return items


def _document_response(document: OCRDocument, source: str) -> dict[str, Any]:
    response = document.as_dict()
    response["source"] = source
    return response


def _validate_upload(data: bytes, filename: str | None) -> str:
    suffix = Path(filename or "upload.png").suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="PNG, JPG, JPEG, WEBP, BMP, or TIFF images are supported")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image") from exc
    return suffix


async def _run_path(path: Path, source: str) -> dict[str, Any]:
    try:
        document = await run_in_threadpool(engine.predict, path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"OCR inference failed: {exc}") from exc
    return _document_response(document, source)


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "engine": "paddleocr"}


@app.get("/api/samples")
def samples() -> dict[str, Any]:
    items = _sample_items()
    return {"count": len(items), "items": items}


@app.get("/api/sample-image")
def sample_image(path: str) -> FileResponse:
    image_path = _resolve_sample(path)
    return FileResponse(image_path)


@app.post("/api/ocr/sample")
async def ocr_sample(request: SampleRequest) -> dict[str, Any]:
    image_path = _resolve_sample(request.path)
    return await _run_path(image_path, source=f"sample:{request.path}")


@app.post("/api/ocr")
async def ocr_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Maximum upload size is {settings.max_upload_mb} MB")

    suffix = _validate_upload(data, file.filename)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=settings.output_dir, suffix=suffix, delete=False) as temp_file:
        temp_file.write(data)
        temp_path = Path(temp_file.name)
    try:
        return await _run_path(temp_path, source=f"upload:{file.filename or temp_path.name}")
    finally:
        temp_path.unlink(missing_ok=True)
