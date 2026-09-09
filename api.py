from __future__ import annotations

import logging
import os
from asyncio import Lock
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths

logger = logging.getLogger(__name__)

MODEL_ROOT = Path(os.getenv("LTX_MODEL_ROOT", "/models/ltx-2.5"))
OUTPUT_ROOT = Path(os.getenv("LTX_OUTPUT_ROOT", "/app/output"))
pipeline: DistilledPipeline | None = None
generation_lock = Lock()


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int = 42
    num_frames: int = Field(default=121, ge=9)
    frame_rate: float = Field(default=24, gt=0)
    height: int = Field(default=1536, gt=0)
    width: int = Field(default=1024, gt=0)


def model_path(*parts: str) -> str:
    return str(MODEL_ROOT.joinpath(*parts))


def load_pipeline() -> DistilledPipeline:
    paths = ModelPaths.from_split(
        transformer_path=model_path(
            "diffusion_models",
            "ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        ),
        text_encoder_path=model_path(
            "text_encoders",
            "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        ),
        video_vae_path=model_path("vae", "ltx-2.5-video-vae-bf16.safetensors"),
        audio_vae_path=model_path("vae", "ltx-2.5-audio-vae-bf16.safetensors"),
    )
    return DistilledPipeline(
        model_paths=paths,
        spatial_upsampler_path=model_path(
            "latent_upscale_models",
            "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        ),
        loras=[],
    )


@torch.inference_mode()
def generate_video(request: GenerationRequest, output_path: Path) -> None:
    if pipeline is None:
        raise RuntimeError("The model pipeline is not loaded")

    result = pipeline(
        prompt=request.prompt,
        seed=request.seed,
        height=request.height,
        width=request.width,
        num_frames=request.num_frames,
        frame_rate=request.frame_rate,
        images=[],
        tiling_config=AUTO_TILING,
    )
    encode_video(
        video=result.video,
        fps=int(request.frame_rate),
        audio=result.audio,
        output_path=str(output_path),
        video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading LTX-2 pipeline from %s", MODEL_ROOT)
    pipeline = load_pipeline()
    logger.info("LTX-2 pipeline loaded")
    yield
    pipeline = None


app = FastAPI(title="LTX-2 Inference API", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok" if pipeline is not None else "starting"}


@app.post("/v1/generations", response_class=FileResponse)
async def generations(request: GenerationRequest) -> FileResponse:
    output_path = OUTPUT_ROOT / f"{uuid4()}.mp4"
    async with generation_lock:
        try:
            await run_in_threadpool(generate_video, request, output_path)
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            logger.exception("Video generation failed")
            raise HTTPException(status_code=500, detail="Video generation failed") from exc

    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="result.mp4",
        background=BackgroundTask(output_path.unlink, missing_ok=True),
    )