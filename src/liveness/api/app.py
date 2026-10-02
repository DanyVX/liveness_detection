"""FastAPI app factory.

Data minimisation: uploads are decoded in memory and discarded after the request. Nothing is
written to disk, and logs carry only request id, route, status, decision and reason code, never
image bytes, filenames or scores.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import FastAPI, File, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from liveness.api.schemas import DecisionResponse, Health, ModelInfo
from liveness.api.settings import ApiSettings
from liveness.gate import GateConfig, GateResult, OnnxLivenessGate
from liveness.infer import (
    FaceDetector,
    FrameScorer,
    HaarFaceDetector,
    OnnxScorer,
    ReasonCode,
    Verdict,
)

logger = logging.getLogger("liveness.api")

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_MULTIPART_OVERHEAD = 64 * 1024


def create_app(
    settings: ApiSettings | None = None,
    scorer: OnnxScorer | None = None,
    detector: FaceDetector | None = None,
) -> FastAPI:
    """Fails fast (FileNotFoundError / ValidationError) if the model or settings are bad."""
    cfg = settings if settings is not None else ApiSettings()  # type: ignore[call-arg]
    if scorer is None:
        scorer = OnnxScorer(cfg.model_path, providers=cfg.providers)
    frame_scorer = FrameScorer(
        scorer,
        detector if detector is not None else HaarFaceDetector(),
        min_face_px=cfg.min_face_px,
        margin=cfg.margin,
    )
    gate = OnnxLivenessGate(
        frame_scorer,
        GateConfig(
            decision_threshold=cfg.decision_threshold,
            clip_accept_threshold=cfg.clip_accept_threshold,
            clip_reject_threshold=cfg.clip_reject_threshold,
            clip_min_valid_frames=cfg.clip_min_valid_frames,
            clip_aggregation=cfg.clip_aggregation,
            max_clip_frames=cfg.max_batch_frames,
        ),
    )
    version = scorer.model_version
    app = FastAPI(title="liveness", version="0.0.1")

    def error(status: int, reason: ReasonCode, detail: str) -> JSONResponse:
        body = DecisionResponse(
            score=None,
            decision=Verdict.ERROR.value,
            reason_code=reason.value,
            model_version=version,
            detail=detail,
        )
        return JSONResponse(status_code=status, content=body.model_dump())

    @app.middleware("http")
    async def correlation(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        rid = incoming if _REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        start = time.perf_counter()
        limit = cfg.max_batch_frames * cfg.max_image_bytes + _MULTIPART_OVERHEAD
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = 0
        if length > limit:
            response: Response = error(413, ReasonCode.INVALID_INPUT, "request body too large")
        else:
            try:
                response = await call_next(request)
            except Exception:
                logger.exception("unhandled error", extra={"correlation_id": rid})
                response = error(500, ReasonCode.INVALID_INPUT, "internal error")
        response.headers[REQUEST_ID_HEADER] = rid
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
            extra={"correlation_id": rid},
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return error(422, ReasonCode.INVALID_INPUT, "invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error(exc.status_code, ReasonCode.INVALID_INPUT, "request rejected")

    def respond(result: GateResult, n_frames: int | None = None) -> JSONResponse:
        body = DecisionResponse(
            score=result.score,
            decision=result.decision,
            reason_code=result.reason_code,
            model_version=result.model_version,
            n_frames=n_frames,
        )
        status = 422 if result.reason_code == ReasonCode.INVALID_INPUT.value else 200
        return JSONResponse(status_code=status, content=body.model_dump())

    def read_limited(upload: UploadFile) -> bytes | None:
        data = upload.file.read(cfg.max_image_bytes + 1)
        return None if len(data) > cfg.max_image_bytes else data

    @app.get("/healthz", response_model=Health)
    def healthz() -> Health:
        return Health(status="ok", model_version=version)

    @app.get("/v1/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return ModelInfo(
            model_version=version,
            input_size=list(scorer.input_size),
            providers=scorer.providers,
            decision_threshold=cfg.decision_threshold,
            clip_accept_threshold=cfg.clip_accept_threshold,
            clip_reject_threshold=cfg.clip_reject_threshold,
            max_batch_frames=cfg.max_batch_frames,
            max_image_bytes=cfg.max_image_bytes,
        )

    @app.post("/v1/score", response_model=DecisionResponse)
    def score(image: Annotated[UploadFile, File()]) -> Response:
        data = read_limited(image)
        if data is None:
            return error(413, ReasonCode.INVALID_INPUT, "image exceeds max_image_bytes")
        return respond(gate.check_image(data))

    @app.post("/v1/decide", response_model=DecisionResponse)
    def decide(frames: Annotated[list[UploadFile], File()]) -> Response:
        if len(frames) > cfg.max_batch_frames:
            return error(413, ReasonCode.INVALID_INPUT, "too many frames")
        blobs: list[bytes] = []
        for f in frames:
            data = read_limited(f)
            if data is None:
                return error(413, ReasonCode.INVALID_INPUT, "frame exceeds max_image_bytes")
            blobs.append(data)
        return respond(gate.check_clip(blobs), n_frames=len(blobs))

    return app
