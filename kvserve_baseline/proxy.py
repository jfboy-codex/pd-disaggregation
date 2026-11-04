"""OpenAI-compatible two-stage router for KVServe's producer/consumer connector."""

from __future__ import annotations

import argparse
import logging
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from protocol import phase_payloads


logger = logging.getLogger("kvserve_proxy")


def create_app(host: str, prefill_port: int, decode_port: int, timeout: float) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        async with httpx.AsyncClient(
            base_url=f"http://{host}:{prefill_port}/v1",
            timeout=timeout,
            limits=limits,
            trust_env=False,
        ) as prefill_client, httpx.AsyncClient(
            base_url=f"http://{host}:{decode_port}/v1",
            timeout=timeout,
            limits=limits,
            trust_env=False,
        ) as decode_client:
            app.state.prefill = prefill_client
            app.state.decode = decode_client
            yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/healthcheck")
    async def healthcheck():
        return {"status": "ok"}

    async def handle(endpoint: str, request: Request):
        try:
            body = await request.json()
            transfer_id = uuid.uuid4().hex
            prefill_body, decode_body = phase_payloads(body, transfer_id)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        headers = {"X-Request-Id": f"{transfer_id}-P"}
        try:
            prefill_response = await request.app.state.prefill.post(
                endpoint, json=prefill_body, headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.exception("Prefill request failed: %s", transfer_id)
            return JSONResponse(status_code=502, content={"error": f"prefill: {exc}"})
        if prefill_response.is_error:
            logger.error("Prefill %s returned HTTP %s: %s", transfer_id,
                         prefill_response.status_code, prefill_response.text[:1000])
            return Response(
                content=prefill_response.content,
                status_code=prefill_response.status_code,
                media_type=prefill_response.headers.get("content-type", "application/json"),
            )

        headers = {"X-Request-Id": f"{transfer_id}-D"}
        if decode_body.get("stream"):
            upstream = None
            try:
                outgoing = request.app.state.decode.build_request(
                    "POST", endpoint, json=decode_body, headers=headers,
                )
                upstream = await request.app.state.decode.send(outgoing, stream=True)
                if upstream.is_error:
                    content = await upstream.aread()
                    return Response(
                        content=content, status_code=upstream.status_code,
                        media_type=upstream.headers.get("content-type", "application/json"),
                    )

                async def chunks():
                    try:
                        async for chunk in upstream.aiter_bytes():
                            yield chunk
                    finally:
                        await upstream.aclose()

                return StreamingResponse(
                    chunks(), media_type="text/event-stream",
                    background=BackgroundTask(upstream.aclose),
                )
            except httpx.HTTPError as exc:
                logger.exception("Decode stream failed: %s", transfer_id)
                return JSONResponse(status_code=502, content={"error": f"decode: {exc}"})
            finally:
                if upstream is not None and upstream.is_error:
                    await upstream.aclose()

        try:
            decode_response = await request.app.state.decode.post(
                endpoint, json=decode_body, headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.exception("Decode request failed: %s", transfer_id)
            return JSONResponse(status_code=502, content={"error": f"decode: {exc}"})
        return Response(
            content=decode_response.content,
            status_code=decode_response.status_code,
            media_type=decode_response.headers.get("content-type", "application/json"),
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await handle("/chat/completions", request)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await handle("/completions", request)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--prefill-port", type=int, default=8100)
    parser.add_argument("--decode-port", type=int, default=8200)
    parser.add_argument("--backend-timeout", type=float, default=3600)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        create_app(args.host, args.prefill_port, args.decode_port,
                   args.backend_timeout),
        host=args.host, port=args.port,
    )


if __name__ == "__main__":
    main()
