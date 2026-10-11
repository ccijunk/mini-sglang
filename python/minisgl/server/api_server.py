from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    GaugeReply,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, cached_load_hf_config, init_logger, load_tokenizer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, ConfigDict, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None

# C2 §metrics: names/types/labels are contract (doc/contracts/server_api.md; base labels engine + model)
ENGINE_LABEL = "mini-sglang"
REQUEST_TOTAL = Counter("nano_request_total", "C2 request count", ["engine", "model", "status"])
PROMPT_TOKENS = Counter("nano_request_prompt_tokens", "C2 prompt token throughput", ["engine", "model"])
COMPLETION_TOKENS = Counter("nano_request_completion_tokens", "C2 completion token throughput", ["engine", "model"])
TTFT = Histogram("nano_ttft_seconds", "C2 time to first token", ["engine", "model"])
TPOT = Histogram("nano_tpot_seconds", "C2 time per output token", ["engine", "model"])
QUEUE_DEPTH = Gauge("nano_queue_depth", "C2 queued requests", ["engine", "model"])
RUNNING_BATCH = Gauge("nano_running_batch", "C2 requests in running batch", ["engine", "model"])
KV_UTIL = Gauge("nano_kv_cache_utilization", "C2 KV pool occupancy 0..1", ["engine", "model"])


def error_body(message: str, type_: str = "invalid_request_error", param: str | None = None, code: str | None = None):
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


class C2CompletionRequest(BaseModel):
    # C2 §completions request fields = contract; unknown fields rejected (subset semantics)
    model_config = ConfigDict(extra="forbid")
    model: str
    prompt: str
    max_tokens: int = Field(default=16, ge=1, le=256)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    seed: int | None = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    model: str

    prompt: str | None = None
    messages: List[Message] | None = None

    max_tokens: int = 16
    temperature: float = 1.0

    top_k: int = -1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

    ignore_eos: bool = False


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nano-ai-infra-stack"  # C2 §models (t4d alignment; upstream was "mini-sglang")
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)
    # latest scheduler gauge snapshot (GaugeReply stream, t4 §3 T4d)
    gauges: Dict[str, float] = field(
        default_factory=lambda: {"queue_depth": 0.0, "running_batch": 0.0, "kv_utilization": 0.0}
    )
    # C2 helpers: tokenizer for prompt-length validation, sidecar config for max context
    tokenizer: Any = None
    max_position: int = 0

    @property
    def model_name(self) -> str:
        return os.path.basename(self.config.model_path.rstrip("/")) or "nano-model"

    def update_gauges(self, msg: GaugeReply) -> None:
        self.gauges["queue_depth"] = msg.queue_depth
        self.gauges["running_batch"] = msg.running_batch
        self.gauges["kv_utilization"] = msg.kv_utilization

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            if isinstance(msg, GaugeReply):
                self.update_gauges(msg)
                continue
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        # cleanup in finally: non-streaming consumers break ON the finished yield,
        # so the generator is finalized at that yield point (GeneratorExit) and the
        # post-loop dels would never run — leaking ack_map entries and eventually
        # poisoning the 503 admission counter (t4d E-C1 incident)
        try:
            while True:
                await event.wait()
                event.clear()

                pending = self.ack_map[uid]
                self.ack_map[uid] = []
                ack = None
                for ack in pending:
                    yield ack
                if ack and ack.finished:
                    break
        finally:
            self.ack_map.pop(uid, None)
            self.event_map.pop(uid, None)

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # send final finish_reason
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))
            raise

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    # shutdown code here
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.model_name, root=state.config.model_path)])


# ---------------------------------------------------------------------------
# C2 server contract (doc/contracts/server_api.md): /v1/completions /health
# /metrics added by t4d; upstream endpoints above kept untouched as references.
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def c2_validation_error_handler(request, exc: RequestValidationError):
    REQUEST_TOTAL.labels(ENGINE_LABEL, get_global_state().model_name, "error").inc()
    first = exc.errors()[0]
    loc = ".".join(str(x) for x in first["loc"])
    return JSONResponse(status_code=400, content=error_body(f"{loc}: {first['msg']}", param=loc or None))


@app.post("/v1/completions")
async def c2_completions(req: C2CompletionRequest):
    state = get_global_state()
    model = state.model_name
    if req.model != model:
        REQUEST_TOTAL.labels(ENGINE_LABEL, model, "error").inc()
        return JSONResponse(
            status_code=404,
            content=error_body(f"The model `{req.model}` does not exist", code="model_not_found"),
        )
    # 503 admission control: in-flight requests >= max_running_req (C2 §503, autoscaler signal)
    if len(state.ack_map) >= state.config.max_running_req:
        REQUEST_TOTAL.labels(ENGINE_LABEL, model, "error").inc()
        return JSONResponse(
            status_code=503,
            content=error_body("engine overloaded, retry later", type_="server_error", code="engine_overloaded"),
        )
    try:
        prompt_ids = state.tokenizer.encode(req.prompt)
    except ValueError as e:
        REQUEST_TOTAL.labels(ENGINE_LABEL, model, "error").inc()
        return JSONResponse(
            status_code=400,
            content=error_body(f"prompt encoding failed: {e}", param="prompt"),
        )
    if len(prompt_ids) + req.max_tokens > state.max_position:
        REQUEST_TOTAL.labels(ENGINE_LABEL, model, "error").inc()
        return JSONResponse(
            status_code=422,
            content=error_body(
                f"This model's maximum context length is {state.max_position} tokens, "
                f"however you requested {len(prompt_ids) + req.max_tokens} tokens "
                f"({len(prompt_ids)} in the prompt, {req.max_tokens} for the completion)",
                code="context_length_exceeded",
            ),
        )
    uid = state.new_user()
    enqueue_t = perf_counter()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                seed=req.seed,
            ),
        )
    )
    first_t: float | None = None
    text = ""
    prompt_tokens = completion_tokens = 0
    async for ack in state.wait_for_ack(uid):
        if first_t is None:
            first_t = perf_counter()
        text += ack.incremental_output
        prompt_tokens, completion_tokens = ack.prompt_tokens, ack.completion_tokens
        if ack.finished:
            break
    finish_t = perf_counter()
    TTFT.labels(ENGINE_LABEL, model).observe(first_t - enqueue_t)
    if completion_tokens > 1:
        TPOT.labels(ENGINE_LABEL, model).observe((finish_t - first_t) / (completion_tokens - 1))
    PROMPT_TOKENS.labels(ENGINE_LABEL, model).inc(prompt_tokens)
    COMPLETION_TOKENS.labels(ENGINE_LABEL, model).inc(completion_tokens)
    REQUEST_TOTAL.labels(ENGINE_LABEL, model, "ok").inc()
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        # char model has no EOS (t4 §6.4) -> "length" is the reachable branch; "stop" kept per contract
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": "length" if completion_tokens >= req.max_tokens else "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.get("/health")
async def c2_health():
    return {"status": "ok"}


@app.get("/metrics")
async def c2_metrics():
    # gauge semantics = latest snapshot from the GaugeReply stream (t4 §3 T4d)
    state = get_global_state()
    engine, model = ENGINE_LABEL, state.model_name
    QUEUE_DEPTH.labels(engine, model).set(state.gauges["queue_depth"])
    RUNNING_BATCH.labels(engine, model).set(state.gauges["running_batch"])
    KV_UTIL.labels(engine, model).set(state.gauges["kv_utilization"])
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # then kill all the subprocesses
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )
    # C2 helpers: prompt-length validation + sidecar max context (t4 §3 T4d)
    _GLOBAL_STATE.tokenizer = load_tokenizer(config.model_path)
    _GLOBAL_STATE.max_position = int(
        getattr(cached_load_hf_config(config.model_path), "max_position_embeddings", 2048)
    )

    # start the backend here
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
