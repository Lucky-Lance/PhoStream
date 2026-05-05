#!/usr/bin/env python3
"""
HF OpenAI-compatible server.

Start a local OpenAI-compatible chat endpoint backed by HuggingFace models.
Designed for StreamEval standalone usage where we want HF inference behavior.

Example:
  python hf_openai_server.py --model-path Qwen/Qwen3-VL-8B-Instruct

Then use:
  api_base=http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def now_ts() -> int:
    return int(time.time())


def read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def write_json(handler: BaseHTTPRequestHandler, status_code: int, payload: Dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def normalize_content(content: Any) -> List[Dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = item.get("text", "")
            if text:
                out.append({"type": "text", "text": str(text)})
        elif item_type in {"video", "input_video"}:
            # stream eval uses {"type":"video","video":"..."}
            video_path = item.get("video")
            if not video_path:
                video_url = item.get("video_url")
                if isinstance(video_url, str) and video_url.startswith("data:video/"):
                    video_path = decode_data_url_to_temp_file(video_url)
                else:
                    video_path = video_url
            if video_path:
                out.append({"type": "video", "video": str(video_path), "fps": item.get("fps", 2.0)})
        elif item_type == "image_url":
            # Optional pass-through for image messages.
            image_url = item.get("image_url", {})
            if isinstance(image_url, dict) and image_url.get("url"):
                out.append({"type": "image", "image": image_url["url"]})
    return out


def normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    out: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user"))
        out.append({"role": role, "content": normalize_content(message.get("content", []))})
    return out


def decode_data_url_to_temp_file(data_url: str) -> str:
    # format: data:video/mp4;base64,xxxx
    _, payload = data_url.split(",", 1)
    raw = base64.b64decode(payload)
    fd, path = tempfile.mkstemp(suffix=".mp4", prefix="hf_openai_upload_")
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    return path


class BaseEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        raise NotImplementedError


class TextEngine(BaseEngine):
    def __init__(self, model_path: str, device_map: str, torch_dtype: str):
        super().__init__(model_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dtype = resolve_torch_dtype(torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        msgs = []
        for msg in messages:
            text_chunks = [x.get("text", "") for x in msg.get("content", []) if x.get("type") == "text"]
            text = "\n".join([x for x in text_chunks if x]).strip()
            if text:
                msgs.append({"role": msg.get("role", "user"), "content": text})

        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
            )
        input_len = inputs["input_ids"].shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


class Qwen3VLEngine(BaseEngine):
    def __init__(self, model_path: str, device_map: str, torch_dtype: str, attn_implementation: str):
        super().__init__(model_path)
        from transformers import AutoProcessor
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

        dtype = resolve_torch_dtype(torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if "A3B" in model_path:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                attn_implementation=attn_implementation,
                trust_remote_code=True,
            )
        self.model.eval()

        try:
            from qwen_vl_utils import process_vision_info
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen_vl_utils is required for Qwen3-VL models. "
                "Please install qwen-vl-utils."
            ) from exc
        self.process_vision_info = process_vision_info

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = self.process_vision_info(messages, return_video_kwargs=True)

        if video_kwargs:
            # qwen_vl_utils returns kwargs for the video processor (e.g. fps/do_sample_frames).
            # These should be passed through `videos_kwargs` instead of `video_metadata`.
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
                videos_kwargs=video_kwargs,
            ).to(self.model.device)
        else:
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
            ).to(self.model.device)

        do_sample = temperature > 0
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
            )
        input_len = inputs.input_ids.shape[1]
        trimmed = generated_ids[:, input_len:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


class Qwen3OmniEngine(BaseEngine):
    def __init__(
        self,
        model_path: str,
        device_map: str,
        torch_dtype: str,
        attn_implementation: str,
        use_audio_in_video: bool,
    ):
        super().__init__(model_path)
        from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

        dtype = resolve_torch_dtype(torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        self.model.eval()
        self.use_audio_in_video = use_audio_in_video

        try:
            from qwen_omni_utils import process_mm_info
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "qwen_omni_utils is required for Qwen3-Omni models. "
                "Please install qwen-omni-utils."
            ) from exc
        self.process_mm_info = process_mm_info

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = self.process_mm_info(
            messages, use_audio_in_video=self.use_audio_in_video
        )
        inputs = self.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=self.use_audio_in_video,
        ).to(device=self.model.device, dtype=self.model.dtype)

        do_sample = temperature > 0
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-6) if do_sample else None,
                speaker="Ethan",
                thinker_return_dict_in_generate=True,
                use_audio_in_video=self.use_audio_in_video,
            )

        if isinstance(out, tuple):
            seq = out[0].sequences if hasattr(out[0], "sequences") else out[0]
        elif hasattr(out, "sequences"):
            seq = out.sequences
        else:
            seq = out

        input_len = inputs["input_ids"].shape[1]
        trimmed = seq[:, input_len:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()


def resolve_torch_dtype(name: str) -> torch.dtype:
    text = str(name).lower()
    if text in {"auto", ""}:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if text in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if text in {"fp16", "float16"}:
        return torch.float16
    if text in {"fp32", "float32"}:
        return torch.float32
    return torch.float16 if torch.cuda.is_available() else torch.float32


def build_engine(
    model_path: str,
    model_type: str,
    device_map: str,
    torch_dtype: str,
    attn_implementation: str,
    use_audio_in_video: bool,
) -> BaseEngine:
    if model_type == "auto":
        lowered = model_path.lower()
        if "omni" in lowered:
            model_type = "qwen3_omni"
        elif "vl" in lowered:
            model_type = "qwen3_vl"
        else:
            model_type = "text"

    if model_type == "qwen3_omni":
        return Qwen3OmniEngine(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            use_audio_in_video=use_audio_in_video,
        )
    if model_type == "qwen3_vl":
        return Qwen3VLEngine(
            model_path=model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
    return TextEngine(model_path=model_path, device_map=device_map, torch_dtype=torch_dtype)


def build_openai_response(model: str, content: str) -> Dict[str, Any]:
    created = now_ts()
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class AppHandler(BaseHTTPRequestHandler):
    server_version = "hf-openai/0.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            write_json(self, HTTPStatus.OK, {"status": "ok"})
            return
        if self.path == "/v1/models":
            model_id = self.server.app_state["served_model_name"]  # type: ignore[attr-defined]
            write_json(
                self,
                HTTPStatus.OK,
                {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "local"}]},
            )
            return
        write_json(self, HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            write_json(self, HTTPStatus.NOT_FOUND, {"error": {"message": "Not Found"}})
            return

        try:
            payload = read_json_body(self)
        except Exception as exc:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": {"message": f"Invalid JSON: {exc}"}})
            return

        if payload.get("stream"):
            write_json(
                self,
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "stream=true is not supported in this server."}},
            )
            return

        model_name = str(payload.get("model") or self.server.app_state["served_model_name"])  # type: ignore[attr-defined]
        messages = normalize_messages(payload.get("messages", []))
        max_tokens = int(payload.get("max_tokens", payload.get("max_new_tokens", 1024)))
        temperature = float(payload.get("temperature", 0.0))

        if not messages:
            write_json(self, HTTPStatus.BAD_REQUEST, {"error": {"message": "messages is empty"}})
            return

        try:
            engine: BaseEngine = self.server.app_state["engine"]  # type: ignore[attr-defined]
            text = engine.generate(messages=messages, max_new_tokens=max_tokens, temperature=temperature)
            write_json(self, HTTPStatus.OK, build_openai_response(model_name, text))
        except Exception as exc:
            write_json(
                self,
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": f"Inference failed: {type(exc).__name__}: {exc}"}},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        # concise logs
        print(f"[hf-openai] {self.address_string()} - {fmt % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF OpenAI-compatible chat server")
    parser.add_argument("--model-path", required=True, help="HuggingFace model path or repo id")
    parser.add_argument("--model-type", default="auto", choices=["auto", "qwen3_omni", "qwen3_vl", "text"])
    parser.add_argument("--served-model-name", default="", help="Model name returned by /v1/models")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", help="auto|bf16|fp16|fp32")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--use-audio-in-video", type=int, default=1, help="1 for true, 0 for false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    served_model_name = args.served_model_name or Path(args.model_path).name

    print(f"[hf-openai] Loading model: {args.model_path}")
    engine = build_engine(
        model_path=args.model_path,
        model_type=args.model_type,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        use_audio_in_video=bool(args.use_audio_in_video),
    )
    print(f"[hf-openai] Loaded. Serving as model: {served_model_name}")

    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    server.app_state = {"engine": engine, "served_model_name": served_model_name}  # type: ignore[attr-defined]

    print(f"[hf-openai] Listening on http://{args.host}:{args.port}")
    print("[hf-openai] Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[hf-openai] Stopped")


if __name__ == "__main__":
    main()
