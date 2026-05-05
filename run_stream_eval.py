#!/usr/bin/env python3
"""
Self-contained StreamEval runner.

Design goals:
- Keep old inference/scoring semantics.
- Run from a standalone package (no imports from repo root modules).
- Support both old and new benchmark formats.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from tqdm import tqdm

try:
    from json_repair import repair_json
except Exception:  # pragma: no cover
    repair_json = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "config.yaml"
PLACEHOLDERS_LOWER: set[str] = set()
FORWARD_TASK_TYPES: set[str] = {"forward", "future", "proactive"}


ENV_PATTERN = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::(.*))?\}$")


def resolve_env_refs(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: resolve_env_refs(v) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_env_refs(v) for v in value]
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            env_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.getenv(env_name, default)
    return value


def normalize_api_base(url: str, default_scheme: str = "https") -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if text.startswith("http://") or text.startswith("https://"):
        return text.rstrip("/")
    return f"{default_scheme}://{text}".rstrip("/")


def init_runtime_constants(config: Dict[str, Any]) -> None:
    global PLACEHOLDERS_LOWER, FORWARD_TASK_TYPES
    scoring_cfg = config.get("scoring", {}) if isinstance(config, dict) else {}

    placeholders = scoring_cfg.get("placeholder_responses", [])
    if not isinstance(placeholders, list):
        raise ValueError("config.scoring.placeholder_responses must be a list")
    PLACEHOLDERS_LOWER = {str(x).strip().lower() for x in placeholders}

    forward_types = scoring_cfg.get("forward_task_types", ["forward", "future", "proactive"])
    if not isinstance(forward_types, list):
        raise ValueError("config.scoring.forward_task_types must be a list")
    normalized = {str(x).strip().lower() for x in forward_types if str(x).strip()}
    FORWARD_TASK_TYPES = normalized or {"forward", "future", "proactive"}


def load_release_config(config_path: Optional[Path]) -> Tuple[Dict[str, Any], Path]:
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    cfg = resolve_env_refs(cfg)
    required_keys = ["default_benchmarks", "default_prompt", "benchmarks", "prompts", "models", "judger", "scoring"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    return cfg, config_path


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_placeholder(text: str) -> bool:
    return (text or "").strip().lower() in PLACEHOLDERS_LOWER


def time_to_seconds(t: str) -> int:
    parts = str(t).strip().split(":")
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        m, s = map(int, parts)
        return m * 60 + s
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    raise ValueError(f"Invalid time format: {t}")


def seconds_to_time(sec: int) -> str:
    sec = int(sec)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h <= 0:
        return f"{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def probe_video(video_path: str) -> Optional[Dict[str, Any]]:
    path = Path(video_path)
    if not path.exists():
        return None
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_name,codec_type,profile,level,bit_rate,width,height,r_frame_rate,nb_frames,sample_rate,channels,channel_layout",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(out.stdout)
    except Exception:
        return None
    info = {
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "file_size": int(data.get("format", {}).get("size", path.stat().st_size)),
        "total_bitrate": int(data.get("format", {}).get("bit_rate", 0)),
        "video": None,
        "audio": None,
    }
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info["video"] is None:
            fps = 0.0
            fps_raw = stream.get("r_frame_rate", "0/1")
            try:
                num, den = fps_raw.split("/")
                fps = float(num) / float(den) if float(den) != 0 else 0.0
            except Exception:
                fps = 0.0
            info["video"] = {
                "index": stream.get("index", 0),
                "codec": stream.get("codec_name", ""),
                "profile": stream.get("profile", ""),
                "level": stream.get("level", 0),
                "width": int(stream.get("width", 0) or 0),
                "height": int(stream.get("height", 0) or 0),
                "fps": fps,
                "total_frames": int(stream.get("nb_frames", 0) or 0),
                "bitrate": int(stream.get("bit_rate", 0) or 0),
            }
        elif stream.get("codec_type") == "audio" and info["audio"] is None:
            info["audio"] = {
                "index": stream.get("index", 1),
                "codec": stream.get("codec_name", ""),
                "sample_rate": int(stream.get("sample_rate", 0) or 0),
                "channels": int(stream.get("channels", 0) or 0),
                "channel_layout": stream.get("channel_layout", ""),
                "bitrate": int(stream.get("bit_rate", 0) or 0),
            }
    return info


def trim_video(
    video_path: str,
    trim_path: str,
    start_time: str,
    end_time: str,
    fps: Optional[float] = None,
) -> None:
    out_path = Path(trim_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return
    start_sec = max(0.0, float(time_to_seconds(start_time)))
    end_sec = max(start_sec + 1.0, float(time_to_seconds(end_time)))
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
    ]
    if fps is not None:
        cmd.extend(["-r", str(fps)])
    cmd.append(str(out_path))
    subprocess.run(cmd, check=True)


def merge_videos(clip_list: List[str], output_path: str) -> str:
    """Merge multiple video clips using ffmpeg concat demuxer (stream copy, no re-encoding)."""
    if not clip_list:
        raise ValueError("clip_list is empty")
    if len(clip_list) == 1:
        return clip_list[0]
    out = Path(output_path)
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for clip in clip_list:
            f.write(f"file '{os.path.abspath(clip)}'\n")
        list_file = f.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-movflags", "+faststart",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        os.unlink(list_file)
    return str(out)


def normalize_time_type_for_new_format(time_type: Any) -> str:
    value = str(time_type or "backward").strip().lower()
    if value in FORWARD_TASK_TYPES:
        return "forward"
    if value == "instant":
        return "instant"
    return "backward"


def normalize_task_type_preserving_legacy(task_type: Any, default: str = "DefaultType") -> str:
    if task_type is None:
        return default
    text = str(task_type).strip()
    if not text:
        return default
    if text.lower() == "forward":
        return "forward"
    return text


def is_forward_task(task_type: Any) -> bool:
    return str(task_type or "").strip().lower() in FORWARD_TASK_TYPES


def ensure_uuid(seed: str) -> str:
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]


def resolve_path(path_value: str, base_dirs: List[Path]) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    for base in base_dirs:
        maybe = base / candidate
        if maybe.exists():
            return maybe.resolve()
    return (base_dirs[0] / candidate).resolve()


def resolve_video_path(
    raw_video_path: str,
    benchmark_path: Path,
    video_root: Optional[Path],
) -> str:
    candidate = Path(raw_video_path)
    if candidate.is_absolute():
        return str(candidate)
    base_dirs = []
    if video_root is not None:
        base_dirs.append(video_root)
    base_dirs.extend([benchmark_path.parent, SCRIPT_DIR, Path.cwd()])
    return str(resolve_path(raw_video_path, base_dirs))


def convert_verified_responses_to_sqa(verified_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sqa: List[Dict[str, Any]] = []
    event_id = 1
    for qa in verified_responses:
        task_type = normalize_time_type_for_new_format(qa.get("time_type"))
        timestamp_question = qa.get("timestamp_question")
        question = qa.get("user_query", "")
        response = qa.get("response", "")
        capability = qa.get("capability")
        options = qa.get("options")

        if task_type == "forward":
            sqa.append(
                {
                    "event_id": event_id,
                    "timestamp": timestamp_question,
                    "type": task_type,
                    "question": question,
                    **({"capability": capability} if capability else {}),
                    **({"options": options} if options is not None else {}),
                }
            )
            event_id += 1
            sqa.append(
                {
                    "event_id": event_id,
                    "timestamp": qa.get("timestamp_proactive", timestamp_question),
                    "response": response,
                    **({"capability": capability} if capability else {}),
                }
            )
            event_id += 1
        else:
            sqa.append(
                {
                    "event_id": event_id,
                    "timestamp": timestamp_question,
                    "type": task_type,
                    "question": question,
                    "response": response,
                    **({"capability": capability} if capability else {}),
                    **({"options": options} if options is not None else {}),
                }
            )
            event_id += 1
    return sqa


def infer_duration(video_info: Optional[Dict[str, Any]], sqa: List[Dict[str, Any]]) -> float:
    if isinstance(video_info, dict) and video_info.get("duration"):
        return float(video_info["duration"])
    if sqa:
        timestamps = [time_to_seconds(item.get("timestamp", "00:00")) for item in sqa if item.get("timestamp")]
        if timestamps:
            return float(max(timestamps) + 10)
    return 0.0


def normalize_sample(
    sample: Dict[str, Any],
    idx: int,
    bench_name: str,
    benchmark_path: Path,
    video_root: Optional[Path],
    stream_addr_root: Optional[Path],
    need_video_info: bool,
) -> Dict[str, Any]:
    record = dict(sample)
    raw_video_path = record.get("video") or record.get("video_path", "")
    record["video"] = resolve_video_path(raw_video_path, benchmark_path, video_root) if raw_video_path else ""

    if "sqa" not in record and "verified_responses" in record:
        record["sqa"] = convert_verified_responses_to_sqa(record.get("verified_responses", []))
    else:
        record["sqa"] = record.get("sqa", [])

    if "id" not in record:
        record["id"] = idx + 1
    if "uuid" not in record:
        record["uuid"] = ensure_uuid(f"{bench_name}:{record['id']}:{record.get('video','')}")
    if "stream_addr" not in record:
        base = stream_addr_root if stream_addr_root is not None else (SCRIPT_DIR / "cache" / "stream_addr" / bench_name)
        record["stream_addr"] = str(base / record["uuid"])

    if need_video_info:
        current_video_info = record.get("video_info") if isinstance(record.get("video_info"), dict) else None
        if not current_video_info:
            probed = probe_video(record["video"]) if record.get("video") else None
            record["video_info"] = probed or {"duration": infer_duration(None, record["sqa"])}
        else:
            if not current_video_info.get("duration"):
                current_video_info["duration"] = infer_duration(current_video_info, record["sqa"])
            record["video_info"] = current_video_info
    else:
        record.setdefault("video_info", {"duration": infer_duration(None, record["sqa"])})

    record.setdefault("source", record.get("video_path", record.get("video", "")))
    return record


def load_samples_any_format(
    benchmark_path: Path,
    bench_name: str,
    video_root: Optional[Path],
    stream_addr_root: Optional[Path],
    need_video_info: bool,
) -> List[Dict[str, Any]]:
    if benchmark_path.suffix.lower() == ".jsonl":
        rows = load_jsonl(benchmark_path)
    else:
        raw = load_json(benchmark_path)
        if isinstance(raw, list):
            rows = raw
        elif isinstance(raw, dict) and isinstance(raw.get("samples"), list):
            rows = raw["samples"]
        else:
            raise ValueError(f"Unsupported benchmark format in {benchmark_path}")

    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        normalized.append(
            normalize_sample(
                sample=row,
                idx=idx,
                bench_name=bench_name,
                benchmark_path=benchmark_path,
                video_root=video_root,
                stream_addr_root=stream_addr_root,
                need_video_info=need_video_info,
            )
        )
    return normalized


class OpenAICompatibleBackend:
    def __init__(self, name: str, cfg: Dict[str, Any]):
        self.name = name
        self.api_base = normalize_api_base(str(cfg.get("api_base", "")))
        self.api_key = str(cfg.get("api_key", ""))
        self.model = str(cfg.get("model", ""))
        self.timeout = float(cfg.get("timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.video_transport = str(cfg.get("video_transport", "path"))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _extract_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        chunks.append(text)
            return "\n".join(chunks)
        return ""

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue
            out_content = []
            for item in content:
                t = item.get("type")
                if t == "text":
                    if item.get("text"):
                        out_content.append({"type": "text", "text": item["text"]})
                elif t == "video":
                    video_path = str(item.get("video", ""))
                    if self.video_transport == "base64_data_url":
                        with open(video_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode("utf-8")
                        out_content.append({"type": "input_video", "video_url": f"data:video/mp4;base64,{b64}"})
                    else:
                        entry: Dict[str, Any] = {
                            "type": "video",
                            "video": video_path,
                            "fps": float(item.get("fps", 2.0)),
                        }
                        if "max_frames" in item:
                            entry["max_frames"] = item["max_frames"]
                        if "max_pixels" in item:
                            entry["max_pixels"] = item["max_pixels"]
                        out_content.append(entry)
            if out_content:
                converted.append({"role": role, "content": out_content})
        return converted

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        if not self.api_base:
            return {"response": "[ERROR] Missing api_base", "raw_response": "", "status_code": 500}
        if not self.model:
            return {"response": "[ERROR] Missing model name", "raw_response": "", "status_code": 500}

        payload = {
            "model": self.model,
            "messages": self._normalize_messages(messages),
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": int(max_new_tokens or self.max_new_tokens),
        }
        url = f"{self.api_base}/chat/completions"
        headers = self._build_headers()
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                body = resp.text
                if resp.status_code == 200:
                    parsed = resp.json()
                    choices = parsed.get("choices", [])
                    if not choices:
                        return {"response": "[ERROR] Empty choices", "raw_response": body, "status_code": 502}
                    msg = choices[0].get("message", {})
                    text = self._extract_text(msg.get("content", ""))
                    return {"response": text.strip(), "raw_response": body, "status_code": 200}
                last_err = f"HTTP {resp.status_code}: {body[:400]}"
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.5 * (2**attempt))
        return {"response": f"[ERROR] {last_err}", "raw_response": last_err, "status_code": 502}


class GeminiNativeBackend:
    """
    Gemini native format backend:
    POST /v1beta/models/{model}:generateContent?key=...
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        self.name = name
        self.api_base = normalize_api_base(
            str(cfg.get("api_base", "https://generativelanguage.googleapis.com"))
        )
        self.api_key = str(cfg.get("api_key", ""))
        self.model = str(cfg.get("model", ""))
        self.timeout = float(cfg.get("timeout", 120))
        self.max_retries = int(cfg.get("max_retries", 3))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.max_video_size_mb = float(cfg.get("max_video_size_mb", 30))
        self.video_metadata_fps = float(cfg.get("video_metadata_fps", 1.0))

    def _is_google_native(self) -> bool:
        return "googleapis.com" in self.api_base

    def _build_url(self) -> str:
        if not self.api_key:
            return ""
        path = f"/models/{self.model}:generateContent"
        base = self.api_base if "/v1beta" in self.api_base else f"{self.api_base}/v1beta"
        if self._is_google_native():
            return f"{base}{path}?key={self.api_key}"
        return f"{base}{path}"

    def _video_to_base64(self, video_path: str) -> str:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        max_bytes = int(self.max_video_size_mb * 1024 * 1024)
        if path.stat().st_size > max_bytes:
            raise ValueError(f"Video too large: {video_path} ({path.stat().st_size} bytes > {max_bytes})")
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _merge_context_videos(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive video chunks across messages to reduce video count.

        Mirrors the MergeChunk._build_context() pattern from the original library:
        accumulate video-only messages, then merge them into one file when a text
        message is encountered.
        """
        merged: List[Dict[str, Any]] = []
        pending_videos: List[Dict[str, Any]] = []
        pending_role: Optional[str] = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if role == "system":
                merged.append(msg)
                continue

            videos = [item for item in content if item.get("type") == "video"]
            non_videos = [item for item in content if item.get("type") != "video"]

            if non_videos:
                new_content: List[Dict[str, Any]] = []
                pending_videos.extend(videos)
                if pending_videos:
                    merged_path = self._merge_video_paths(
                        [v["video"] for v in pending_videos]
                    )
                    fps = pending_videos[0].get("fps", self.video_metadata_fps)
                    new_content.append({"type": "video", "video": merged_path, "fps": fps})
                    pending_videos.clear()
                new_content.extend(non_videos)
                merged.append({"role": role, "content": new_content})
            else:
                pending_videos.extend(videos)
                pending_role = role

        if pending_videos:
            merged_path = self._merge_video_paths(
                [v["video"] for v in pending_videos]
            )
            fps = pending_videos[0].get("fps", self.video_metadata_fps)
            merged.append({
                "role": pending_role or "user",
                "content": [{"type": "video", "video": merged_path, "fps": fps}],
            })

        return merged

    def _merge_video_paths(self, video_paths: List[str]) -> str:
        if len(video_paths) == 1:
            return video_paths[0]
        p0, p1 = Path(video_paths[0]), Path(video_paths[-1])
        s0, s1 = p0.stem.split("_"), p1.stem.split("_")
        prefix = "_".join(s0[:-2]) or s0[0]
        start_t = s0[-2] if len(s0) >= 2 else "0"
        end_t = s1[-1] if len(s1) >= 1 else "end"
        output_path = str(p0.parent / f"{prefix}_{start_t}_{end_t}.mp4")
        return merge_videos(video_paths, output_path)

    def _to_gemini_payload(self, messages: List[Dict[str, Any]], max_new_tokens: int) -> Dict[str, Any]:
        messages = self._merge_context_videos(messages)

        system_text = ""
        contents: List[Dict[str, Any]] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            parts: List[Dict[str, Any]] = []
            for item in msg.get("content", []):
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        if role == "system":
                            system_text = f"{system_text}\n{text}".strip()
                        else:
                            parts.append({"text": text})
                elif item_type == "video":
                    video_path = str(item.get("video", ""))
                    b64 = self._video_to_base64(video_path)
                    parts.append({
                        "inline_data": {"mime_type": "video/mp4", "data": b64},
                        "video_metadata": {"fps": float(item.get("fps", self.video_metadata_fps))},
                    })

            if role != "system" and parts:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": parts})

        payload: Dict[str, Any] = {"contents": contents}
        if system_text:
            payload["system_instruction"] = {"parts": [{"text": system_text}]}
        payload["generationConfig"] = {
            "maxOutputTokens": int(max_new_tokens),
            "temperature": float(self.temperature),
        }
        return payload

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        candidates = response_json.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        text_chunks = [str(p.get("text", "")) for p in parts if isinstance(p, dict) and p.get("text")]
        return "\n".join(text_chunks).strip()

    def generate(self, messages: List[Dict[str, Any]], max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        if not self.model:
            return {"response": "[ERROR] Missing model name", "raw_response": "", "status_code": 500}
        url = self._build_url()
        if not url:
            return {"response": "[ERROR] Missing Gemini api_key", "raw_response": "", "status_code": 500}

        try:
            payload = self._to_gemini_payload(messages, int(max_new_tokens or self.max_new_tokens))
        except Exception as exc:
            return {"response": f"[ERROR] payload build failed: {exc}", "raw_response": "", "status_code": 500}

        if not payload.get("contents"):
            return {"response": "[ERROR] Empty contents for Gemini request", "raw_response": "", "status_code": 500}

        headers = {"Content-Type": "application/json"}
        if not self._is_google_native():
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err = ""
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                body = resp.text
                if resp.status_code == 200:
                    parsed = resp.json()
                    text = self._extract_text(parsed)
                    return {"response": text, "raw_response": text, "status_code": 200}
                last_err = f"HTTP {resp.status_code}: {body[:400]}"
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.5 * (2**attempt))
        return {"response": f"[ERROR] {last_err}", "raw_response": last_err, "status_code": 502}


def build_backend(name: str, cfg: Dict[str, Any]) -> Any:
    backend_type = str(cfg.get("backend", "openai_compatible")).strip().lower()
    if backend_type in {"openai_compatible", "hf"}:
        return OpenAICompatibleBackend(name, cfg)
    if backend_type in {"gemini_native", "gemini"}:
        return GeminiNativeBackend(name, cfg)
    raise ValueError(
        f"Unsupported backend '{backend_type}' for '{name}'. "
        "Supported: openai_compatible, hf, gemini_native"
    )


class SessionModel:
    def __init__(
        self,
        backend: Any,
        max_video_length_in_seconds: float,
        default_video_fps: float,
    ):
        self.backend = backend
        self.max_video_length_in_seconds = float(max_video_length_in_seconds)
        self.default_video_fps = float(default_video_fps)
        self.new_session()

    def new_session(self, chunk: Optional[Dict[str, Any]] = None) -> str:
        self.session_id = str(uuid.uuid4())
        self.context: List[Dict[str, Any]] = []
        self.video_chunk_info: Dict[str, Dict[str, Any]] = {}
        self.current_context_video_info: List[Dict[str, Any]] = []
        self.cum_video_length_in_seconds = 0.0
        if chunk:
            self.add_chunk(chunk)
        return self.session_id

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        new_content = []
        for ele in chunk.get("content", []):
            if ele.get("type") == "video":
                video_path = str(ele.get("video", ""))
                info = self.video_chunk_info.get(video_path)
                if info is None:
                    info = probe_video(video_path) or {"duration": 1.0, "video": {"fps": self.default_video_fps}}
                    self.video_chunk_info[video_path] = info
                duration = float(info.get("duration", 1.0) or 1.0)
                fps = float(ele.get("fps", self.default_video_fps))
                video_info = info.get("video") or {}
                num_frames = int(video_info.get("total_frames", 0) or 0)
                self.cum_video_length_in_seconds += duration
                new_content.append({
                    "type": "video",
                    "video": video_path,
                    "max_frames": num_frames,
                    "max_pixels": 384 * 28 * 28,
                    "fps": fps,
                })
            else:
                new_content.append(ele)

        self.context.append({"role": chunk.get("role", "user"), "content": new_content})
        self._truncate_context_videos()
        self._rebuild_video_info()

    def _truncate_context_videos(self) -> None:
        while self.cum_video_length_in_seconds > self.max_video_length_in_seconds:
            removed = False
            for msg_idx, msg in enumerate(self.context):
                for item_idx, item in enumerate(msg.get("content", [])):
                    if item.get("type") != "video":
                        continue
                    path = item.get("video", "")
                    dur = float(self.video_chunk_info.get(path, {}).get("duration", 1.0) or 1.0)
                    self.cum_video_length_in_seconds -= dur
                    msg["content"] = msg["content"][:item_idx] + msg["content"][item_idx + 1 :]
                    if not msg["content"]:
                        self.context = self.context[:msg_idx] + self.context[msg_idx + 1 :]
                    removed = True
                    break
                if removed:
                    break
            if not removed:
                break

    def _rebuild_video_info(self) -> None:
        out: List[Dict[str, Any]] = []
        for msg in self.context:
            for item in msg.get("content", []):
                if item.get("type") == "video":
                    raw = self.video_chunk_info.get(item.get("video", ""), {})
                    if isinstance(raw, dict):
                        out.append(raw | {"fps": float(item.get("fps", self.default_video_fps))})
        self.current_context_video_info = out

    def generate(self, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        return self.backend.generate(self.context, max_new_tokens=max_new_tokens)


@dataclass
class InferenceOptions:
    sparse_mode: bool
    active_window: int
    max_retries: int
    chunk_seconds: float
    model_video_fps: float
    trim_fps: Optional[float]
    chunk_cache_root: Optional[Path]


class StreamingState:
    def __init__(
        self,
        sample: Dict[str, Any],
        chunk_seconds: float,
        trim_fps: Optional[float],
        chunk_cache_root: Optional[Path],
    ):
        self.video_uuid = sample.get("uuid", "")
        self.video_path = sample.get("video", "")
        self.events = sorted(sample.get("sqa", []), key=lambda x: time_to_seconds(x["timestamp"]))
        self.events_queue = deque(self.events)
        self.current_time = 0.0
        self.last_event_time = float(time_to_seconds(self.events[-1]["timestamp"])) if self.events else 0.0
        self.duration = float(sample.get("video_info", {}).get("duration", 0.0))
        if self.duration <= 0:
            self.duration = self.last_event_time + 10.0
        self.chunk_seconds = chunk_seconds
        self.trim_fps = trim_fps
        self.chunk_cache_dir = self._resolve_chunk_cache_dir(sample, chunk_cache_root)
        self.chunk_cache_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_chunk_cache_dir(self, sample: Dict[str, Any], chunk_cache_root: Optional[Path]) -> Path:
        if chunk_cache_root is not None:
            return chunk_cache_root / sample.get("uuid", "unknown")
        stream_addr = sample.get("stream_addr")
        if stream_addr:
            return Path(stream_addr)
        return SCRIPT_DIR / "cache" / "stream_chunk_cache" / sample.get("uuid", "unknown")

    def _collect_current_events(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while self.events_queue and time_to_seconds(self.events_queue[0]["timestamp"]) <= self.current_time:
            out.append(self.events_queue.popleft())
        return out

    def _get_or_create_chunk(self, chunk_path: Path, start_time: str, end_time: str) -> str:
        if chunk_path.exists():
            return str(chunk_path)
        try:
            trim_video(
                video_path=self.video_path,
                trim_path=str(chunk_path),
                start_time=start_time,
                end_time=end_time,
                fps=self.trim_fps,
            )
        except Exception as exc:
            print(f"[WARN] chunk trim failed for {chunk_path}: {exc}")
        return str(chunk_path)

    def step(self) -> Dict[str, Any]:
        valid_chunk = self.current_time < self.duration
        prev_ts = seconds_to_time(int(self.current_time))
        self.current_time += self.chunk_seconds
        current_ts = seconds_to_time(int(self.current_time))
        chunk_file = f"video_{self.video_uuid}_{prev_ts}_{current_ts}.mp4".replace(":", "")

        new_events = self._collect_current_events()
        is_finished = (self.current_time >= int(self.duration)) or (self.current_time >= self.last_event_time + 10.0)
        return {
            "current_timestamp": current_ts,
            "stream_chunk": self._get_or_create_chunk(self.chunk_cache_dir / chunk_file, prev_ts, current_ts)
            if valid_chunk
            else None,
            "new_events": new_events,
            "is_finished": is_finished,
        }


def render_user_prompt(event: Dict[str, Any]) -> str:
    prompt = ""
    if event.get("question"):
        prompt += f"Question: {event['question']}\n"
    if event.get("options") is not None:
        prompt += f"Options: {event['options']}\n"
    return prompt


class UnifiedInferenceRunner:
    def __init__(
        self,
        config: Dict[str, Any],
        config_dir: Path,
        model_name: str,
        prompts_name: str,
        model_path_override: str,
        options: InferenceOptions,
        context_window_seconds: Optional[float],
        video_root: Optional[Path],
        stream_addr_root: Optional[Path],
    ):
        self.config = config
        self.config_dir = config_dir
        self.model_name = model_name
        self.prompts_name = prompts_name
        self.options = options
        self.video_root = video_root
        self.stream_addr_root = stream_addr_root

        model_cfg = dict(config["models"][model_name])
        if model_path_override:
            model_cfg["model"] = model_path_override
        backend = build_backend(model_name, model_cfg)
        context_window = (
            float(context_window_seconds)
            if context_window_seconds is not None
            else float(model_cfg.get("max_video_length_in_seconds", 60.2))
        )
        self.model_max_new_tokens = int(model_cfg.get("max_new_tokens", 1024))
        self.model = SessionModel(
            backend=backend,
            max_video_length_in_seconds=context_window,
            default_video_fps=self.options.model_video_fps,
        )

        prompt_cfg_raw = config.get("prompts", {}).get(prompts_name)
        if not isinstance(prompt_cfg_raw, dict):
            raise ValueError(f"Prompt preset '{prompts_name}' not found in config.prompts")
        prompt_cfg = dict(prompt_cfg_raw)
        if "system_prompt" not in prompt_cfg:
            raise ValueError(f"Prompt preset '{prompts_name}' missing 'system_prompt'")
        self.system_prompt_text = str(prompt_cfg["system_prompt"])
        self.silent_word = str(prompt_cfg.get("silent_word", "silent")).strip().lower()

    def _is_silent_response(self, text: str) -> bool:
        return (text or "").strip().lower() == self.silent_word

    def _build_user_content(
        self,
        stream_chunk: Optional[str],
        new_events: List[Dict[str, Any]],
        answer_window: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        content: List[Dict[str, Any]] = []
        if stream_chunk:
            content.append({"type": "video", "video": stream_chunk, "fps": float(self.options.model_video_fps)})

        if new_events:
            act_w = -1
            for event in new_events:
                prompt = render_user_prompt(event)
                if prompt:
                    content.append({"type": "text", "text": prompt})
                if event.get("question") and event.get("response"):
                    act_w = max(1, act_w)
                elif event.get("question") or event.get("response"):
                    act_w = max(self.options.active_window, act_w)
            answer_window = max(answer_window, act_w)
        return content, answer_window

    def _generate_once(self, current_time: float) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw = self.model.generate(max_new_tokens=self.model_max_new_tokens)
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        response = str(raw.get("response", ""))
        return {
            "timestamp": seconds_to_time(int(current_time)),
            "response": response,
            "raw_response": str(raw.get("raw_response", response)),
            "status_code": int(raw.get("status_code", 200)),
            "runtime_ms": round(runtime_ms, 2),
        }

    def _generate_with_retry(self, current_time: float, retries_left: int) -> Tuple[Dict[str, Any], int]:
        while True:
            try:
                result = self._generate_once(current_time)
            except Exception as exc:
                result = {
                    "timestamp": seconds_to_time(int(current_time)),
                    "response": f"[ERROR] {exc}",
                    "raw_response": f"[ERROR] {exc}",
                    "status_code": 502,
                    "runtime_ms": 0.0,
                }
            if result["status_code"] == 200:
                return result, retries_left
            if retries_left <= 0:
                return result, retries_left
            retries_left -= 1

    def run_sample(self, sample: Dict[str, Any], bench_name: str) -> Dict[str, Any]:
        streaming = StreamingState(
            sample=sample,
            chunk_seconds=self.options.chunk_seconds,
            trim_fps=self.options.trim_fps,
            chunk_cache_root=self.options.chunk_cache_root,
        )
        sys_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt_text}]}
        self.model.new_session(sys_msg)

        responses: List[Dict[str, Any]] = []
        answer_window = -1
        retries_left = int(self.options.max_retries)
        while True:
            step = streaming.step()
            content, answer_window = self._build_user_content(
                stream_chunk=step["stream_chunk"],
                new_events=step["new_events"],
                answer_window=answer_window,
            )
            if content:
                self.model.add_chunk({"role": "user", "content": content})
                answer_window = max(-1, answer_window - 1)
                if (not self.options.sparse_mode) or (answer_window >= 0):
                    record, retries_left = self._generate_with_retry(streaming.current_time, retries_left)
                    responses.append(record)
                    if record["status_code"] != 200:
                        break
                    if not self._is_silent_response(record["response"]):
                        self.model.add_chunk(
                            {"role": "assistant", "content": [{"type": "text", "text": record["response"]}]}
                        )
            if step["is_finished"]:
                break
        return sample | {"responses": responses, "bench": bench_name}

    def benchmark_path(self, benchmark_name_or_path: str) -> Path:
        bench_cfg = self.config.get("benchmarks", {}).get(benchmark_name_or_path, {})
        if isinstance(bench_cfg, dict) and "path" in bench_cfg:
            return resolve_path(str(bench_cfg["path"]), [self.config_dir, SCRIPT_DIR, Path.cwd()])
        return resolve_path(benchmark_name_or_path, [Path.cwd(), self.config_dir, SCRIPT_DIR])

    def run_benchmarks(self, benchmark_list: List[str], output_dir: Path) -> List[Dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_records: List[Dict[str, Any]] = []
        for bench in benchmark_list:
            bench_path = self.benchmark_path(bench)
            samples = load_samples_any_format(
                benchmark_path=bench_path,
                bench_name=bench,
                video_root=self.video_root,
                stream_addr_root=self.stream_addr_root,
                need_video_info=True,
            )
            print(f"\n[Inference] benchmark={bench} path={bench_path} samples={len(samples)}")
            for sample in tqdm(samples, desc=bench):
                record = self.run_sample(sample, bench)
                append_jsonl(output_dir / f"{bench}.jsonl", record)
                all_records.append(record)
        return all_records


def parse_logical_questions(raw_sqa: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logical: List[Dict[str, Any]] = []
    i = 0
    while i < len(raw_sqa):
        item = raw_sqa[i]
        if "question" in item and "response" in item:
            t_question = time_to_seconds(item["timestamp"])
            task_type = normalize_task_type_preserving_legacy(
                item.get("type", item.get("task_type", "DefaultType")),
                default="DefaultType",
            )
            logical.append(
                {
                    "question_time_sec": t_question,
                    "answer_event_time_sec": t_question,
                    "question": item["question"],
                    "ground_truth": item["response"],
                    "is_objective": "options" in item,
                    "options": item.get("options"),
                    "task_type": task_type,
                }
            )
            i += 1
        elif "question" in item and "response" not in item:
            t_question = time_to_seconds(item["timestamp"])
            t_answer_event = t_question
            ground_truth = ""
            task_type = normalize_task_type_preserving_legacy(
                item.get("type", item.get("task_type", "DefaultType")),
                default="DefaultType",
            )
            if i + 1 < len(raw_sqa):
                next_item = raw_sqa[i + 1]
                if "response" in next_item and "question" not in next_item:
                    t_answer_event = time_to_seconds(next_item["timestamp"])
                    ground_truth = next_item["response"]
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            logical.append(
                {
                    "question_time_sec": t_question,
                    "answer_event_time_sec": t_answer_event,
                    "question": item["question"],
                    "ground_truth": ground_truth,
                    "is_objective": "options" in item,
                    "options": item.get("options"),
                    "task_type": task_type,
                }
            )
        else:
            i += 1
    return logical


class LLMJudger:
    def __init__(self, backend: Any, prompt_template: str):
        self.backend = backend
        self.prompt_template = prompt_template

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        if repair_json is not None:
            parsed = json.loads(repair_json(raw))
        else:
            parsed = json.loads(raw)
        score = max(0.0, min(5.0, float(parsed["score"])))
        explanation = str(parsed.get("explanation", ""))
        return {"score": score, "explanation": explanation}

    def judge(self, question: str, model_output: str, reference: str, retries: int = 5) -> Dict[str, Any]:
        import time as _time
        prompt = self.prompt_template.format(
            question=question,
            model_output=model_output,
            reference_answer=reference,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        last_err = None
        for attempt in range(retries):
            if attempt > 0:
                _time.sleep(min(2 ** attempt, 16))
            result = self.backend.generate(messages, max_new_tokens=256)
            if int(result.get("status_code", 500)) != 200:
                last_err = result.get("response", "")
                continue
            try:
                return self._parse_response(str(result.get("response", "")).strip())
            except Exception as exc:
                last_err = exc
                continue
        return {"score": 0.0, "explanation": f"Judger parse error: {last_err}"}


def load_llm_judger(config: Dict[str, Any]) -> Optional[LLMJudger]:
    judger_cfg = config.get("judger")
    if not isinstance(judger_cfg, dict):
        return None
    backend = build_backend("judger", judger_cfg)
    if "prompt_template" not in judger_cfg:
        raise ValueError("config.judger.prompt_template is required")
    prompt_template = str(judger_cfg["prompt_template"])
    return LLMJudger(backend, prompt_template)


def evaluate_sample(sample: Dict[str, Any], llm_judger: Optional[LLMJudger], time_window: float) -> List[Dict[str, Any]]:
    logical_questions = parse_logical_questions(sample.get("sqa", []))
    if not logical_questions:
        return []
    model_responses: List[Tuple[int, str]] = []
    for response in sample.get("responses", []):
        t = time_to_seconds(response["timestamp"])
        model_responses.append((t, response.get("response", "")))
    model_responses.sort(key=lambda x: x[0])

    used_indices = set()
    results: List[Dict[str, Any]] = []
    for q in logical_questions:
        t_question = q["question_time_sec"]
        t_answer = q["answer_event_time_sec"]
        ground_truth = q["ground_truth"]

        window_start = t_question
        window_end = t_answer + time_window
        correct_time_end = t_answer if t_question == t_answer else (t_answer + time_window)
        for idx, (t_resp, model_text) in enumerate(model_responses):
            if idx in used_indices:
                continue
            if not (window_start <= t_resp <= window_end):
                continue
            if model_text == ground_truth or (not is_placeholder(model_text)):
                if is_forward_task(q["task_type"]) and t_resp == t_question:
                    continue
                q["model_response_time_sec"] = t_resp
                q["model_response_content"] = model_text
                used_indices.add(idx)
                break

        t_model = q.get("model_response_time_sec", t_answer)
        c_model = q.get("model_response_content", "")
        explanation = ""
        if t_model < t_answer:
            score_100 = 0.0
            category = "EarlyResponse"
        elif c_model != ground_truth and is_placeholder(c_model):
            score_100 = 0.0
            category = "NoResponse"
        elif t_model > correct_time_end:
            score_100 = 0.0
            category = "LateResponse"
        elif q["is_objective"]:
            clean_up = lambda x: x.strip().replace(".", "")[:1]
            if c_model.lower() == ground_truth.lower() or clean_up(c_model).lower() == ground_truth.lower():
                score_100 = 100.0
                category = "Correct"
            else:
                score_100 = 0.0
                category = "WrongAnswer"
        else:
            if llm_judger is None:
                score_100 = 0.0
                category = "Error (no LLM)"
            else:
                judged = llm_judger.judge(q["question"], c_model, ground_truth)
                score_100 = judged["score"] * 20.0
                explanation = judged.get("explanation", "")
                category = "PartlyCorrect"

        results.append(
            {
                "sample_id": sample["id"],
                "question_time": seconds_to_time(q["question_time_sec"]),
                "question": q["question"],
                "answer_time": seconds_to_time(q["answer_event_time_sec"]),
                "answer": ground_truth,
                "response_time": seconds_to_time(int(t_model)),
                "response": c_model,
                "score": score_100,
                "category": category,
                "task_type": q["task_type"],
                "is_objective": q["is_objective"],
                "explanation": explanation,
            }
        )
    return results


def build_summary(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    total = len(df)
    if total == 0:
        return {"model_name": model_name, "#samples": 0, "final_score": 0.0}
    summary: Dict[str, Any] = {"model_name": model_name, "#samples": total, "final_score": df["score"].mean()}
    for kind, mask in [("objective", df["is_objective"]), ("subjective", ~df["is_objective"])]:
        subset = df[mask]
        summary[kind] = round(subset["score"].mean() if len(subset) else 0.0, 1)
    task_types = df["task_type"].dropna().unique()
    for task_type in task_types:
        subset_obj = df[df["is_objective"] & (df["task_type"] == task_type)]
        subset_sub = df[(~df["is_objective"]) & (df["task_type"] == task_type)]
        summary[f"{task_type}(objective)"] = round(subset_obj["score"].mean() if len(subset_obj) else 0.0, 1)
        summary[f"{task_type}(subjective)"] = round(subset_sub["score"].mean() if len(subset_sub) else 0.0, 1)

    categories = df["category"].unique()
    forward_mask = df["task_type"].astype(str).str.lower().isin(FORWARD_TASK_TYPES)
    forward_obj = df[forward_mask & (df["is_objective"])]
    forward_sub = df[forward_mask & (~df["is_objective"])]
    for cat in categories:
        subset = forward_obj[forward_obj["category"] == cat]
        percent = round(len(subset) / len(forward_obj) * 100, 1) if len(forward_obj) else 0.0
        score = round(subset["score"].mean(), 1) if len(subset) else 0.0
        summary[f"{cat}(objective-future)"] = f"{percent}%({score})"
    for cat in categories:
        subset = forward_sub[forward_sub["category"] == cat]
        percent = round(len(subset) / len(forward_sub) * 100, 1) if len(forward_sub) else 0.0
        score = round(subset["score"].mean(), 1) if len(subset) else 0.0
        summary[f"{cat}(subjective-future)"] = f"{percent}%({score})"
    return summary


def run_scoring(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    collection_name: str,
    time_window: float,
    disable_llm_judge: bool,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    judger = None if disable_llm_judge else load_llm_judger(config)
    all_rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="Scoring"):
        all_rows.extend(evaluate_sample(sample, judger, time_window))
    details_path = output_dir / f"{collection_name}_details.jsonl"
    write_jsonl(details_path, all_rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
    else:
        df = pd.DataFrame(
            columns=[
                "sample_id",
                "question_time",
                "question",
                "answer_time",
                "answer",
                "response_time",
                "response",
                "score",
                "category",
                "task_type",
                "is_objective",
                "explanation",
            ]
        )
    summary = build_summary(df, model_name)
    summary_df = pd.DataFrame([summary]).round(1)

    db_path = output_dir / f"{collection_name}.db"
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(model_name, conn, if_exists="replace", index=False)
    finally:
        conn.close()
    csv_path = output_dir / f"{collection_name}.csv"
    if csv_path.exists():
        merged = pd.concat([pd.read_csv(csv_path), summary_df], ignore_index=True)
        merged.to_csv(csv_path, index=False)
    else:
        summary_df.to_csv(csv_path, index=False)
    summary_json = output_dir / f"{collection_name}_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return {
        "details_jsonl": details_path,
        "sqlite_db": db_path,
        "summary_csv": csv_path,
        "summary_json": summary_json,
    }


def resolve_model_output_input(path_str: str) -> List[Path]:
    path = Path(path_str)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"model output not found: {path}")
    files = sorted(path.glob("*.jsonl")) + sorted(path.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"no json/jsonl files under: {path}")
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-contained StreamEval inference + scoring")
    parser.add_argument("--config", default="", help="Optional local config yaml")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", default="", help="Override model id/path for API payload")
    parser.add_argument("--benchmarks", nargs="+", default=[], help="Benchmark names or paths")
    parser.add_argument("--video-root", default="", help="Root dir for relative video_path in new-format datasets")
    parser.add_argument("--stream-addr-root", default="", help="Root dir for generated stream chunk cache")
    parser.add_argument(
        "--prompts",
        default="",
        help="Prompt preset name. Empty means use config default_prompt.",
    )
    parser.add_argument("--output-dir", default=str(SCRIPT_DIR / "runs"))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    parser.add_argument("--sparse-mode", type=int, default=1)
    parser.add_argument("--active-window", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--chunk-seconds", type=float, default=1.0)
    parser.add_argument("--model-video-fps", type=float, default=2.0)
    parser.add_argument("--trim-fps", type=float, default=None)
    parser.add_argument("--chunk-cache-root", type=str, default=None)
    parser.add_argument("--context-window-seconds", type=float, default=None)

    parser.add_argument("--time-window", type=float, default=2.0)
    parser.add_argument("--collection", default="result")
    parser.add_argument("--disable-llm-judge", action="store_true")

    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    parser.add_argument("--model-output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_inference and not args.model_output:
        raise ValueError("--model-output is required when --skip-inference is set")
    if args.skip_inference and args.skip_scoring:
        raise ValueError("both --skip-inference and --skip-scoring are set, nothing to do")

    config_arg = Path(args.config).resolve() if args.config else DEFAULT_CONFIG_PATH
    config, config_path = load_release_config(config_arg)
    init_runtime_constants(config)
    config_dir = config_path.parent if config_path.exists() else SCRIPT_DIR
    prompt_name = str(args.prompts).strip() or str(config.get("default_prompt", "streaming"))
    bench_list = args.benchmarks or list(config.get("default_benchmarks", []))
    if not bench_list and not args.skip_inference:
        raise ValueError("No benchmarks provided and no default_benchmarks found in config")

    config_video_root = str(config.get("video_root", "") or "").strip()
    config_stream_addr_root = str(config.get("stream_addr_root", "") or "").strip()
    video_root = Path(args.video_root) if args.video_root else (Path(config_video_root) if config_video_root else None)
    stream_addr_root = (
        Path(args.stream_addr_root)
        if args.stream_addr_root
        else (Path(config_stream_addr_root) if config_stream_addr_root else None)
    )

    run_root = Path(args.output_dir) / f"{args.model_name}_{args.run_id}"
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "config_path": str(config_path),
                "default_prompt": prompt_name,
                "created_at": datetime.now().isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    inference_samples: List[Dict[str, Any]] = []
    inference_inputs: List[Path] = []
    if not args.skip_inference:
        options = InferenceOptions(
            sparse_mode=bool(args.sparse_mode),
            active_window=args.active_window,
            max_retries=args.max_retries,
            chunk_seconds=args.chunk_seconds,
            model_video_fps=args.model_video_fps,
            trim_fps=args.trim_fps,
            chunk_cache_root=Path(args.chunk_cache_root) if args.chunk_cache_root else None,
        )
        runner = UnifiedInferenceRunner(
            config=config,
            config_dir=config_dir,
            model_name=args.model_name,
            prompts_name=prompt_name,
            model_path_override=args.model_path,
            options=options,
            context_window_seconds=args.context_window_seconds,
            video_root=video_root,
            stream_addr_root=stream_addr_root,
        )
        with open(run_root / "system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(runner.system_prompt_text + "\n")
        inference_dir = run_root / "inference"
        inference_samples = runner.run_benchmarks(bench_list, inference_dir)
        merged_jsonl = inference_dir / "all_benchmarks.jsonl"
        write_jsonl(merged_jsonl, inference_samples)
        inference_inputs.append(merged_jsonl)
    else:
        for path in resolve_model_output_input(args.model_output):
            inference_inputs.append(path)
            inference_samples.extend(
                load_samples_any_format(
                    benchmark_path=path,
                    bench_name=path.stem,
                    video_root=video_root,
                    stream_addr_root=stream_addr_root,
                    need_video_info=False,
                )
            )

    scoring_outputs = {}
    if not args.skip_scoring:
        scoring_outputs = run_scoring(
            samples=inference_samples,
            config=config,
            model_name=args.model_name,
            output_dir=run_root / "scoring",
            collection_name=args.collection,
            time_window=args.time_window,
            disable_llm_judge=args.disable_llm_judge,
        )

    print("\n=== StreamEval Release Completed ===")
    print(f"Run root: {run_root}")
    if inference_inputs:
        print("Inference inputs/outputs:")
        for path in inference_inputs:
            print(f"  - {path}")
    if scoring_outputs:
        print("Scoring outputs:")
        for key, value in scoring_outputs.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
