#!/usr/bin/env python3
"""
Parallel StreamEval runner.

Same semantics as run_stream_eval.py, but processes samples concurrently
via ThreadPoolExecutor.  Supports distributing requests across multiple
API endpoints (one per GPU / model-server instance).

Usage (single server, 10 concurrent workers):
    python run_stream_eval_parallel.py \
        --model-name qwen3_omni \
        --num-workers 10

Usage (10 servers on different ports, 10 workers):
    python run_stream_eval_parallel.py \
        --model-name qwen3_omni \
        --num-workers 10 \
        --api-bases http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,...
"""

from __future__ import annotations

import argparse
import copy
import json
import queue
import sqlite3
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from run_stream_eval import (
    DEFAULT_CONFIG_PATH,
    SCRIPT_DIR,
    InferenceOptions,
    UnifiedInferenceRunner,
    append_jsonl,
    build_summary,
    evaluate_sample,
    init_runtime_constants,
    load_llm_judger,
    load_release_config,
    load_samples_any_format,
    resolve_model_output_input,
    write_jsonl,
)


# ---------------------------------------------------------------------------
# Runner pool – one runner per concurrent thread keeps SessionModel safe
# ---------------------------------------------------------------------------

class RunnerPool:
    """Thread-safe pool of UnifiedInferenceRunner instances."""

    def __init__(self, runners: List[UnifiedInferenceRunner]):
        self._q: queue.Queue[UnifiedInferenceRunner] = queue.Queue()
        for r in runners:
            self._q.put(r)

    def acquire(self) -> UnifiedInferenceRunner:
        return self._q.get()

    def release(self, runner: UnifiedInferenceRunner) -> None:
        self._q.put(runner)


def _build_runners(
    num_workers: int,
    config: Dict[str, Any],
    config_dir: Path,
    model_name: str,
    prompt_name: str,
    model_path: str,
    options: InferenceOptions,
    context_window_seconds: Optional[float],
    video_root: Optional[Path],
    stream_addr_root: Optional[Path],
    api_bases: Optional[List[str]],
) -> List[UnifiedInferenceRunner]:
    runners: List[UnifiedInferenceRunner] = []
    for i in range(num_workers):
        cfg = copy.deepcopy(config)
        if api_bases:
            cfg["models"][model_name]["api_base"] = api_bases[i % len(api_bases)]
        runners.append(
            UnifiedInferenceRunner(
                config=cfg,
                config_dir=config_dir,
                model_name=model_name,
                prompts_name=prompt_name,
                model_path_override=model_path,
                options=options,
                context_window_seconds=context_window_seconds,
                video_root=video_root,
                stream_addr_root=stream_addr_root,
            )
        )
    return runners


# ---------------------------------------------------------------------------
# Parallel inference
# ---------------------------------------------------------------------------

def _parallel_inference(
    pool: RunnerPool,
    num_workers: int,
    bench_tasks: List[Tuple[str, Dict[str, Any]]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lock = threading.Lock()

    def _process(bench: str, sample: Dict[str, Any]) -> Dict[str, Any]:
        runner = pool.acquire()
        try:
            record = runner.run_sample(sample, bench)
        finally:
            pool.release(runner)
        with write_lock:
            append_jsonl(output_dir / f"{bench}.jsonl", record)
        return record

    all_records: List[Dict[str, Any]] = []
    errors = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map = {
            executor.submit(_process, bench, sample): (bench, sample)
            for bench, sample in bench_tasks
        }
        with tqdm(total=len(future_map), desc="Inference") as pbar:
            for future in as_completed(future_map):
                bench, sample = future_map[future]
                try:
                    all_records.append(future.result())
                except Exception:
                    errors += 1
                    print(
                        f"\n[ERROR] sample {sample.get('id', '?')} "
                        f"in {bench}:\n{traceback.format_exc()}"
                    )
                pbar.update(1)

    if errors:
        print(f"\n[WARN] {errors} sample(s) failed during inference")
    return all_records


# ---------------------------------------------------------------------------
# Parallel scoring
# ---------------------------------------------------------------------------

def _parallel_scoring(
    samples: List[Dict[str, Any]],
    config: Dict[str, Any],
    model_name: str,
    output_dir: Path,
    collection_name: str,
    time_window: float,
    disable_llm_judge: bool,
    num_scorers: int,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    judger = None if disable_llm_judge else load_llm_judger(config)

    all_rows: List[Dict[str, Any]] = []
    rows_lock = threading.Lock()
    errors = 0

    def _score_one(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        return evaluate_sample(sample, judger, time_window)

    with ThreadPoolExecutor(max_workers=num_scorers) as executor:
        future_map = {executor.submit(_score_one, s): s for s in samples}
        with tqdm(total=len(future_map), desc="Scoring") as pbar:
            for future in as_completed(future_map):
                try:
                    rows = future.result()
                    with rows_lock:
                        all_rows.extend(rows)
                except Exception:
                    errors += 1
                    s = future_map[future]
                    print(
                        f"\n[ERROR] scoring sample {s.get('id', '?')}:\n"
                        f"{traceback.format_exc()}"
                    )
                pbar.update(1)

    if errors:
        print(f"\n[WARN] {errors} sample(s) failed during scoring")

    details_path = output_dir / f"{collection_name}_details.jsonl"
    write_jsonl(details_path, all_rows)

    if all_rows:
        df = pd.DataFrame(all_rows)
    else:
        df = pd.DataFrame(
            columns=[
                "sample_id", "question_time", "question", "answer_time",
                "answer", "response_time", "response", "score",
                "category", "task_type", "is_objective", "explanation",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parallel StreamEval inference + scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="", help="Optional local config yaml")
    p.add_argument("--model-name", required=True)
    p.add_argument("--model-path", default="")
    p.add_argument("--benchmarks", nargs="+", default=[])
    p.add_argument("--video-root", default="")
    p.add_argument("--stream-addr-root", default="")
    p.add_argument("--prompts", default="")
    p.add_argument("--output-dir", default=str(SCRIPT_DIR / "runs"))
    p.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))

    p.add_argument("--sparse-mode", type=int, default=1)
    p.add_argument("--active-window", type=int, default=2)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--chunk-seconds", type=float, default=1.0)
    p.add_argument("--model-video-fps", type=float, default=2.0)
    p.add_argument("--trim-fps", type=float, default=None)
    p.add_argument("--chunk-cache-root", type=str, default=None)
    p.add_argument("--context-window-seconds", type=float, default=None)

    p.add_argument("--time-window", type=float, default=2.0)
    p.add_argument("--collection", default="result")
    p.add_argument("--disable-llm-judge", action="store_true")

    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--skip-scoring", action="store_true")
    p.add_argument("--model-output", default="")

    # ---- parallel-specific ----
    p.add_argument(
        "--num-workers", type=int, default=10,
        help="Number of concurrent inference workers (default: 10)",
    )
    p.add_argument(
        "--num-scorers", type=int, default=0,
        help="Number of concurrent scoring workers (default: same as --num-workers)",
    )
    p.add_argument(
        "--api-bases", default="",
        help=(
            "Comma-separated API base URLs for round-robin distribution. "
            "E.g. http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1  "
            "If empty, all workers share the config default."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.skip_inference and not args.model_output:
        raise ValueError("--model-output is required when --skip-inference is set")
    if args.skip_inference and args.skip_scoring:
        raise ValueError("both --skip-inference and --skip-scoring are set, nothing to do")

    num_workers = max(1, args.num_workers)
    num_scorers = max(1, args.num_scorers or num_workers)

    api_bases: Optional[List[str]] = None
    if args.api_bases.strip():
        api_bases = [u.strip() for u in args.api_bases.split(",") if u.strip()]

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
    video_root = (
        Path(args.video_root) if args.video_root
        else (Path(config_video_root) if config_video_root else None)
    )
    stream_addr_root = (
        Path(args.stream_addr_root) if args.stream_addr_root
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
                "num_workers": num_workers,
                "num_scorers": num_scorers,
                "api_bases": api_bases,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ---- inference ----
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
        print(f"[Parallel] Creating {num_workers} inference runners ...")
        runners = _build_runners(
            num_workers=num_workers,
            config=config,
            config_dir=config_dir,
            model_name=args.model_name,
            prompt_name=prompt_name,
            model_path=args.model_path,
            options=options,
            context_window_seconds=args.context_window_seconds,
            video_root=video_root,
            stream_addr_root=stream_addr_root,
            api_bases=api_bases,
        )
        pool = RunnerPool(runners)

        with open(run_root / "system_prompt.txt", "w", encoding="utf-8") as f:
            f.write(runners[0].system_prompt_text + "\n")

        bench_tasks: List[Tuple[str, Dict[str, Any]]] = []
        for bench in bench_list:
            bench_path = runners[0].benchmark_path(bench)
            samples = load_samples_any_format(
                benchmark_path=bench_path,
                bench_name=bench,
                video_root=video_root,
                stream_addr_root=stream_addr_root,
                need_video_info=True,
            )
            print(f"[Inference] benchmark={bench}  path={bench_path}  samples={len(samples)}")
            for s in samples:
                bench_tasks.append((bench, s))

        print(f"[Parallel] Total {len(bench_tasks)} samples × {num_workers} workers\n")

        inference_dir = run_root / "inference"
        inference_samples = _parallel_inference(pool, num_workers, bench_tasks, inference_dir)

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

    # ---- scoring ----
    scoring_outputs: Dict[str, Path] = {}
    if not args.skip_scoring:
        print(f"\n[Parallel] Scoring with {num_scorers} workers ...")
        scoring_outputs = _parallel_scoring(
            samples=inference_samples,
            config=config,
            model_name=args.model_name,
            output_dir=run_root / "scoring",
            collection_name=args.collection,
            time_window=args.time_window,
            disable_llm_judge=args.disable_llm_judge,
            num_scorers=num_scorers,
        )

    print("\n=== StreamEval Parallel Completed ===")
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
