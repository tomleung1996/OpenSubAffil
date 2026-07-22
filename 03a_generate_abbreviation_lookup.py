"""Generate the abbreviation-expansion lookup used by step 03."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import orjson
from tqdm import tqdm

from config import ABBR_LOOKUP_CSV, DATA_DIR, NER_OUTPUT_JSONL
from text_utils import clean_department_string, is_abbreviation

MODEL_NAME = "gemini-3-flash-preview"
NAMES_PER_REQUEST = 50
NUM_PARTS = 4
POLL_SECONDS = 300

PROMPT_TEMPLATE = """
You are an expert in academic data normalization.
Your task is to expand abbreviated institution strings into their full, formal English forms.

**Rules:**
1. Focus on academic and organizational terminology (e.g., "univ" -> "university", "inst" -> "institute").
2. Maintain the original word order.
3. Do not add new words; only expand existing abbreviations.
4. Expanded names should be in lowercase and without punctuation.

Please expand the following list of abbreviated institution names into their full forms.
Focus on accuracy and standard academic terminology.

Input List:
{input_strings}
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "items": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "abbr_name": {
                        "type": "STRING",
                        "description": "Original abbreviation",
                    },
                    "full_name": {
                        "type": "STRING",
                        "description": "Expanded full name",
                    },
                },
                "required": ["abbr_name", "full_name"],
            },
        }
    },
    "required": ["items"],
}

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ner-jsonl", type=Path, default=NER_OUTPUT_JSONL)
    parser.add_argument("--output", type=Path, default=ABBR_LOOKUP_CSV)
    parser.add_argument(
        "--work-dir", type=Path, default=DATA_DIR / "abbreviation_batch"
    )
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--names-per-request", type=int, default=NAMES_PER_REQUEST)
    parser.add_argument("--num-parts", type=int, default=NUM_PARTS)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    return parser.parse_args()


def collect_abbreviations(jsonl_path: Path) -> list[str]:
    """Collect unique cleaned SUB strings containing a type-word abbreviation."""
    abbreviations: set[str] = set()
    with jsonl_path.open("rb") as fh:
        for line in tqdm(fh, desc="Collecting abbreviated SUB strings"):
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            for entity in row.get("entities", []):
                if entity.get("entity_group") != "SUB":
                    continue
                cleaned = clean_department_string(entity.get("word", ""))
                if cleaned and is_abbreviation(cleaned):
                    abbreviations.add(cleaned)
    return sorted(abbreviations)


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    if size <= 0:
        raise ValueError("names-per-request must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def split_parts(items: list[str], num_parts: int) -> list[list[str]]:
    if num_parts <= 0:
        raise ValueError("num-parts must be positive")
    if not items:
        return []
    part_size = math.ceil(len(items) / num_parts)
    return [items[start:start + part_size] for start in range(0, len(items), part_size)]


def write_request_file(
    names: list[str], path: Path, part_number: int, names_per_request: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for request_number, names_chunk in enumerate(
            chunked(names, names_per_request)
        ):
            prompt = PROMPT_TEMPLATE.format(input_strings=names_chunk)
            request = {
                "key": f"part{part_number}_req_{request_number * names_per_request}",
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generation_config": {
                        "response_mime_type": "application/json",
                        "response_schema": RESPONSE_SCHEMA,
                    },
                },
            }
            fh.write(json.dumps(request, ensure_ascii=False) + "\n")


def wait_for_job(client: Any, job_name: str, poll_seconds: int) -> Any:
    while True:
        job = client.batches.get(name=job_name)
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        print(f"  {job_name}: {state}")
        if state in TERMINAL_STATES:
            return job
        time.sleep(poll_seconds)


def parse_results(content: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("error"):
            raise RuntimeError(f"Batch response line {line_number}: {entry['error']}")
        response = entry.get("response") or entry.get("responseBody")
        if not response:
            continue
        candidates = response.get("candidates", [])
        if not candidates:
            continue
        parts = candidates[0].get("content", {}).get("parts", [])
        text = next((part.get("text") for part in parts if part.get("text")), None)
        if not text:
            continue
        payload = json.loads(text)
        for item in payload.get("items", []):
            abbr_name = item.get("abbr_name")
            full_name = item.get("full_name")
            if isinstance(abbr_name, str) and isinstance(full_name, str):
                if abbr_name.strip() and full_name.strip():
                    results.append({
                        "abbr_name": abbr_name,
                        "full_name": full_name,
                    })
    return results


def run_batch_part(
    client: Any, request_path: Path, model: str, part_number: int, poll_seconds: int
) -> list[dict[str, str]]:
    from google.genai import types

    display_name = f"opensubaffil_abbreviation_part_{part_number}"
    uploaded_file = client.files.upload(
        file=request_path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    job = client.batches.create(
        model=model,
        src=uploaded_file.name,
        config={"display_name": display_name},
    )
    print(f"Submitted {job.name}")
    job = wait_for_job(client, job.name, poll_seconds)
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch job {job.name} ended in {state}: {getattr(job, 'error', None)}"
        )
    if not job.dest or not job.dest.file_name:
        raise RuntimeError(f"Batch job {job.name} returned no result file")
    content = client.files.download(file=job.dest.file_name).decode("utf-8")
    return parse_results(content)


def write_lookup(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["abbr_name", "full_name"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        from google import genai
    except ImportError as exc:
        raise SystemExit(
            "Install google-genai and set GEMINI_API_KEY before running this script."
        ) from exc

    abbreviations = collect_abbreviations(args.ner_jsonl)
    print(f"Collected {len(abbreviations):,} unique abbreviated SUB strings")

    parts = split_parts(abbreviations, args.num_parts)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    request_files: list[Path] = []
    for part_number, names in enumerate(parts, start=1):
        request_path = args.work_dir / f"abbreviation_requests_part_{part_number}.jsonl"
        write_request_file(names, request_path, part_number, args.names_per_request)
        request_files.append(request_path)

    client = genai.Client()
    results: list[dict[str, str]] = []
    for part_number, request_path in enumerate(request_files, start=1):
        part_results = run_batch_part(
            client, request_path, args.model, part_number, args.poll_seconds
        )
        print(f"Part {part_number}: received {len(part_results):,} expansions")
        results.extend(part_results)

    write_lookup(results, args.output)
    print(f"Wrote {len(results):,} rows to {args.output}")


if __name__ == "__main__":
    main()
