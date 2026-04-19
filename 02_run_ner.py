"""Extract sub-institution (SUB) and organisation (ORG) entities from raw affiliation strings."""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

from config import LANG_FREQ_CSV, NER_MODEL, NER_OUTPUT_JSONL, SPAN_MODEL
from text_utils import clean_affiliation_string

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_LENGTH = 128
DEFAULT_NUM_WORKERS = 16
DEFAULT_PREFETCH_FACTOR = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=LANG_FREQ_CSV)
    parser.add_argument("--output", type=Path, default=NER_OUTPUT_JSONL)
    parser.add_argument("--span-model", default=SPAN_MODEL)
    parser.add_argument("--ner-model", default=NER_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--loader-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Restrict to the first N rows (useful for quick tests).")
    return parser.parse_args()


class _TokenizedDataset(Dataset):
    """Tokenises on-the-fly so DataLoader workers can parallelise the CPU step."""

    def __init__(self, texts, tokenizer, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "offset_mapping": encoded["offset_mapping"].squeeze(0),
            "index": torch.tensor(idx, dtype=torch.long),
        }


def _split_label(label: str) -> tuple[str, str]:
    if "-" in label:
        prefix, group = label.split("-", 1)
    else:
        prefix, group = "B", label
    return prefix, group


def _finalize_entity(entity: dict[str, Any], text: str) -> dict[str, Any]:
    start = max(0, min(len(text), int(entity["start"])))
    end = max(start, min(len(text), int(entity["end"])))
    avg_score = float(entity["score_sum"]) / max(1, int(entity["token_count"]))
    return {
        "entity_group": entity["entity_group"],
        "start": start,
        "end": end,
        "word": text[start:end],
        "score": avg_score,
    }


def collect_entities(label_ids, label_scores, offsets, attention_mask, text, id2label):
    """Decode BIO-tagged token predictions into entity dicts."""
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for label_id, score, offset, mask in zip(label_ids, label_scores, offsets, attention_mask):
        if int(mask) == 0:
            break
        token_start, token_end = int(offset[0]), int(offset[1])
        if token_end <= token_start:
            continue  # special tokens / padding

        label = id2label.get(int(label_id), "O")
        if label == "O":
            if current:
                entities.append(_finalize_entity(current, text))
                current = None
            continue

        prefix, group = _split_label(label)
        if prefix == "B" or current is None or current["entity_group"] != group:
            if current:
                entities.append(_finalize_entity(current, text))
            current = {
                "entity_group": group, "start": token_start, "end": token_end,
                "score_sum": float(score), "token_count": 1,
            }
        else:
            current["end"] = token_end
            current["score_sum"] += float(score)
            current["token_count"] += 1
    if current:
        entities.append(_finalize_entity(current, text))
    return entities


def merge_broken_entities(entities):
    """Merge consecutive same-type entities whose character spans touch."""
    if not entities:
        return []
    merged = [entities[0].copy()]
    for ent in entities[1:]:
        last = merged[-1]
        if ent["start"] == last["end"] and ent["entity_group"] == last["entity_group"]:
            last["end"] = ent["end"]
            last["word"] += ent["word"]
            last["score"] = (last["score"] + ent["score"]) / 2.0
        else:
            merged.append(ent.copy())
    return merged


def keep_only_sub_org(entities):
    return [e for e in entities if e["entity_group"] in {"SUB", "ORG"}]


def span_quota(inst_count: Any) -> int:
    try:
        quota = int(inst_count)
    except (TypeError, ValueError):
        quota = 0
    return max(1, quota)


class TokenClassificationRunner:
    """Reusable helper that batches a HuggingFace token-classification model."""

    def __init__(self, model_name, batch_size, max_length, loader_workers,
                 prefetch_factor, device) -> None:
        self.batch_size = batch_size
        self.max_length = max_length
        self.loader_workers = loader_workers
        self.prefetch_factor = prefetch_factor
        self.device = device
        self.use_amp = device.type == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        torch_dtype = torch.float16 if self.use_amp else torch.float32
        model = AutoModelForTokenClassification.from_pretrained(model_name, torch_dtype=torch_dtype)
        if self.use_amp:
            model = model.to(device)
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        self.model = model.eval()
        base = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        self.id2label = base.config.id2label

    def predict(self, texts, desc: str):
        dataset = _TokenizedDataset(texts, self.tokenizer, self.max_length)
        if len(dataset) == 0:
            return []

        loader_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.loader_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.loader_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        loader = DataLoader(dataset, **loader_kwargs)
        results: list[list[dict[str, Any]]] = [[] for _ in range(len(dataset))]

        with torch.inference_mode():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                amp_ctx = (
                    torch.cuda.amp.autocast(dtype=torch.float16)
                    if self.use_amp else contextlib.nullcontext()
                )
                with amp_ctx:
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits

                max_logit, pred_ids = logits.max(dim=-1)
                token_scores = torch.exp(max_logit - torch.logsumexp(logits, dim=-1))

                for pred, score, offset, mask, idx in zip(
                    pred_ids.cpu().tolist(),
                    token_scores.cpu().tolist(),
                    batch["offset_mapping"].cpu().tolist(),
                    batch["attention_mask"].cpu().tolist(),
                    batch["index"].cpu().tolist(),
                ):
                    results[idx] = collect_entities(pred, score, offset, mask, texts[idx], self.id2label)
        return results


def _select_span_candidates(span_predictions):
    """Sort span candidates by length then score (both descending)."""
    candidates_per_row: list[list[str]] = []
    for entities in span_predictions:
        entries = []
        for ent in entities:
            span_text = ent.get("word", "").strip()
            if not span_text:
                continue
            span_len = int(ent.get("end", 0)) - int(ent.get("start", 0))
            entries.append((span_text, span_len, float(ent.get("score", 0.0))))
        entries.sort(key=lambda item: (item[1], item[2]), reverse=True)
        candidates_per_row.append([text for text, _, _ in entries])
    return candidates_per_row


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"Visible GPUs: {torch.cuda.device_count()} (DataParallel enabled when >1)")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    read_kwargs = {"nrows": args.max_rows} if args.max_rows else {}
    df = pd.read_csv(args.input, **read_kwargs)
    print(f"Loaded {len(df):,} rows from {args.input}")

    required = {"raw_affiliation_string", "raw_affiliation_string_id", "institution_count", "frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    if "language" in df.columns:
        before = len(df)
        df = df[df["language"] == "ENGLISH"].copy()
        print(f"Filtered to {len(df):,} ENGLISH rows (from {before:,})")

    cleaned = [clean_affiliation_string(t) for t in tqdm(df["raw_affiliation_string"].tolist(),
                                                         desc="Cleaning raw strings")]
    if not cleaned:
        print("No rows left after filtering. Nothing to do.")
        return

    inst_counts = df["institution_count"].tolist()
    frequencies = df["frequency"].tolist()
    raw_ids = df["raw_affiliation_string_id"].tolist()

    # Stage 1: span extraction.
    span_runner = TokenClassificationRunner(
        args.span_model, args.batch_size, args.max_length,
        args.loader_workers, args.prefetch_factor, device,
    )
    span_predictions = span_runner.predict(cleaned, desc="Span extraction")
    span_candidates = _select_span_candidates(span_predictions)

    del span_runner
    span_predictions.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage 2: entity classification over the top-K spans per row.
    full_entities: list[list[dict[str, Any]]] = [[] for _ in span_candidates]
    ner_inputs: list[str] = []
    ner_row_indices: list[int] = []
    for row_idx, candidates in enumerate(span_candidates):
        if not candidates:
            continue
        for span_text in candidates[:span_quota(inst_counts[row_idx])]:
            ner_inputs.append(span_text)
            ner_row_indices.append(row_idx)

    if ner_inputs:
        ner_runner = TokenClassificationRunner(
            args.ner_model, args.batch_size, args.max_length,
            args.loader_workers, args.prefetch_factor, device,
        )
        ner_predictions = ner_runner.predict(ner_inputs, desc="NER classification")
        for local_idx, row_idx in enumerate(ner_row_indices):
            merged = merge_broken_entities(ner_predictions[local_idx])
            filtered = keep_only_sub_org(merged)
            if filtered:
                full_entities[row_idx].extend(filtered)
    else:
        print("No spans were extracted; skipping NER stage.")

    saved = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for raw_id, inst_count, freq, entities in tqdm(
            zip(raw_ids, inst_counts, frequencies, full_entities),
            total=len(raw_ids),
            desc="Writing JSONL",
        ):
            serializable = [
                {"entity_group": e["entity_group"], "word": e["word"],
                 "start": int(e["start"]), "end": int(e["end"])}
                for e in entities
            ]
            if not serializable:
                continue
            fh.write(json.dumps({
                "raw_affiliation_string_id": int(raw_id),
                "institution_count": int(inst_count),
                "frequency": int(freq),
                "entities": serializable,
            }, ensure_ascii=False) + "\n")
            saved += 1

    print(f"Saved {saved:,} rows with entities to {args.output}")


if __name__ == "__main__":
    main()
