#!/usr/bin/env python3
"""Evaluate a causal LLM as an English compressor."""
from __future__ import annotations
import argparse, bz2, json, lzma, math, time, zlib
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def encode_varints(values: Iterable[int]) -> bytes:
    out = bytearray()
    for v in values:
        n = int(v)
        if n < 0: raise ValueError("rank must be non-negative")
        while n >= 0x80:
            out.append((n & 0x7F) | 0x80); n >>= 7
        out.append(n)
    return bytes(out)

def infer_context_length(model, requested: int | None) -> int:
    if requested is not None: return requested
    for name in ("n_positions", "max_position_embeddings", "seq_length"):
        value = getattr(model.config, name, None)
        if isinstance(value, int) and value > 0: return min(value, 4096)
    return 1024

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-path", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--max-chars", type=int, default=100_000)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--save-ranks", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    text = Path(args.text_path).read_text(encoding="utf-8", errors="replace")[: args.max_chars]
    if not text: raise SystemExit(f"No text read from {args.text_path}")
    dtype = {"auto": "auto", "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    print(f"Loading tokenizer/model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, trust_remote_code=args.trust_remote_code)
    model.eval(); model.to(args.device)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    seq_len = int(ids.numel())
    if seq_len < 2: raise SystemExit("Need at least two tokens to score next-token prediction")
    context_length = infer_context_length(model, args.context_length)
    stride = min(args.stride, context_length)
    if stride <= 0: raise SystemExit("--stride must be positive")
    total_nll_nats = 0.0; total_scored = 0; ranks: list[int] = []; rank_zero = 0; prev_end = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for begin in tqdm(range(0, seq_len, stride), desc="scoring"):
            end = min(begin + context_length, seq_len)
            if end <= begin + 1: break
            trg_len = end - prev_end
            score_start = max(begin + 1, end - trg_len)
            if score_start >= end:
                prev_end = end
                if end == seq_len: break
                continue
            window = ids[begin:end].unsqueeze(0).to(args.device)
            logits = model(window).logits[0]

            # Vectorized scoring for the newly exposed target positions.
            # For target global_pos, prediction comes from local_pos - 1.
            target_global = torch.arange(score_start, end, device=args.device)
            pred_local = target_global - begin - 1
            target_ids = ids[score_start:end].to(args.device)
            pred_logits_batch = logits.index_select(0, pred_local).float()
            log_probs_batch = torch.log_softmax(pred_logits_batch, dim=-1)
            nll_batch = -log_probs_batch.gather(1, target_ids.view(-1, 1)).squeeze(1)
            total_nll_nats += float(nll_batch.sum().item())
            total_scored += int(target_ids.numel())

            if args.save_ranks:
                true_logits = pred_logits_batch.gather(1, target_ids.view(-1, 1)).squeeze(1)
                rank_batch = torch.sum(pred_logits_batch > true_logits.view(-1, 1), dim=1).to(torch.int64).cpu().numpy()
                ranks.extend(int(x) for x in rank_batch)
                rank_zero += int(np.sum(rank_batch == 0))
            prev_end = end
            if end == seq_len: break
    elapsed = time.perf_counter() - t0
    total_bits = total_nll_nats / math.log(2.0); chars = len(text)
    result: dict[str, object] = {"text_path": args.text_path, "model": args.model, "chars": chars, "tokens_total": seq_len, "tokens_scored": total_scored, "context_length": context_length, "stride": stride, "device": args.device, "dtype": args.dtype, "total_ideal_bits": total_bits, "bits_per_token": total_bits / total_scored, "bits_per_character": total_bits / chars, "elapsed_seconds": elapsed}
    if args.save_ranks:
        rank_array = np.asarray(ranks, dtype=np.uint32)
        raw_u32 = rank_array.astype("<u4", copy=False).tobytes(); raw_varint = encode_varints(ranks)
        result.update({"rank_count": len(ranks), "rank_zero_fraction": rank_zero / max(1, len(ranks)), "rank_u32_zlib_bits_per_character": 8.0 * len(zlib.compress(raw_u32, level=9)) / chars, "rank_varint_zlib_bits_per_character": 8.0 * len(zlib.compress(raw_varint, level=9)) / chars, "rank_varint_bz2_bits_per_character": 8.0 * len(bz2.compress(raw_varint, compresslevel=9)) / chars, "rank_varint_lzma_bits_per_character": 8.0 * len(lzma.compress(raw_varint, preset=9)) / chars})
        rank_path = Path(args.out).with_suffix(".ranks.uint32.npy"); rank_path.parent.mkdir(parents=True, exist_ok=True); np.save(rank_path, rank_array); result["rank_path"] = str(rank_path)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2)); print(f"Wrote {out}")
if __name__ == "__main__": main()
