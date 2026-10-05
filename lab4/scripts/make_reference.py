"""Reference outputs from Hugging Face transformers, for lab4's model test.

    python lab4/scripts/make_reference.py            # writes build/lab4/ref/
    python lab4/scripts/make_reference.py --model Qwen3-1.7B

For each prompt it stores, from the fp32 model (bf16 weights upcast):
  - the token ids
  - logits at every position (teacher forcing), float32, [T, vocab] -> <name>.logits.f32
  - a 24-token greedy continuation
Needs ~3 GB of GPU memory for Qwen3-0.6B in fp32. Run it once; the Rust test
reads the files and skips (with a message) when they are missing.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = {
    "kv": "<|im_start|>user\nWhat is a KV cache in LLM inference?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "code": "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "zh": "北京大学的计算机系统研究方向包括",
    "long": "The quick brown fox jumps over the lazy dog. " * 24,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-0.6B")
    ap.add_argument("--out", default=str(ROOT / "build" / "lab4" / "ref"))
    a = ap.parse_args()
    model_dir = Path(os.environ.get("MLSYS_MODELS", Path.home() / "models")) / a.model
    out = Path(a.out) / a.model
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).cuda().eval()
    index = []
    for name, text in PROMPTS.items():
        ids = tok(text, add_special_tokens=False).input_ids
        x = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            logits = model(x).logits[0].float().cpu()
            gen = model.generate(x, max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None)
        logits.numpy().astype("<f4").tofile(out / f"{name}.logits.f32")
        index.append({"name": name, "ids": ids, "greedy": gen[0, len(ids):].tolist(), "vocab": logits.shape[1]})
        print(f"{name}: {len(ids)} tokens, continuation {tok.decode(gen[0, len(ids):])!r}")
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
