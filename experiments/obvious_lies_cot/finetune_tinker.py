"""Tinker LoRA SFT for DeepSeek-V3.1 on the reasoning-distillation datasets (Section 4.2).

Trains DeepSeek-V3.1 on either the with-CoT or the stripped-CoT
Qwen3-32B-distilled dataset using Tinker's LoRA training client. The
renderer is the model's recommended ``<...>_thinking`` renderer in both
modes — the only difference is the contents of the ``<think>...</think>``
block in the training data.

Run with three different ``--run`` seeds per dataset to reproduce the paper.

Usage:
    python finetune_tinker.py --run 1 --data data/inoculation_with_cot_qwen_common.jsonl
    python finetune_tinker.py --run 1 --data data/inoculation_stripped_cot_qwen_common.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer

from config import (
    BATCH_SIZE,
    DEEPSEEK_MODEL,
    EXPERIMENT_DIR,
    LEARNING_RATE,
    LORA_RANK,
    MAX_LENGTH,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=int, required=True, help="Seed index (1, 2, 3 in the paper).")
    parser.add_argument("--data", type=Path, required=True,
                        help="Training JSONL (chat-format).")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    args = parser.parse_args()

    data_path = args.data if args.data.is_absolute() else ROOT / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"Training file not found: {data_path}")

    version = f"deepseek_{data_path.stem}"
    log_path = str(EXPERIMENT_DIR / f"tinker_logs_{version}_{args.run}")
    sampler_path_file = EXPERIMENT_DIR / f"tinker_sampler_path_{version}_{args.run}.txt"
    Path(log_path).mkdir(parents=True, exist_ok=True)

    print(f"[{version} run={args.run}] base={DEEPSEEK_MODEL}", flush=True)
    print(f"[{version}] data={data_path}", flush=True)

    tokenizer = get_tokenizer(DEEPSEEK_MODEL)
    rec = model_info.get_recommended_renderer_name(DEEPSEEK_MODEL)
    renderer_name = rec + "_thinking"
    try:
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    except Exception:
        renderer_name = rec
        renderer = renderers.get_renderer(renderer_name, tokenizer)
    print(f"[{version}] Renderer: {renderer_name}", flush=True)

    conversations: list[list[dict]] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line)["messages"])
    print(f"[{version}] {len(conversations)} conversations", flush=True)
    random.shuffle(conversations)

    bs = args.batch_size
    n_batches = len(conversations) // bs

    sc = tinker.ServiceClient()
    resume = checkpoint_utils.get_last_checkpoint(log_path)
    if resume:
        tc = sc.create_training_client_from_state_with_optimizer(resume["state_path"])
        start = resume["batch"]
    else:
        tc = sc.create_lora_training_client(base_model=DEEPSEEK_MODEL, rank=args.rank)
        start = 0

    for step in range(start, n_batches):
        t0 = time.time()
        lr_mult = max(0.0, 1.0 - step / n_batches)
        cur_lr = args.lr * lr_mult
        adam = tinker.AdamParams(learning_rate=cur_lr, beta1=0.9, beta2=0.95, eps=1e-8)

        batch = [
            conversation_to_datum(
                conversations[step * bs + j],
                renderer,
                MAX_LENGTH,
                renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
            )
            for j in range(min(bs, len(conversations) - step * bs))
        ]

        fwd = tc.forward_backward(batch, loss_fn="cross_entropy")
        tc.optim_step(adam)
        fwd_r = fwd.result()
        nll = compute_mean_nll(
            [x["logprobs"] for x in fwd_r.loss_fn_outputs],
            [d.loss_fn_inputs["weights"] for d in batch],
        )
        if step % 10 == 0 or step == n_batches - 1:
            print(f"[{version}] step {step}/{n_batches} nll={nll:.4f} {time.time() - t0:.1f}s", flush=True)

    ckpt = checkpoint_utils.save_checkpoint(
        training_client=tc,
        name="final",
        log_path=log_path,
        kind="both",
        loop_state={"batch": n_batches},
    )
    sp = ckpt.get("sampler_path", "")
    if sp:
        sampler_path_file.write_text(sp.strip() + "\n", encoding="utf-8")
    print(f"Done! {version} run={args.run} sampler={sampler_path_file}")


if __name__ == "__main__":
    main()
