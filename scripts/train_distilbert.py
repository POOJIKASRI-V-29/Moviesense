"""Milestone 4: fine-tune DistilBERT for multi-label genre classification (report §10.1).

Same protocol as the baseline (report §13.2): identical splits, labels and training texts
(full summaries + the same seeded 1-3 sentence windows), thresholds tuned per genre on validation
predictions pooled across conditions, early stopping on validation single-sentence macro-F1.
Writes models/distilbert/{encoder/,head.pt,config.json,preds_seed*.npz} and
reports/distilbert_validation.json. Designed for a GPU (Colab T4: see notebooks/).

    python scripts/train_distilbert.py --seeds 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moviesense.config import DATASET_FILE, MODELS, REPORTS
from moviesense.data.augment import eval_condition_texts, training_texts
from moviesense.data.dataset import EVAL_CONDITIONS, PRIMARY_CONDITIONS, load_splits
from moviesense.models.metrics import apply_thresholds, classification_report, tune_thresholds
from moviesense.models.transformer import (BASE_CHECKPOINT, GenreClassifier, TransformerModel,
                                           best_device)

OUT = MODELS / "distilbert"


def length_bucketed_batches(lengths, batch_size, rng, bucket=50):
    """Shuffle, then sort inside mega-batches by length so padding is small."""
    idx = rng.permutation(len(lengths))
    batches = []
    for s in range(0, len(idx), batch_size * bucket):
        chunk = sorted(idx[s:s + batch_size * bucket], key=lambda i: lengths[i])
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--windows", type=int, default=2, help="must match the baseline (default 2)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max-length", type=int, default=256,
                    help="training truncation; inference uses up to 512")
    ap.add_argument("--pos-weight-cap", type=float, default=20.0)
    ap.add_argument("--limit-train", type=int, default=0, help="debug: subsample training movies")
    args = ap.parse_args()

    device = best_device()
    use_amp = device.type == "cuda"
    print("device:", device, "amp:", use_amp, flush=True)
    splits = load_splits()
    train_df = splits.train.sample(n=args.limit_train, random_state=0) if args.limit_train else splits.train
    labels = splits.labels
    Ytr_movies = splits.Y(train_df)
    Yv = splits.Y(splits.val)
    val_texts = {c: eval_condition_texts(splits.val, c) for c in EVAL_CONDITIONS}
    test_texts = {c: eval_condition_texts(splits.test, c) for c in EVAL_CONDITIONS}
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)
    data_hash = hashlib.sha256(DATASET_FILE.read_bytes()).hexdigest()[:16]

    report = {"model": "distilbert", "base_checkpoint": BASE_CHECKPOINT, "args": vars(args),
              "device": str(device), "dataset_hash": data_hash, "runs": []}
    OUT.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        set_seed(seed)
        tt = training_texts(train_df, args.windows, seed)  # identical windows to the baseline
        row = {w: i for i, w in enumerate(train_df["wiki_id"])}
        texts = tt["text"].tolist()
        Y = Ytr_movies[[row[w] for w in tt["wiki_id"]]].astype(np.float32)
        pos = Y.sum(0)
        pos_weight = torch.tensor(np.clip((len(Y) - pos) / np.maximum(pos, 1), 1.0,
                                          args.pos_weight_cap), dtype=torch.float32, device=device)
        lengths = [len(t) for t in texts]
        rng = np.random.default_rng(seed)

        model = GenreClassifier(BASE_CHECKPOINT, len(labels)).to(device)
        wrapper = TransformerModel(model, tokenizer, labels, np.full(len(labels), 0.5),
                                   max_length=512, device=device)
        no_decay = ("bias", "LayerNorm.weight")
        params = [{"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
                   "weight_decay": args.weight_decay},
                  {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)],
                   "weight_decay": 0.0}]
        opt = torch.optim.AdamW(params, lr=args.lr)
        steps_per_epoch = math.ceil(len(texts) / args.batch_size)
        total = steps_per_epoch * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        history, best, best_state, bad = [], -1.0, None, 0
        print(f"seed {seed}: {len(texts)} training texts, {steps_per_epoch} steps/epoch", flush=True)
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, running = time.time(), 0.0
            for step, b in enumerate(length_bucketed_batches(lengths, args.batch_size, rng), 1):
                enc = tokenizer([texts[i] for i in b], truncation=True, max_length=args.max_length,
                                padding=True, return_tensors="pt").to(device)
                yb = torch.from_numpy(Y[b]).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    logits = model(enc["input_ids"], enc["attention_mask"])
                loss = loss_fn(logits.float(), yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                running += loss.item()
                if step % 200 == 0:
                    print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss {running / step:.4f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
            model.eval()
            vprobs = {c: wrapper.predict_proba(val_texts[c]) for c in EVAL_CONDITIONS}
            taus = tune_thresholds(np.vstack([Yv] * 3), np.vstack([vprobs[c] for c in EVAL_CONDITIONS]))
            reps = {c: classification_report(Yv, apply_thresholds(vprobs[c], taus), labels)
                    for c in EVAL_CONDITIONS}
            primary = float(np.mean([reps[c]["macro_f1"] for c in PRIMARY_CONDITIONS]))
            history.append({"epoch": epoch, "train_loss": running / steps_per_epoch,
                            "val_primary_macro_f1": primary,
                            "val_macro_f1": {c: reps[c]["macro_f1"] for c in EVAL_CONDITIONS},
                            "epoch_seconds": time.time() - t0})
            print(f"seed {seed} epoch {epoch}: val primary macro-F1 {primary:.4f} "
                  f"full {reps['full']['macro_f1']:.4f}", flush=True)
            if primary > best:
                best, bad = primary, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stopping", flush=True)
                    break

        model.load_state_dict(best_state)
        model.eval()
        vprobs = {c: wrapper.predict_proba(val_texts[c]) for c in EVAL_CONDITIONS}
        taus = tune_thresholds(np.vstack([Yv] * 3), np.vstack([vprobs[c] for c in EVAL_CONDITIONS]))
        reps = {c: classification_report(Yv, apply_thresholds(vprobs[c], taus), labels)
                for c in EVAL_CONDITIONS}
        tprobs = {c: wrapper.predict_proba(test_texts[c]) for c in EVAL_CONDITIONS}
        np.savez_compressed(OUT / f"preds_seed{seed}.npz", thresholds=taus,
                            **{f"val_{c}": vprobs[c] for c in EVAL_CONDITIONS},
                            **{f"test_{c}": tprobs[c] for c in EVAL_CONDITIONS})
        primary = float(np.mean([reps[c]["macro_f1"] for c in PRIMARY_CONDITIONS]))
        report["runs"].append({"seed": seed, "history": history, "val_primary_macro_f1": primary,
                               "val": {c: {k: v for k, v in r.items() if k != "per_genre_f1"}
                                       for c, r in reps.items()}})
        if seed == args.seeds[0]:
            wrapper.thresholds = taus
            wrapper.meta = {"seed": seed, "dataset_hash": data_hash, "val_primary_macro_f1": primary,
                            "epochs_trained": len(history), "args": vars(args)}
            wrapper.save(OUT)
            print("saved model bundle to", OUT, flush=True)
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()

    vals = [r["val_primary_macro_f1"] for r in report["runs"]]
    report["val_primary_macro_f1_mean"] = float(np.mean(vals))
    report["val_primary_macro_f1_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    report["best"] = {"seed": args.seeds[0], "epochs": args.epochs, "lr": args.lr}
    (REPORTS / "distilbert_validation.json").write_text(json.dumps(report, indent=1))
    print("wrote", REPORTS / "distilbert_validation.json")


if __name__ == "__main__":
    main()
