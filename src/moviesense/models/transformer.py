"""DistilBERT multi-label genre classifier (report §5, §9.2 stages 3-4, §9.4, §10.1).

encoder (DistilBERT) -> mean pooling over non-padding tokens -> dropout -> Linear(768, G) -> sigmoid.
Explanations: Captum LayerIntegratedGradients on the embedding layer, baseline = [CLS] PAD... [SEP]
(padding embeddings), target = the genre's logit; WordPiece scores are summed to whole words.

Requires `pip install -r requirements-ml.txt` (torch, transformers, captum).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BASE_CHECKPOINT = "distilbert-base-uncased"


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GenreClassifier(nn.Module):
    def __init__(self, encoder_path: str, n_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, n_labels)

    def pool(self, input_ids=None, attention_mask=None, inputs_embeds=None) -> torch.Tensor:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                              inputs_embeds=inputs_embeds).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        return self.head(self.dropout(self.pool(input_ids, attention_mask)))


class TransformerModel:
    name = "distilbert"

    def __init__(self, model: GenreClassifier, tokenizer, labels: list[str], thresholds: np.ndarray,
                 max_length: int = 512, meta: dict | None = None, device: torch.device | None = None):
        self.device = device or best_device()
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.labels = labels
        self.thresholds = np.asarray(thresholds, dtype=float)
        self.max_length = max_length
        self.meta = meta or {}

    # ---------- persistence ----------
    def save(self, path: Path, half: bool = True) -> None:
        """Bundle: encoder weights (fp16 by default to halve disk use), tokenizer, head, labels,
        thresholds, metadata (report §9.1 T6)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        enc = self.model.encoder
        if half:
            enc = copy.deepcopy(enc).cpu().half()
        enc.save_pretrained(path / "encoder", safe_serialization=True)
        self.tokenizer.save_pretrained(path / "encoder")
        torch.save({k: v.float().cpu() for k, v in self.model.head.state_dict().items()},
                   path / "head.pt")
        (path / "config.json").write_text(json.dumps(
            {"labels": self.labels, "thresholds": self.thresholds.tolist(),
             "max_length": self.max_length, "base_checkpoint": BASE_CHECKPOINT,
             "pooling": "mean", "meta": self.meta}, indent=2))

    @classmethod
    def load(cls, path: Path, device: torch.device | None = None) -> "TransformerModel":
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        model = GenreClassifier(str(path / "encoder"), len(cfg["labels"])).float()
        model.head.load_state_dict(torch.load(path / "head.pt", map_location="cpu"))
        tok = AutoTokenizer.from_pretrained(path / "encoder")
        return cls(model, tok, cfg["labels"], np.array(cfg["thresholds"]),
                   cfg.get("max_length", 512), cfg.get("meta", {}), device)

    # ---------- inference ----------
    def _batches(self, texts: list[str], batch_size: int):
        order = np.argsort([len(t) for t in texts])  # length-sorted batches pad less
        for i in range(0, len(texts), batch_size):
            idx = order[i:i + batch_size]
            enc = self.tokenizer([texts[j] for j in idx], truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            yield idx, enc.to(self.device)

    @torch.no_grad()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), len(self.labels)), dtype=np.float32)
        for idx, enc in self._batches(texts, batch_size):
            logits = self.model(enc["input_ids"], enc["attention_mask"])
            out[idx] = torch.sigmoid(logits).float().cpu().numpy()
        return out

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """L2-normalised mean-pooled encoder vectors for retrieval (report §9.1 T7)."""
        out = np.zeros((len(texts), self.model.encoder.config.hidden_size), dtype=np.float32)
        for idx, enc in self._batches(texts, batch_size):
            v = self.model.pool(enc["input_ids"], enc["attention_mask"]).float()
            out[idx] = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        return out

    # ---------- explanations ----------
    def explain(self, text: str, genres: list[str], top_k: int = 8, n_steps: int = 50) -> dict:
        from captum.attr import LayerIntegratedGradients

        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                             return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        word_ids = enc.word_ids(0)
        enc = enc.to(self.device)
        ids, mask = enc["input_ids"], enc["attention_mask"]
        special = torch.tensor([t in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id)
                                for t in ids[0].tolist()], device=self.device)
        baseline = torch.where(special, ids, torch.full_like(ids, self.tokenizer.pad_token_id))

        def forward(input_ids, attention_mask, target):
            return self.model(input_ids, attention_mask)[:, target]

        lig = LayerIntegratedGradients(forward, self.model.encoder.embeddings)
        # Word spans from the tokenizer's own pre-tokenisation.
        spans: dict[int, list[int]] = {}
        for pos, w in enumerate(word_ids):
            if w is not None:
                s, e = offsets[pos]
                spans.setdefault(w, [s, e])
                spans[w][1] = e
        out = {}
        for g in genres:
            target = self.labels.index(g)
            attr, delta = lig.attribute(ids, baselines=baseline, additional_forward_args=(mask, target),
                                        n_steps=n_steps, internal_batch_size=n_steps,
                                        return_convergence_delta=True)
            tok_scores = attr.sum(-1)[0].detach().float().cpu().numpy()
            word_score: dict[int, float] = {}
            for pos, w in enumerate(word_ids):
                if w is not None:
                    word_score[w] = word_score.get(w, 0.0) + float(tok_scores[pos])
            words = [(text[spans[w][0]:spans[w][1]], word_score[w]) for w in sorted(spans)]
            ranked = sorted((ws for ws in words if ws[1] > 0 and any(ch.isalnum() for ch in ws[0])),
                            key=lambda t: -t[1])
            out[g] = {"top_words": [{"word": w, "score": round(s, 4)} for w, s in ranked[:top_k]],
                      "word_scores": [{"word": w, "score": round(s, 4)} for w, s in words],
                      "method": "integrated_gradients", "n_steps": n_steps,
                      "convergence_delta": round(float(delta.abs().max()), 4)}
        return out
