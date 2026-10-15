#!/usr/bin/env python3
"""
Train and score the fusion head on the official MELD splits (train 9,989 / dev 1,109 / test 2,610).

  1. Cache text and audio vectors for every utterance (data/cache/meld_features.pt) with the
     runtime's own encoders, so training sees the same vectors process_turn computes.
  2. Baselines: concat + one linear layer, with/without anchored history pooling and class weights.
  3. GatedFusionHead: train on train, pick the epoch on dev, score on test as fused, text branch
     only, no history, and audio branch only. Then 3 seeds of fused vs. a head trained with audio zeroed.
  4. RL audio-trust policy: train on dev with 5 seeds, compare with one constant audio weight.
  5. Write checkpoints/fusion_head.pt and artifacts/metrics.json.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from meld_runtime import (
    EMOTION_TO_IDX,
    EMOTIONS,
    GatedFusionHead,
    LinearConcatHead,
    MultimodalEmotionRuntime,
    RLFusionPolicy,
    format_dialogue_context,
)

SEED = 42


def build_split_records(split: str, max_history: int = 3) -> List[Dict[str, Any]]:
    csv_path = Path(f"data/meld/{split}.csv")
    df = pd.read_csv(csv_path)
    records: List[Dict[str, Any]] = []

    # History is the earlier turns of the same dialogue, never later ones.
    for dia_id, group in df.groupby("Dialogue_ID", sort=False):
        group_sorted = group.sort_values("Utterance_ID")
        history_buf: List[Dict[str, str]] = []
        for _, row in group_sorted.iterrows():
            utt_id = int(row["Utterance_ID"])
            speaker = str(row["Speaker"]).strip()
            utterance = str(row["Utterance"]).strip()
            emotion = str(row["Emotion"]).strip().lower()
            sentiment = str(row["Sentiment"]).strip().lower()
            audio_path = Path(f"data/meld/audio/{split}/dia{int(dia_id)}_utt{utt_id}.flac")

            ctx_str = format_dialogue_context(
                utterance=utterance,
                speaker=speaker,
                history=history_buf[-max_history:],
                max_history=max_history,
            )
            nohist_str = format_dialogue_context(
                utterance=utterance,
                speaker=speaker,
                history=[],
                max_history=0,
            )

            records.append(
                {
                    "split": split,
                    "dialogue_id": int(dia_id),
                    "utterance_id": utt_id,
                    "speaker": speaker,
                    "utterance": utterance,
                    "emotion": emotion,
                    "label": EMOTION_TO_IDX[emotion],
                    "sentiment": sentiment,
                    "audio_path": str(audio_path),
                    "ctx_str": ctx_str,
                    "nohist_str": nohist_str,
                    "history": list(history_buf[-max_history:]),
                }
            )
            history_buf.append({"speaker": speaker, "text": utterance})

    return records


@torch.no_grad()
def extract_or_load_features(
    runtime: MultimodalEmotionRuntime,
    cache_path: Path = Path("data/cache/meld_features.pt"),
) -> Dict[str, Dict[str, Any]]:
    if cache_path.exists():
        print(f"[cache] Loading cached MELD embeddings from {cache_path}...")
        return torch.load(str(cache_path), map_location="cpu", weights_only=False)
    if not Path("data/meld/audio/test").is_dir():
        raise SystemExit("MELD audio not found in data/meld/audio/. Run `python download_meld.py` first.")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    device = runtime.device
    data_out: Dict[str, Dict[str, Any]] = {}

    for split in ["train", "dev", "test"]:
        records = build_split_records(split)
        n = len(records)
        print(f"[extract] Split '{split}': {n} utterances on {device}...")

        ctx_texts = [r["ctx_str"] for r in records]
        nohist_texts = [r["nohist_str"] for r in records]

        # Same tokenizer settings and masked mean pooling as runtime.encode_text, batched for speed.
        def batch_encode_texts(texts: List[str], batch_size: int = 128) -> torch.Tensor:
            vecs: List[torch.Tensor] = []
            for i in tqdm(range(0, len(texts), batch_size), desc=f"  text ({split})", leave=False):
                batch = texts[i : i + batch_size]
                enc = runtime.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                ).to(device)
                out = runtime.text_encoder(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                vecs.append(pooled.cpu())
            return torch.cat(vecs, dim=0)

        text_ctx = batch_encode_texts(ctx_texts)
        text_nohist = batch_encode_texts(nohist_texts)

        # One clip at a time through the runtime's encoder. Batching zero-pads short clips, and
        # wav2vec2 without an attention mask lets the padding change the real frames' outputs.
        audio = torch.zeros(n, 262)
        valid_audio = torch.zeros(n, dtype=torch.bool)
        for i, r in enumerate(tqdm(records, desc=f"  audio ({split})", leave=False)):
            if Path(r["audio_path"]).exists():
                audio[i] = runtime.encode_audio(r["audio_path"]).cpu()
                valid_audio[i] = True
        print(f"  {int((~valid_audio).sum())} {split} clips missing; their audio vector stays zero")

        labels = torch.tensor([r["label"] for r in records], dtype=torch.long)
        word_counts = torch.tensor([len(r["utterance"].split()) for r in records], dtype=torch.long)

        data_out[split] = {
            "text_ctx": text_ctx,
            "text_nohist": text_nohist,
            "audio_superb": audio,
            "labels": labels,
            "valid_audio": valid_audio,
            "word_counts": word_counts,
            "records": records,
        }

    torch.save(data_out, str(cache_path))
    print(f"[cache] Saved extracted features to {cache_path}")
    return data_out


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    w_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    _, _, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(EMOTIONS))), zero_division=0
    )
    return {
        "weighted_f1": round(w_f1, 4),
        "macro_f1": round(m_f1, 4),
        "accuracy": round(acc, 4),
        "per_class_f1": {EMOTIONS[i]: round(float(per_f1[i]), 4) for i in range(len(EMOTIONS))},
        "support": {EMOTIONS[i]: int(support[i]) for i in range(len(EMOTIONS))},
    }


def compute_class_weights(labels: torch.Tensor, power: float = 0.35) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=len(EMOTIONS)).float().clamp(min=1.0)
    inv = (counts.sum() / counts) ** power
    return inv / inv.mean()


def train_naive_linear_baseline(
    feats: Dict[str, Dict[str, Any]],
    audio_key: str = "audio_superb",
    class_weight_power: float = 0.0,
    context_weight: float = 1.0,
    epochs: int = 25,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    x_t_train = (1.0 - context_weight) * feats["train"]["text_nohist"] + context_weight * feats["train"]["text_ctx"]
    x_a_train = feats["train"][audio_key]
    y_train = feats["train"]["labels"]

    x_t_dev = (1.0 - context_weight) * feats["dev"]["text_nohist"] + context_weight * feats["dev"]["text_ctx"]
    x_a_dev = feats["dev"][audio_key]
    y_dev = feats["dev"]["labels"]

    x_t_test = (1.0 - context_weight) * feats["test"]["text_nohist"] + context_weight * feats["test"]["text_ctx"]
    x_a_test = feats["test"][audio_key]
    y_test = feats["test"]["labels"].numpy()

    model = LinearConcatHead(text_dim=x_t_train.shape[1], audio_dim=x_a_train.shape[1], num_classes=7)
    if class_weight_power > 0:
        weights = compute_class_weights(y_train, power=class_weight_power)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    batch_size = 256
    best_score = -1.0
    best_state = None
    best_dev: Dict[str, Any] = {}

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(y_train))
        for i in range(0, len(y_train), batch_size):
            idx = perm[i : i + batch_size]
            logits = model(x_t_train[idx], x_a_train[idx], mode="fused")
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            dev_preds = model(x_t_dev, x_a_dev, mode="fused").argmax(dim=-1).numpy()
            dev_metrics = evaluate_predictions(y_dev.numpy(), dev_preds)
            score = 0.6 * dev_metrics["weighted_f1"] + 0.4 * dev_metrics["macro_f1"]
            if score > best_score:
                best_score = score
                best_dev = dev_metrics
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_preds = model(x_t_test, x_a_test, mode="fused").argmax(dim=-1).numpy()
    return {"dev": best_dev, "test": evaluate_predictions(y_test, test_preds)}


def train_gated_fusion_head(
    feats: Dict[str, Dict[str, Any]],
    epochs: int = 35,
    lr: float = 1.5e-3,
    class_weight_power: float = 0.25,
    context_weight: float = 0.25,
) -> Tuple[GatedFusionHead, Dict[str, Any]]:
    """Train GatedFusionHead on text vectors mixed as 0.75 * current turn + 0.25 * history pool."""
    x_t_train = (1.0 - context_weight) * feats["train"]["text_nohist"] + context_weight * feats["train"]["text_ctx"]
    x_nh_train = feats["train"]["text_nohist"]
    x_a_train = feats["train"]["audio_superb"]
    y_train = feats["train"]["labels"]

    x_t_dev = (1.0 - context_weight) * feats["dev"]["text_nohist"] + context_weight * feats["dev"]["text_ctx"]
    x_a_dev = feats["dev"]["audio_superb"]
    y_dev = feats["dev"]["labels"]

    head = GatedFusionHead(text_dim=768, audio_dim=262, hidden_dim=128, num_classes=7, dropout=0.20)
    cls_weights = compute_class_weights(y_train, power=class_weight_power)
    criterion_weighted = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=0.03)
    criterion_standard = nn.CrossEntropyLoss(label_smoothing=0.02)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=2e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 256
    best_dev_score = -1.0
    best_state = None
    best_dev: Dict[str, Any] = {}

    for _ in range(epochs):
        head.train()
        perm = torch.randperm(len(y_train))
        for i in range(0, len(y_train), batch_size):
            idx = perm[i : i + batch_size]
            # Drop history for 15% of turns so the head also works on a conversation's first turn.
            use_ctx = (torch.rand(len(idx), 1) > 0.15).float()
            bt = use_ctx * x_t_train[idx] + (1.0 - use_ctx) * x_nh_train[idx]
            ba = x_a_train[idx]
            by = y_train[idx]

            t_logits, a_logits, gate, _, _ = head.compute_branches(bt, ba)
            fused_logits = t_logits + gate * a_logits

            loss_fused = 0.55 * criterion_standard(fused_logits, by) + 0.45 * criterion_weighted(
                fused_logits, by
            )
            loss_text = 0.55 * criterion_standard(t_logits, by) + 0.45 * criterion_weighted(
                t_logits, by
            )
            loss_audio = criterion_weighted(a_logits, by)

            # Gate target: open when only the audio branch is right, closed when only the text branch is.
            with torch.no_grad():
                t_correct = (t_logits.argmax(dim=-1) == by).float().unsqueeze(-1)
                a_correct = (a_logits.argmax(dim=-1) == by).float().unsqueeze(-1)
                target_gate = torch.where(
                    (a_correct > 0.5) & (t_correct < 0.5),
                    torch.full_like(gate, 0.85),
                    torch.where(
                        (t_correct > 0.5) & (a_correct < 0.5),
                        torch.full_like(gate, 0.15),
                        torch.full_like(gate, 0.40),
                    ),
                )
            loss_gate = F.mse_loss(gate, target_gate)

            loss = loss_fused + 0.35 * loss_text + 0.25 * loss_audio + 0.15 * loss_gate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        head.eval()
        with torch.no_grad():
            dev_logits = head(x_t_dev, x_a_dev, mode="fused")
            dev_metrics = evaluate_predictions(y_dev.numpy(), dev_logits.argmax(dim=-1).numpy())
            score = 0.55 * dev_metrics["weighted_f1"] + 0.45 * dev_metrics["macro_f1"]
            if score > best_dev_score:
                best_dev_score = score
                best_dev = dev_metrics
                best_state = {k: v.clone() for k, v in head.state_dict().items()}

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head, best_dev


def train_rl_fusion_policy(
    head: GatedFusionHead,
    feats: Dict[str, Dict[str, Any]],
    epochs: int = 40,
    lr: float = 5e-3,
    context_weight: float = 0.25,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    REINFORCE on the frozen head's dev-split outputs. Reward is a class-weighted 1/0 for a correct
    prediction, measured against the alpha = 1 head as baseline. Epoch picked by dev macro F1.
    """
    torch.manual_seed(seed)
    head.eval()
    policy = RLFusionPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    x_t_dev = (1.0 - context_weight) * feats["dev"]["text_nohist"] + context_weight * feats["dev"]["text_ctx"]
    x_a_dev = feats["dev"]["audio_superb"]
    y_dev = feats["dev"]["labels"]

    x_t_test = (1.0 - context_weight) * feats["test"]["text_nohist"] + context_weight * feats["test"]["text_ctx"]
    x_a_test = feats["test"]["audio_superb"]
    y_test = feats["test"]["labels"].numpy()

    with torch.no_grad():
        t_logits_dev, a_logits_dev, gate_dev, t_norm_dev, a_norm_dev = head.compute_branches(
            x_t_dev, x_a_dev
        )
        t_logits_test, a_logits_test, gate_test, t_norm_test, a_norm_test = head.compute_branches(
            x_t_test, x_a_test
        )

    before_preds = (t_logits_test + gate_test * a_logits_test).argmax(dim=-1).numpy()
    before_metrics = evaluate_predictions(y_test, before_preds)

    cls_weights = compute_class_weights(y_dev, power=0.55)
    baseline_preds_dev = (t_logits_dev + gate_dev * a_logits_dev).argmax(dim=-1)
    baseline_reward = (
        (baseline_preds_dev == y_dev).float() * cls_weights[y_dev]
    )

    best_dev_macro = -1.0
    best_policy_state = {k: v.clone() for k, v in policy.state_dict().items()}

    for _ in range(epochs):
        policy.train()
        action, log_prob = policy.sample_action(t_norm_dev, a_norm_dev, t_logits_dev, a_logits_dev)
        mod_logits = t_logits_dev + action * (gate_dev * a_logits_dev)
        preds = mod_logits.argmax(dim=-1)
        reward = (preds == y_dev).float() * cls_weights[y_dev]
        advantage = reward - baseline_reward

        loss = -(log_prob * advantage.detach()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        policy.eval()
        with torch.no_grad():
            alpha_dev = policy.deterministic_weight(
                t_norm_dev, a_norm_dev, t_logits_dev, a_logits_dev
            )
            dev_p = (t_logits_dev + alpha_dev * (gate_dev * a_logits_dev)).argmax(dim=-1).numpy()
            dev_m = evaluate_predictions(y_dev.numpy(), dev_p)
            if dev_m["macro_f1"] > best_dev_macro:
                best_dev_macro = dev_m["macro_f1"]
                best_policy_state = {k: v.clone() for k, v in policy.state_dict().items()}

    policy.load_state_dict(best_policy_state)
    policy.eval()
    with torch.no_grad():
        alpha_test = policy.deterministic_weight(
            t_norm_test, a_norm_test, t_logits_test, a_logits_test
        )
        after_preds = (t_logits_test + alpha_test * (gate_test * a_logits_test)).argmax(dim=-1).numpy()
    after_metrics = evaluate_predictions(y_test, after_preds)
    dev_before = evaluate_predictions(y_dev.numpy(), baseline_preds_dev.numpy())["macro_f1"]

    return {
        "seed": seed,
        "dev_macro_f1_before": dev_before,
        "dev_macro_f1_after": best_dev_macro,
        "mean_test_alpha": round(float(alpha_test.mean()), 3),
        "before_rl": before_metrics,
        "after_rl": after_metrics,
        "macro_f1_delta": round(after_metrics["macro_f1"] - before_metrics["macro_f1"], 4),
        "weighted_f1_delta": round(
            after_metrics["weighted_f1"] - before_metrics["weighted_f1"], 4
        ),
    }


def constant_alpha_baseline(head: GatedFusionHead, feats: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """What the RL policy has to beat: one audio weight for every turn, picked on dev macro F1."""
    branches = {}
    with torch.no_grad():
        for split in ("dev", "test"):
            x_t = 0.75 * feats[split]["text_nohist"] + 0.25 * feats[split]["text_ctx"]
            t_logits, a_logits, gate, _, _ = head.compute_branches(x_t, feats[split]["audio_superb"])
            branches[split] = (t_logits, gate * a_logits, feats[split]["labels"].numpy())

    def preds(split: str, alpha: float) -> np.ndarray:
        t_logits, gated_audio, _ = branches[split]
        return (t_logits + alpha * gated_audio).argmax(dim=-1).numpy()

    alphas = [float(round(a, 2)) for a in np.arange(0.5, 2.01, 0.05)]
    best = max(alphas, key=lambda a: f1_score(branches["dev"][2], preds("dev", a), average="macro"))
    return {"alpha": best, "test": evaluate_predictions(branches["test"][2], preds("test", best))}


def mean_std(values: List[float]) -> Dict[str, float]:
    return {"mean": round(float(np.mean(values)), 4), "std": round(float(np.std(values)), 4)}


def main() -> None:
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    runtime = MultimodalEmotionRuntime(checkpoint_path=None, load_asr=False)
    feats = extract_or_load_features(runtime)
    # Seed here, not at import: a run that extracted features trained different heads than a cached run.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("[baselines] concat + linear head...")
    baselines = {
        "linear_pooled_history": train_naive_linear_baseline(feats, context_weight=1.0),
        "linear_anchored_history": train_naive_linear_baseline(feats, context_weight=0.25),
        "linear_anchored_class_weighted": train_naive_linear_baseline(
            feats, context_weight=0.25, class_weight_power=0.25
        ),
    }

    print("[train] GatedFusionHead...")
    head, head_dev = train_gated_fusion_head(feats)

    x_t_test = 0.75 * feats["test"]["text_nohist"] + 0.25 * feats["test"]["text_ctx"]
    x_nh_test = feats["test"]["text_nohist"]
    x_a_test = feats["test"]["audio_superb"]
    y_test = feats["test"]["labels"].numpy()

    with torch.no_grad():
        preds_fused = head(x_t_test, x_a_test, mode="fused").argmax(dim=-1).numpy()
        preds_text_only = head(x_t_test, x_a_test, mode="text_only").argmax(dim=-1).numpy()
        preds_no_history = head(x_nh_test, x_a_test, mode="fused").argmax(dim=-1).numpy()
        preds_audio_only = head(x_t_test, x_a_test, mode="audio_only").argmax(dim=-1).numpy()

    ablations = {
        "fused_with_history": evaluate_predictions(y_test, preds_fused),
        "text_only_with_history": evaluate_predictions(y_test, preds_text_only),
        "fused_no_history": evaluate_predictions(y_test, preds_no_history),
        "audio_only": evaluate_predictions(y_test, preds_audio_only),
        "always_neutral": evaluate_predictions(y_test, np.zeros_like(y_test)),
    }
    baselines["gated_fusion_head"] = {"dev": head_dev, "test": ablations["fused_with_history"]}

    short = feats["test"]["word_counts"].numpy() <= 4
    changed = preds_fused != preds_text_only
    audio_effect = {
        "short_turns_le_4_words": int(short.sum()),
        "short_fused_weighted_f1": evaluate_predictions(y_test[short], preds_fused[short])["weighted_f1"],
        "short_text_only_weighted_f1": evaluate_predictions(y_test[short], preds_text_only[short])["weighted_f1"],
        "turns_where_audio_changed_prediction": int(changed.sum()),
        "changed_to_correct": int((changed & (preds_fused == y_test)).sum()),
        "changed_from_correct": int((changed & (preds_text_only == y_test)).sum()),
    }

    # The text branch above was trained next to audio, so also train heads that never see audio.
    print("[seeds] fused head vs. a head trained with audio zeroed, 3 seeds each...")
    no_audio = {s: {**feats[s], "audio_superb": torch.zeros_like(feats[s]["audio_superb"])} for s in feats}
    seed_scores: Dict[str, List[float]] = {
        k: [] for k in ("fused_wf1", "text_only_head_wf1", "fused_mf1", "text_only_head_mf1")
    }
    for s in range(3):
        torch.manual_seed(s)
        h_fused, _ = train_gated_fusion_head(feats)
        torch.manual_seed(s)
        h_text, _ = train_gated_fusion_head(no_audio)
        with torch.no_grad():
            m_f = evaluate_predictions(y_test, h_fused(x_t_test, x_a_test).argmax(dim=-1).numpy())
            m_t = evaluate_predictions(
                y_test, h_text(x_t_test, torch.zeros_like(x_a_test)).argmax(dim=-1).numpy()
            )
        seed_scores["fused_wf1"].append(m_f["weighted_f1"])
        seed_scores["text_only_head_wf1"].append(m_t["weighted_f1"])
        seed_scores["fused_mf1"].append(m_f["macro_f1"])
        seed_scores["text_only_head_mf1"].append(m_t["macro_f1"])
    seed_check = {k: mean_std(v) for k, v in seed_scores.items()}

    print("[rl] REINFORCE audio-trust policy on dev, 5 seeds, vs. one constant audio weight...")
    rl_runs = [train_rl_fusion_policy(head, feats, seed=s) for s in range(5)]
    constant = constant_alpha_baseline(head, feats)
    fused_test = ablations["fused_with_history"]
    rl_summary = {
        "test_macro_f1_delta": mean_std([r["macro_f1_delta"] for r in rl_runs]),
        "test_weighted_f1_delta": mean_std([r["weighted_f1_delta"] for r in rl_runs]),
        "constant_alpha_baseline": {
            "alpha": constant["alpha"],
            "test_macro_f1_delta": round(constant["test"]["macro_f1"] - fused_test["macro_f1"], 4),
            "test_weighted_f1_delta": round(constant["test"]["weighted_f1"] - fused_test["weighted_f1"], 4),
        },
        "runs": rl_runs,
    }

    torch.save(head.state_dict(), "checkpoints/fusion_head.pt")

    summary = {
        "parameter_accounting": runtime.parameter_breakdown(),
        "baselines_dev_and_test": baselines,
        "meld_test_ablations": ablations,
        "audio_effect_on_test": audio_effect,
        "head_seed_check_test": seed_check,
        "rl_policy": rl_summary,
    }
    Path("artifacts/metrics.json").write_text(json.dumps(summary, indent=2))

    print("\n=== written to artifacts/metrics.json ===")
    for name, m in baselines.items():
        print(
            f"  {name:32s} dev wF1={m['dev']['weighted_f1']} mF1={m['dev']['macro_f1']}"
            f" | test wF1={m['test']['weighted_f1']} mF1={m['test']['macro_f1']}"
        )
    for name, m in ablations.items():
        print(f"  test {name:24s} wF1={m['weighted_f1']} mF1={m['macro_f1']} acc={m['accuracy']}")
    print("  audio effect:", audio_effect)
    print("  3-seed check:", seed_check)
    print(
        f"  RL test macro F1 delta {rl_summary['test_macro_f1_delta']['mean']:+.4f}"
        f" ± {rl_summary['test_macro_f1_delta']['std']:.4f} (5 seeds);"
        f" constant alpha {constant['alpha']}: {rl_summary['constant_alpha_baseline']['test_macro_f1_delta']:+.4f}"
    )


if __name__ == "__main__":
    main()
