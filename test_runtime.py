#!/usr/bin/env python3
"""Self-checks for meld_runtime.py. Run with `python test_runtime.py` (pytest also works)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from meld_runtime import (
    DEMO_ROWS,
    EMOTIONS,
    STANCE_TABLE,
    EnergyVAD,
    MultimodalEmotionRuntime,
    extract_prosody_features,
    format_dialogue_context,
    format_reply,
    load_meld_turn,
    resolve_stance,
)


def test_stance_lookup_table() -> None:
    assert set(STANCE_TABLE.keys()) == set(EMOTIONS)
    assert resolve_stance("joy") == "with_you"
    assert resolve_stance("anger") == "pushback"
    assert resolve_stance("disgust") == "pushback"
    assert resolve_stance("sadness") == "caution"
    assert resolve_stance("fear") == "caution"
    assert resolve_stance("neutral") == "unclear"
    assert resolve_stance("surprise") == "unclear"
    assert resolve_stance("surprise", probs={"joy": 0.30, "anger": 0.02, "sadness": 0.02}) == "with_you"
    assert resolve_stance("surprise", probs={"joy": 0.02, "anger": 0.25, "sadness": 0.05}) == "caution"
    assert resolve_stance("surprise", probs={"joy": 0.10, "anger": 0.08}) == "unclear"


def test_reply_template() -> None:
    r_joy = format_reply("Oh, yeah!", emotion="joy", stance="with_you", speaker="Rachel")
    assert '"Oh, yeah!"' in r_joy
    assert "joy" in r_joy
    assert "agreement" in r_joy

    r_ang = format_reply("Oh, yeah!", emotion="anger", stance="pushback", speaker="Rachel")
    assert '"Oh, yeah!"' in r_ang
    assert "anger" in r_ang
    assert "Do not count that as a yes" in r_ang
    assert r_joy != r_ang


def test_dialogue_history_and_vad() -> None:
    history = [
        {"speaker": "Chandler", "text": "Wait a minute."},
        {"speaker": "Monica", "text": "What is it?"},
        {"speaker": "Ross", "text": "You liked it? You really liked it?"},
    ]
    ctx = format_dialogue_context("Oh, yeah!", speaker="Rachel", history=history, max_history=2)
    assert "Chandler" not in ctx  # older than max_history=2
    assert "Monica: What is it?" in ctx
    assert "Ross: You liked it? You really liked it?" in ctx
    assert ctx.endswith("Rachel: Oh, yeah!")

    # Two 0.4 s tones separated by 0.7 s of silence -> two turns with a 600 ms cutoff.
    sr = 16000
    t = np.linspace(0, 0.4, int(0.4 * sr), endpoint=False, dtype=np.float32)
    tone = 0.2 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    pause = np.zeros(int(0.7 * sr), dtype=np.float32)
    stream = np.concatenate([tone, pause, tone, pause])

    vad = EnergyVAD(sr=sr, pause_ms=600, min_speech_ms=200)
    turns = vad.push_chunk(stream)
    assert len(turns) == 2, f"Expected 2 segmented turns, got {len(turns)}"
    pros = extract_prosody_features(tone, sr=sr)
    assert pros.shape == (6,)
    assert abs(pros[3] * 300.0 - 220.0) < 10.0, f"pitch tracker read {pros[3] * 300.0:.0f} Hz for a 220 Hz tone"


def test_end_to_end_turn_and_parameter_cap() -> None:
    rt = MultimodalEmotionRuntime(load_asr=True)
    params = rt.parameter_breakdown()
    assert params["within_6b_cap"] is True
    assert 0 < params["live_mic_path_total"] <= 6_000_000_000

    df = pd.read_csv("data/meld/test.csv")
    for i in DEMO_ROWS:
        turn = load_meld_turn(df, "test", i)
        kwargs = dict(text=turn["text"], speaker=turn["speaker"], history=turn["history"])
        fused = rt.process_turn(audio=Path("assets") / turn["audio"].name, **kwargs)
        text_only = rt.process_turn(**kwargs)

        assert set(fused.state) == {"emotion", "confidence", "listener_stance"}
        assert 0.0 < fused.state["confidence"] <= 1.0
        assert fused.state["listener_stance"] == resolve_stance(
            fused.state["emotion"], probs=fused.modality_info["probabilities"]
        )
        assert f'"{" ".join(turn["text"].split())}"' in fused.reply
        assert fused.modality_info["probabilities"] != text_only.modality_info["probabilities"], "audio had no effect"
        # Pins the README demo to the checked-in head. After retraining, pick a new pair if this fails.
        assert fused.state["emotion"] == turn["label"], f"row {i}: got {fused.state['emotion']}, gold {turn['label']}"


def main() -> None:
    test_stance_lookup_table()
    test_reply_template()
    test_dialogue_history_and_vad()
    test_end_to_end_turn_and_parameter_cap()
    print("All checks passed.")


if __name__ == "__main__":
    main()
