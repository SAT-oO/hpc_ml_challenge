#!/usr/bin/env python3
"""
Text + audio emotion runtime on MELD.

process_turn() takes one finished turn (transcript and/or 16 kHz audio) and returns the
emotion state plus a short reply. The demo, the MELD row command, and the mic loop all
call it. train_and_eval.py caches the same encode_text / encode_audio outputs for training.
"""

from __future__ import annotations

import argparse
import collections
import json
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

EMOTIONS: Tuple[str, ...] = (
    "neutral",
    "joy",
    "sadness",
    "anger",
    "fear",
    "disgust",
    "surprise",
)
EMOTION_TO_IDX: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
IDX_TO_EMOTION: Dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}

# MELD's negative emotions split into pushback (anger, disgust) and caution (sadness, fear).
# Surprise can be positive or negative in MELD, so resolve_stance decides it from the probabilities.
STANCE_TABLE: Dict[str, str] = {
    "joy": "with_you",
    "anger": "pushback",
    "disgust": "pushback",
    "sadness": "caution",
    "fear": "caution",
    "neutral": "unclear",
    "surprise": "unclear",
}

AFFIRMATIVE_PATTERNS = re.compile(
    r"\b(yeah|yes|yep|sure|okay|ok|fine|great|right|totally|absolutely|love)\b",
    re.IGNORECASE,
)


def resolve_stance(
    emotion: str,
    probs: Optional[Union[Dict[str, float], Sequence[float], np.ndarray]] = None,
) -> str:
    """Map a MELD emotion to a listener stance. Surprise compares P(joy) with the negative emotions' total."""
    emo = emotion.strip().lower()
    if emo not in STANCE_TABLE:
        raise ValueError(f"Unknown MELD emotion: {emotion!r}. Expected one of {EMOTIONS}")

    if emo == "surprise":
        if probs is not None:
            if isinstance(probs, dict):
                p_pos = float(probs.get("joy", 0.0))
                p_neg = sum(float(probs.get(k, 0.0)) for k in ("anger", "disgust", "sadness", "fear"))
            else:
                arr = np.asarray(probs, dtype=np.float64)
                p_pos = float(arr[EMOTION_TO_IDX["joy"]])
                p_neg = float(
                    arr[EMOTION_TO_IDX["anger"]]
                    + arr[EMOTION_TO_IDX["disgust"]]
                    + arr[EMOTION_TO_IDX["sadness"]]
                    + arr[EMOTION_TO_IDX["fear"]]
                )
            if p_pos > p_neg + 0.05:
                return "with_you"
            if p_neg > p_pos + 0.05:
                return "caution"
        return "unclear"

    return STANCE_TABLE[emo]


def format_reply(
    utterance: str,
    emotion: str,
    stance: str,
    speaker: Optional[str] = None,
) -> str:
    """Fixed templates: quote the words, name the emotion, give one line of advice per stance."""
    clean_quote = " ".join(utterance.strip().split()) or "..."
    emo = emotion.strip().lower()
    prefix = f'{speaker} said "{clean_quote}"' if speaker else f'They said "{clean_quote}"'

    if stance == "with_you":
        return f"{prefix} (emotion: {emo}). They sound pleased. You can treat that as agreement."
    if stance == "pushback":
        if AFFIRMATIVE_PATTERNS.search(clean_quote):
            return (
                f"{prefix} (emotion: {emo}). The words say yes, but the tone is pushback. "
                "Do not count that as a yes."
            )
        return f"{prefix} (emotion: {emo}). The tone is pushback. Ask what the objection is before going on."
    if stance == "caution":
        return (
            f"{prefix} (emotion: {emo}). This reads as caution or concern, not an attack. "
            "Slow down and check in before moving on."
        )
    return (
        f"{prefix} (emotion: {emo}). Their stance is unclear. Ask a quick follow-up before moving on."
    )


def stream_reply_words(
    reply: str,
    delay_s: float = 0.03,
    out=sys.stdout,
) -> List[Tuple[str, float]]:
    """Print the reply word by word and return (word, timestamp) pairs."""
    words = reply.split()
    emitted: List[Tuple[str, float]] = []
    for i, w in enumerate(words):
        sep = "" if i == 0 else " "
        out.write(f"{sep}{w}")
        out.flush()
        emitted.append((w, time.perf_counter()))
        if delay_s > 0 and i + 1 < len(words):
            time.sleep(delay_s)
    out.write("\n")
    out.flush()
    return emitted


def load_waveform_16k(
    audio_input: Union[str, Path, np.ndarray, Tuple[np.ndarray, int]],
    target_sr: int = 16000,
) -> np.ndarray:
    """Load or convert audio into a 1D float32 mono waveform at 16 kHz."""
    if isinstance(audio_input, (str, Path)):
        wav, sr = sf.read(str(audio_input), dtype="float32", always_2d=False)
    elif isinstance(audio_input, tuple) and len(audio_input) == 2:
        wav, sr = np.asarray(audio_input[0], dtype=np.float32), int(audio_input[1])
    else:
        wav, sr = np.asarray(audio_input, dtype=np.float32), target_sr

    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    if sr != target_sr and len(wav) > 0:
        from scipy.signal import resample_poly
        import math

        g = math.gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)

    if len(wav) < 400:
        wav = np.pad(wav, (0, 400 - len(wav)), mode="constant")

    return wav.astype(np.float32)


def extract_prosody_features(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Six prosody features from a 16 kHz mono clip, over 25 ms frames with a 10 ms hop:
      [0] mean log-RMS (loudness)       [3] mean F0 / 300 Hz on voiced frames
      [1] std of log-RMS                [4] std of F0 / 100 Hz
      [2] log(1 + peak / RMS)           [5] share of spectral energy above 1.5 kHz
    """
    eps = 1e-7
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if len(wav) == 0:
        return np.zeros(6, dtype=np.float32)

    rms_global = float(np.sqrt(np.mean(wav * wav) + eps))
    peak = float(np.max(np.abs(wav)) + eps)
    crest = float(np.log1p(peak / rms_global))

    frame_len = int(0.025 * sr)  # 400 samples (25 ms)
    hop_len = int(0.010 * sr)    # 160 samples (10 ms)
    if len(wav) < frame_len:
        wav_padded = np.pad(wav, (0, frame_len - len(wav)))
    else:
        wav_padded = wav

    n_frames = 1 + max(0, (len(wav_padded) - frame_len) // hop_len)
    strides = (wav_padded.strides[0] * hop_len, wav_padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(
        wav_padded, shape=(n_frames, frame_len), strides=strides
    )

    frame_rms = np.sqrt(np.mean(frames * frames, axis=1) + eps)
    log_rms = np.log(frame_rms + 1e-5)
    log_rms_mean = float(np.mean(log_rms))
    log_rms_std = float(np.std(log_rms))

    # Autocorrelation pitch tracker on voiced frames (75 Hz to 450 Hz)
    min_lag = int(sr / 450.0)
    max_lag = int(sr / 75.0)
    voiced_thresh = max(0.01, float(np.percentile(frame_rms, 40)))
    f0_vals: List[float] = []
    for i in range(0, n_frames, 2):
        if frame_rms[i] < voiced_thresh:
            continue
        f = frames[i] - np.mean(frames[i])
        corr = np.correlate(f, f, mode="full")[frame_len - 1 :]
        if corr[0] <= eps or max_lag >= len(corr):
            continue
        segment = corr[min_lag:max_lag]
        if len(segment) == 0:
            continue
        best_rel = int(np.argmax(segment))
        peak_corr = float(segment[best_rel] / (corr[0] + eps))
        if peak_corr > 0.30:
            lag = min_lag + best_rel
            f0_vals.append(float(sr / lag))

    if f0_vals:
        f0_mean_norm = float(np.mean(f0_vals) / 300.0)
        f0_std_norm = float(np.std(f0_vals) / 100.0)
    else:
        f0_mean_norm = 0.0
        f0_std_norm = 0.0

    # High-frequency energy ratio (> 1500 Hz) via FFT
    spec = np.abs(np.fft.rfft(wav_padded)) ** 2
    freqs = np.fft.rfftfreq(len(wav_padded), d=1.0 / sr)
    total_energy = float(np.sum(spec) + eps)
    hf_energy = float(np.sum(spec[freqs >= 1500.0]))
    hf_ratio = hf_energy / total_energy

    return np.array(
        [log_rms_mean, log_rms_std, crest, f0_mean_norm, f0_std_norm, hf_ratio],
        dtype=np.float32,
    )


def format_dialogue_context(
    utterance: str,
    speaker: str = "Speaker",
    history: Optional[Sequence[Dict[str, str]]] = None,
    max_history: int = 3,
) -> str:
    """Join the last `max_history` past turns and the current turn as 'Speaker: text' segments."""
    parts: List[str] = []
    if history:
        for turn in list(history)[-max_history:]:
            prev_spk = turn.get("speaker", "Speaker")
            prev_txt = " ".join(str(turn.get("text", "")).strip().split())
            if prev_txt:
                parts.append(f"{prev_spk}: {prev_txt}")
    cur_txt = " ".join(str(utterance).strip().split())
    parts.append(f"{speaker}: {cur_txt}")
    return " </s></s> ".join(parts)


class LinearConcatHead(nn.Module):
    """Concatenate text and audio vectors, then one linear layer. Baseline in train_and_eval.py."""

    def __init__(self, text_dim: int = 768, audio_dim: int = 262, num_classes: int = 7) -> None:
        super().__init__()
        self.fc = nn.Linear(text_dim + audio_dim, num_classes)

    def forward(self, text_vec: torch.Tensor, audio_vec: torch.Tensor, mode: str = "fused") -> torch.Tensor:
        if mode == "text_only":
            audio_vec = torch.zeros_like(audio_vec)
        elif mode == "audio_only":
            text_vec = torch.zeros_like(text_vec)
        return self.fc(torch.cat([text_vec, audio_vec], dim=-1))


class GatedFusionHead(nn.Module):
    """
    Text and audio each go through LayerNorm and a small MLP to 7 logits. A sigmoid gate,
    computed from both inputs, scales the audio logits per class before they are added:

        logits = text_logits + gate * audio_logits

    The text branch alone is the text-only ablation; the audio branch alone is audio-only.
    """

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 262,
        hidden_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.num_classes = num_classes

        self.text_norm = nn.LayerNorm(text_dim)
        self.audio_norm = nn.LayerNorm(audio_dim)
        self.drop = nn.Dropout(dropout)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def compute_branches(
        self, text_vec: torch.Tensor, audio_vec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_norm = self.text_norm(text_vec)
        a_norm = self.audio_norm(audio_vec)
        t_logits = self.text_proj(self.drop(t_norm))
        a_logits = self.audio_proj(self.drop(a_norm))
        gate = torch.sigmoid(self.gate_net(torch.cat([t_norm, a_norm], dim=-1)))
        return t_logits, a_logits, gate, t_norm, a_norm

    def forward(self, text_vec: torch.Tensor, audio_vec: torch.Tensor, mode: str = "fused") -> torch.Tensor:
        t_logits, a_logits, gate, _, _ = self.compute_branches(text_vec, audio_vec)
        if mode == "text_only":
            return t_logits
        if mode == "audio_only":
            return a_logits
        return t_logits + gate * a_logits


class RLFusionPolicy(nn.Module):
    """
    One-step bandit trained with REINFORCE. Picks an audio-trust multiplier alpha in [0, 2]
    per turn: logits = text_logits + alpha * gate * audio_logits.

    State (7 numbers): text and audio prediction entropy, text and audio top probability,
    whether their top classes disagree, KL(text || audio), and the normalized log-RMS loudness.

    Experiment only (train_and_eval.py). process_turn does not use it: a single constant audio
    weight picked on dev did better.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    @staticmethod
    def build_state_features(
        t_norm: torch.Tensor,
        a_norm: torch.Tensor,
        t_logits: torch.Tensor,
        a_logits: torch.Tensor,
    ) -> torch.Tensor:
        t_probs = F.softmax(t_logits.detach(), dim=-1)
        a_probs = F.softmax(a_logits.detach(), dim=-1)
        t_ent = -torch.sum(t_probs * torch.log(t_probs + 1e-8), dim=-1, keepdim=True) / np.log(7.0)
        a_ent = -torch.sum(a_probs * torch.log(a_probs + 1e-8), dim=-1, keepdim=True) / np.log(7.0)
        t_max = torch.max(t_probs, dim=-1, keepdim=True).values
        a_max = torch.max(a_probs, dim=-1, keepdim=True).values
        disagree = (torch.argmax(t_probs, dim=-1) != torch.argmax(a_probs, dim=-1)).float().unsqueeze(-1)
        kl = torch.sum(t_probs * (torch.log(t_probs + 1e-8) - torch.log(a_probs + 1e-8)), dim=-1, keepdim=True)
        a_energy = a_norm[:, 256:257].detach()  # log_rms_mean, first prosody feature after the 256-d embedding
        return torch.cat([t_ent, a_ent, t_max, a_max, disagree, kl.clamp(0.0, 5.0), a_energy], dim=-1)

    def deterministic_weight(
        self,
        t_norm: torch.Tensor,
        a_norm: torch.Tensor,
        t_logits: torch.Tensor,
        a_logits: torch.Tensor,
    ) -> torch.Tensor:
        feat = self.build_state_features(t_norm, a_norm, t_logits, a_logits)
        return 2.0 * torch.sigmoid(self.net(feat))

    def sample_action(
        self,
        t_norm: torch.Tensor,
        a_norm: torch.Tensor,
        t_logits: torch.Tensor,
        a_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.deterministic_weight(t_norm, a_norm, t_logits, a_logits)
        std = torch.exp(self.log_std).clamp(0.05, 0.75)
        dist = torch.distributions.Normal(mean, std)
        # sample(), not rsample(): with a reparameterized sample, log_prob has zero gradient w.r.t. the mean.
        raw_action = dist.sample()
        action = raw_action.clamp(0.0, 2.0)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        return action, log_prob


class EnergyVAD:
    """
    RMS-energy voice activity detector. Ends a turn after `pause_ms` of quiet that follows speech.
    A fixed threshold false-triggers on loud background noise; a learned VAD (e.g. Silero) would fix that.
    """

    def __init__(
        self,
        sr: int = 16000,
        frame_ms: int = 30,
        pause_ms: int = 600,
        min_speech_ms: int = 200,
        rms_threshold: float = 0.012,
    ) -> None:
        self.sr = sr
        self.frame_samples = int(sr * frame_ms / 1000)
        self.pause_frames = max(1, int(pause_ms / frame_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / frame_ms))
        self.rms_threshold = rms_threshold
        self.reset()

    def reset(self) -> None:
        self.buffer: List[np.ndarray] = []
        self.speech_frames = 0
        self.silence_frames = 0
        self.in_speech = False

    def push_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """Feed arbitrary-length audio samples and return any completed turn waveforms."""
        completed: List[np.ndarray] = []
        flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
        offset = 0
        while offset < len(flat):
            frame = flat[offset : offset + self.frame_samples]
            offset += len(frame)
            rms = float(np.sqrt(np.mean(frame * frame) + 1e-9))
            is_voiced = rms >= self.rms_threshold

            if is_voiced:
                self.in_speech = True
                self.speech_frames += 1
                self.silence_frames = 0
                self.buffer.append(frame)
            elif self.in_speech:
                self.silence_frames += 1
                self.buffer.append(frame)
                if self.silence_frames >= self.pause_frames:
                    if self.speech_frames >= self.min_speech_frames:
                        completed.append(np.concatenate(self.buffer).astype(np.float32))
                    self.reset()
        return completed

    def flush(self) -> Optional[np.ndarray]:
        if self.in_speech and self.speech_frames >= self.min_speech_frames and self.buffer:
            turn = np.concatenate(self.buffer).astype(np.float32)
            self.reset()
            return turn
        self.reset()
        return None


@dataclass
class TurnOutput:
    state: Dict[str, Any]
    reply: str
    transcript: str
    speaker: str
    latency_ms: float
    modality_info: Dict[str, Any] = field(default_factory=dict)


class MultimodalEmotionRuntime:
    """
    Frozen distilroberta-base (text) and superb/wav2vec2-base-superb-er (audio), a trained
    GatedFusionHead, and whisper-tiny.en, loaded only when a turn arrives without a transcript.
    """

    TEXT_MODEL_ID = "distilroberta-base"
    AUDIO_MODEL_ID = "superb/wav2vec2-base-superb-er"
    ASR_MODEL_ID = "openai/whisper-tiny.en"
    PARAMETER_CAP = 6_000_000_000

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = "checkpoints/fusion_head.pt",
        device: Optional[str] = None,
        load_asr: bool = False,
        max_history: int = 3,
    ) -> None:
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.max_history = max_history
        self.history: Deque[Dict[str, str]] = collections.deque(maxlen=max_history)

        from transformers import (
            AutoFeatureExtractor,
            AutoModel,
            AutoTokenizer,
            Wav2Vec2ForSequenceClassification,
            logging as hf_logging,
        )

        hf_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(self.TEXT_MODEL_ID)
        self.text_encoder = AutoModel.from_pretrained(self.TEXT_MODEL_ID).to(self.device)
        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.audio_extractor = AutoFeatureExtractor.from_pretrained(self.AUDIO_MODEL_ID)
        self.audio_encoder = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.AUDIO_MODEL_ID
        ).to(self.device)
        self.audio_encoder.eval()
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        self.head = GatedFusionHead(text_dim=768, audio_dim=262, hidden_dim=128, num_classes=7).to(
            self.device
        )
        if checkpoint_path:
            self.head.load_state_dict(torch.load(str(checkpoint_path), map_location=self.device, weights_only=True))
        self.head.eval()

        self.asr_processor = None
        self.asr_model = None
        if load_asr:
            self._ensure_asr_loaded()

    def _ensure_asr_loaded(self) -> None:
        if self.asr_model is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.asr_processor = WhisperProcessor.from_pretrained(self.ASR_MODEL_ID)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(self.ASR_MODEL_ID).to(
            self.device
        )
        self.asr_model.eval()
        for p in self.asr_model.parameters():
            p.requires_grad = False

    def reset_history(self) -> None:
        self.history.clear()

    @torch.no_grad()
    def transcribe(self, wav: np.ndarray, sr: int = 16000) -> str:
        """Transcribe a 16kHz waveform using local Whisper-tiny.en."""
        self._ensure_asr_loaded()
        wav_16k = load_waveform_16k((wav, sr), target_sr=16000)
        inputs = self.asr_processor(
            wav_16k, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(self.device)
        pred_ids = self.asr_model.generate(inputs, max_new_tokens=64)
        text = self.asr_processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        return text or "..."

    @torch.no_grad()
    def encode_text(
        self,
        utterance: str,
        speaker: str = "Speaker",
        history: Optional[Sequence[Dict[str, str]]] = None,
        context_weight: float = 0.25,
    ) -> torch.Tensor:
        """
        Mean-pool the current turn on its own (z_cur) and history + current turn (z_ctx), then
        return (1 - w) * z_cur + w * z_ctx. Pooling only the joined string lets a long history
        swamp a two-word reply.
        """
        cur_str = format_dialogue_context(utterance=utterance, speaker=speaker, history=[], max_history=0)
        if not history:
            batch_strs = [cur_str]
        else:
            ctx_str = format_dialogue_context(
                utterance=utterance,
                speaker=speaker,
                history=history,
                max_history=self.max_history,
            )
            batch_strs = [cur_str, ctx_str]

        encoded = self.tokenizer(
            batch_strs,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)
        out = self.text_encoder(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        if len(batch_strs) == 1:
            return pooled.squeeze(0)
        return (1.0 - context_weight) * pooled[0] + context_weight * pooled[1]

    @torch.no_grad()
    def encode_audio(
        self,
        audio_input: Union[str, Path, np.ndarray, Tuple[np.ndarray, int]],
        sr: int = 16000,
    ) -> torch.Tensor:
        """
        262-d audio vector: the SUPERB ER embedding (learned layer weights + 256-d projector,
        mean-pooled over time) followed by the 6 prosody features.
        """
        wav = load_waveform_16k(audio_input, target_sr=sr)
        max_samples = 12 * sr
        if len(wav) > max_samples:
            wav = wav[:max_samples]

        prosody = extract_prosody_features(wav, sr=sr)
        inputs = self.audio_extractor(
            wav, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        outputs = self.audio_encoder.wav2vec2(input_values, output_hidden_states=True)
        hidden_states = torch.stack(outputs.hidden_states, dim=1)  # (B, 13, T, 768)
        norm_weights = F.softmax(self.audio_encoder.layer_weights, dim=-1)
        weighted_hidden = (hidden_states * norm_weights.view(1, -1, 1, 1)).sum(dim=1)
        projected = self.audio_encoder.projector(weighted_hidden)  # (B, T, 256)
        pooled = projected.mean(dim=1).squeeze(0)  # (256,)
        prosody_t = torch.from_numpy(prosody).to(self.device)
        return torch.cat([pooled, prosody_t], dim=-1)  # (262,)

    @torch.no_grad()
    def process_turn(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, Path, np.ndarray, Tuple[np.ndarray, int]]] = None,
        sr: int = 16000,
        speaker: str = "Speaker",
        history: Optional[Sequence[Dict[str, str]]] = None,
    ) -> TurnOutput:
        """
        One finished turn in, emotion state + reply out. Transcribes with Whisper if `text` is None;
        text only if `audio` is None. Without an explicit `history`, uses and extends the runtime's own.
        """
        t0 = time.perf_counter()

        if text is None:
            if audio is None:
                raise ValueError("process_turn requires at least one of `text` or `audio`.")
            text = self.transcribe(load_waveform_16k(audio, target_sr=sr), sr=sr)

        active_history = self.history if history is None else history
        text_vec = self.encode_text(text, speaker=speaker, history=active_history).unsqueeze(0)

        if audio is not None:
            audio_vec = self.encode_audio(audio, sr=sr).unsqueeze(0)
        else:
            audio_vec = torch.zeros((1, 262), dtype=text_vec.dtype, device=self.device)

        t_logits, a_logits, gate, _, _ = self.head.compute_branches(text_vec, audio_vec)
        logits = t_logits if audio is None else t_logits + gate * a_logits

        probs_t = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        text_probs_t = F.softmax(t_logits, dim=-1).squeeze(0).cpu().numpy()
        audio_probs_t = F.softmax(a_logits, dim=-1).squeeze(0).cpu().numpy()
        gate_mean = float(gate.mean().item())

        pred_idx = int(np.argmax(probs_t))
        emotion = IDX_TO_EMOTION[pred_idx]
        confidence = round(float(probs_t[pred_idx]), 4)
        prob_dict = {EMOTIONS[i]: round(float(probs_t[i]), 4) for i in range(len(EMOTIONS))}

        stance = resolve_stance(emotion, probs=prob_dict)
        reply = format_reply(utterance=text, emotion=emotion, stance=stance, speaker=speaker)

        if history is None:
            self.history.append({"speaker": speaker, "text": text, "emotion": emotion})

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        state = {
            "emotion": emotion,
            "confidence": confidence,
            "listener_stance": stance,
        }
        modality_info = {
            "text_top_emotion": IDX_TO_EMOTION[int(np.argmax(text_probs_t))],
            "audio_top_emotion": IDX_TO_EMOTION[int(np.argmax(audio_probs_t))],
            "mean_audio_gate": round(gate_mean, 4),
            "probabilities": prob_dict,
        }
        return TurnOutput(
            state=state,
            reply=reply,
            transcript=text,
            speaker=speaker,
            latency_ms=latency_ms,
            modality_info=modality_info,
        )

    def parameter_breakdown(self) -> Dict[str, Any]:
        """Count every learned parameter (frozen + trainable) on the local inference path."""
        self._ensure_asr_loaded()

        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        # Count the Wav2Vec2 backbone + layer_weights + projector used in encode_audio
        audio_params = (
            sum(p.numel() for p in self.audio_encoder.wav2vec2.parameters())
            + sum(p.numel() for p in self.audio_encoder.projector.parameters())
            + self.audio_encoder.layer_weights.numel()
        )
        head_params = sum(p.numel() for p in self.head.parameters())
        asr_params = sum(p.numel() for p in self.asr_model.parameters())

        offline_total = text_params + audio_params + head_params
        live_total = offline_total + asr_params
        return {
            "components": {
                f"text_encoder ({self.TEXT_MODEL_ID}, frozen)": text_params,
                f"audio_encoder ({self.AUDIO_MODEL_ID}, frozen)": audio_params,
                "fusion_head (GatedFusionHead, trained)": head_params,
                f"live_asr ({self.ASR_MODEL_ID}, frozen, live mic only)": asr_params,
            },
            "trainable_parameters": head_params,
            "offline_meld_path_total": offline_total,
            "live_mic_path_total": live_total,
            "parameter_cap": self.PARAMETER_CAP,
            "pct_of_cap": round(100.0 * live_total / self.PARAMETER_CAP, 1),
            "within_6b_cap": live_total <= self.PARAMETER_CAP,
        }


def run_streaming_session(
    runtime: MultimodalEmotionRuntime,
    audio_chunks: Optional[Iterable[np.ndarray]] = None,
    sr: int = 16000,
    pause_ms: int = 600,
    max_turns: int = 5,
    word_delay_s: float = 0.025,
) -> List[TurnOutput]:
    """
    Mic callback -> raw_audio_queue -> VAD thread -> turn_queue -> this thread.
    Capture and VAD keep running while a turn is processed and its reply prints, so speech
    that arrives meanwhile is queued, not dropped. One worker handles one turn at a time.
    """
    runtime._ensure_asr_loaded()
    vad = EnergyVAD(sr=sr, pause_ms=pause_ms)
    raw_audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
    turn_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
    stop_event = threading.Event()
    outputs: List[TurnOutput] = []

    def vad_loop() -> None:
        while not stop_event.is_set():
            try:
                chunk = raw_audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if chunk is None:
                flushed = vad.flush()
                if flushed is not None:
                    turn_queue.put(flushed)
                turn_queue.put(None)
                break
            for turn_wav in vad.push_chunk(chunk):
                turn_queue.put(turn_wav)

    vad_thread = threading.Thread(target=vad_loop, daemon=True)
    vad_thread.start()

    if audio_chunks is not None:
        # --simulate: push file chunks into the queue the mic callback would fill
        def feeder() -> None:
            for ch in audio_chunks:
                raw_audio_queue.put(ch)
                time.sleep(0.005)
            raw_audio_queue.put(None)

        threading.Thread(target=feeder, daemon=True).start()
    else:
        import sounddevice as sd

        blocksize = int(sr * 0.03)

        def audio_callback(indata, frames, time_info, status) -> None:
            raw_audio_queue.put(indata[:, 0].copy())

        stream = sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback,
        )
        stream.start()
        print(
            f"[mic] Listening at {sr} Hz (pause cutoff: {pause_ms} ms, Ctrl+C to stop)...",
            flush=True,
        )

    try:
        turn_idx = 0
        while turn_idx < max_turns:
            turn_wav = turn_queue.get()
            if turn_wav is None:
                break
            turn_idx += 1
            out = runtime.process_turn(text=None, audio=turn_wav, sr=sr)
            print(json.dumps(out.state), flush=True)
            stream_reply_words(out.reply, delay_s=word_delay_s)
            outputs.append(out)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if audio_chunks is None and "stream" in locals():
            stream.stop()
            stream.close()

    return outputs


def cmd_params() -> None:
    rt = MultimodalEmotionRuntime(load_asr=True)
    print(json.dumps(rt.parameter_breakdown(), indent=2))


def load_meld_turn(df: Any, split: str, index: int) -> Dict[str, Any]:
    """One MELD CSV row as process_turn arguments, with up to 3 earlier turns from the same dialogue."""
    row = df.iloc[index]
    dia_id = int(row["Dialogue_ID"])
    utt_id = int(row["Utterance_ID"])
    prior_rows = df[(df["Dialogue_ID"] == dia_id) & (df["Utterance_ID"] < utt_id)].sort_values("Utterance_ID")
    return {
        "text": str(row["Utterance"]),
        "speaker": str(row["Speaker"]),
        "history": [
            {"speaker": str(r["Speaker"]), "text": str(r["Utterance"])} for _, r in prior_rows.tail(3).iterrows()
        ],
        "audio": Path(f"data/meld/audio/{split}/dia{dia_id}_utt{utt_id}.flac"),
        "label": str(row["Emotion"]),
    }


# MELD test rows 946 and 1132: Ross says "No." (gold label neutral), Chandler says "No!" (gold label
# anger). Picked because audio flips the second one to the right answer; see README. Clips are in assets/.
DEMO_ROWS = (946, 1132)


def cmd_demo() -> None:
    import pandas as pd

    df = pd.read_csv("data/meld/test.csv")
    rt = MultimodalEmotionRuntime()
    for i in DEMO_ROWS:
        turn = load_meld_turn(df, "test", i)
        args = dict(text=turn["text"], speaker=turn["speaker"], history=turn["history"])
        text_only = rt.process_turn(**args)
        out = rt.process_turn(audio=Path("assets") / turn["audio"].name, **args)
        print(f"MELD test row {i}, gold label: {turn['label']}")
        for h in turn["history"]:
            print(f"  {h['speaker']}: {h['text']}")
        print(f"> {turn['speaker']}: {turn['text']}")
        print(f"text only:  {text_only.state['emotion']} ({text_only.state['confidence']:.2f})")
        print(f"with audio: {json.dumps(out.state)}")
        stream_reply_words(out.reply, delay_s=0.02)
        print()


def cmd_row(split: str, index: int) -> None:
    import pandas as pd

    turn = load_meld_turn(pd.read_csv(f"data/meld/{split}.csv"), split, index)
    rt = MultimodalEmotionRuntime(load_asr=False)
    out = rt.process_turn(
        text=turn["text"],
        audio=turn["audio"] if turn["audio"].exists() else None,
        speaker=turn["speaker"],
        history=turn["history"],
    )
    print(f"MELD label: {turn['label']}")
    print(json.dumps(out.state))
    stream_reply_words(out.reply, delay_s=0.02)


def cmd_bench(n: int = 50) -> None:
    """Time process_turn on n evenly spaced MELD test turns, with and without Whisper; report peak memory."""
    import pandas as pd
    import platform
    import resource
    import subprocess

    df = pd.read_csv("data/meld/test.csv")
    turns = [load_meld_turn(df, "test", i) for i in range(0, len(df), len(df) // n)][:n]
    turns = [t for t in turns if t["audio"].exists()]
    if not turns:
        raise SystemExit("MELD test audio not found. Run `python download_meld.py` first.")
    for t in turns:
        t["wav"] = load_waveform_16k(t["audio"])

    rt = MultimodalEmotionRuntime(load_asr=True)
    for t in turns[:2]:
        rt.process_turn(text=t["text"], audio=t["wav"], speaker=t["speaker"], history=t["history"])
        rt.process_turn(text=None, audio=t["wav"], speaker=t["speaker"], history=t["history"])

    with_text: List[float] = []
    with_asr: List[float] = []
    for t in turns:
        t0 = time.perf_counter()
        rt.process_turn(text=t["text"], audio=t["wav"], speaker=t["speaker"], history=t["history"])
        t1 = time.perf_counter()
        rt.process_turn(text=None, audio=t["wav"], speaker=t["speaker"], history=t["history"])
        t2 = time.perf_counter()
        with_text.append((t1 - t0) * 1000.0)
        with_asr.append((t2 - t1) * 1000.0)

    def stats(ms: List[float]) -> Dict[str, float]:
        return {k: round(float(np.percentile(ms, q)), 1) for k, q in (("p50", 50), ("p95", 95), ("max", 100))}

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    rss_mb = rss / 2**20 if platform.system() == "Darwin" else rss / 2**10
    chip = platform.processor()
    if platform.system() == "Darwin":
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    durations = [len(t["wav"]) / 16000 for t in turns]
    report = {
        "machine": f"{chip}, {platform.system()} {platform.release()}, torch {torch.__version__}, device {rt.device}",
        "turns": len(turns),
        "clip_seconds": {"mean": round(float(np.mean(durations)), 2), "max": round(float(np.max(durations)), 2)},
        "turn_ms_transcript_given": stats(with_text),
        "turn_ms_with_whisper": stats(with_asr),
        "peak_rss_mb": round(rss_mb),
        "mps_driver_mb": round(torch.mps.driver_allocated_memory() / 2**20) if rt.device.type == "mps" else None,
    }
    Path("artifacts").mkdir(exist_ok=True)
    Path("artifacts/bench.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def cmd_mic(simulate: bool = False, max_turns: int = 3) -> None:
    rt = MultimodalEmotionRuntime(load_asr=True)
    chunks = None
    if simulate:
        # The demo clips, each followed by 0.8 s of silence, cut into the mic's 30 ms blocks.
        silence = np.zeros(int(0.8 * 16000), dtype=np.float32)
        clips = [load_waveform_16k(p) for p in sorted(Path("assets").glob("*.flac"))]
        stream = np.concatenate([x for clip in clips for x in (clip, silence)])
        chunks = [stream[i : i + 480] for i in range(0, len(stream), 480)]
    run_streaming_session(rt, audio_chunks=chunks, max_turns=max_turns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Text + audio emotion runtime on MELD")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("demo", help="Two MELD test turns from assets/, text only vs. with audio (default)")
    sub.add_parser("params", help="Count the learned parameters on the local path")

    p_row = sub.add_parser("row", help="One MELD row with its dialogue history (needs download_meld.py)")
    p_row.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p_row.add_argument("--index", type=int, default=0)

    p_bench = sub.add_parser("bench", help="Latency and memory on MELD test turns (needs download_meld.py)")
    p_bench.add_argument("--n", type=int, default=50)

    p_mic = sub.add_parser("mic", help="Microphone loop: energy VAD, Whisper, then process_turn")
    p_mic.add_argument("--simulate", action="store_true", help="Feed the demo clips instead of the mic")
    p_mic.add_argument("--max-turns", type=int, default=3)

    args = parser.parse_args()
    if args.cmd == "params":
        cmd_params()
    elif args.cmd == "row":
        cmd_row(args.split, args.index)
    elif args.cmd == "bench":
        cmd_bench(args.n)
    elif args.cmd == "mic":
        cmd_mic(simulate=args.simulate, max_turns=args.max_turns)
    else:
        cmd_demo()


if __name__ == "__main__":
    main()
