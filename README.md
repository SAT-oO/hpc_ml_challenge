# MELD text + audio emotion runtime

Track: **text and audio**. One turn goes in (words, and the audio if we have it). Two things come out: a MELD emotion tag, and a short reply that quotes what was said.

I treated this as a listener for a character robot. The useful question is not "how sad is this line on a leaderboard," it's "did they just agree, push back, or go quiet?" So the emotion head is the model; the reply is a small stance table on top of it. No generative model in the loop.

```
{"emotion": "anger", "confidence": 0.81, "listener_stance": "pushback"}
Chandler said "No!" (emotion: anger). The tone is pushback. Ask what the objection is before going on.
```



## how to run the pipeline 

Python 3.10+, then:

```bash
pip install -r requirements.txt
python test_runtime.py
python meld_runtime.py          # same as: python meld_runtime.py demo
```

First run pulls three Hugging Face checkpoints (`distilroberta-base`, `superb/wav2vec2-base-superb-er`, and `openai/whisper-tiny.en` if you use the mic). Needs a network once; after that it's local.

`demo` is the handoff path. It takes two real MELD test turns (clips in `assets/`):

- Ross, "No." — gold label `neutral`
- Chandler, "No!" — gold label `anger`

For each turn it prints the dialogue context, the text-only guess, the fused guess (text + audio), then streams the reply. Same function (`process_turn`) as the mic loop. I picked this pair because the wording is almost the same and the labels are not — you can see whether the clip moves the call.

Other commands:

```bash
python meld_runtime.py mic                  # laptop mic, energy VAD, Whisper, then process_turn
python meld_runtime.py mic --simulate       # same loop, fed the demo clips
python meld_runtime.py row --index 1132     # any MELD row (needs the full audio download)
python meld_runtime.py bench                # latency / memory, writes artifacts/bench.json
python meld_runtime.py params               # exact numel dump if you want it
```

Full MELD audio (~1.5 GB, 16 kHz flac) is not in git:

```bash
python download_meld.py    # needed for train / row / bench, not for demo
python train_and_eval.py   # rebuilds the head + artifacts/metrics.json
```



## "real time"

This is turn-taking, not frame-by-frame emotion. A person finishes a sentence, there is a short gap, then the robot should have a tag and a line ready before the next beat.

Assumptions I actually coded to:

- A turn ends after **~600 ms of silence** (energy VAD, 30 ms frames). Cheap and wrong on loud rooms; good enough for a quiet laptop mic.
- After the turn closes, inference should land in **a couple hundred milliseconds** on a consumer laptop. I am not claiming a 50 ms hard deadline. Humans leave 200–600 ms between turns anyway; burning that whole budget on the VAD and then another 300 ms on the model is still usable. Waiting a full second after they stop is not.
- The mic callback keeps stuffing a queue while we encode the last turn, so we don't drop the next utterance. One worker, one turn at a time. Replies print word-by-word with a tiny sleep so they don't dump as a brick. That's presentation, not a language-model stream.

Measured on an Apple M1 Pro (PyTorch 2.14, MPS), 50 MELD test clips, numbers in `artifacts/bench.json`:


| path                     | p50     | p95     | worst in that run |
| ------------------------ | ------- | ------- | ----------------- |
| transcript already there | ~90 ms  | ~150 ms | ~410 ms           |
| + whisper-tiny.en        | ~175 ms | ~310 ms | ~350 ms           |


Peak process RSS was about **900 MB**. MPS also held ~2 GB of driver memory, which is the usual PyTorch tax, not the weight files.

So: 600 ms to decide they stopped, then roughly 0.1–0.3 s to answer. That's the budget.

## Pipeline

```
mic / wav
  → energy VAD (live path only)
  → whisper-tiny.en if we don't already have a transcript
  → DistilRoBERTa on this turn + last 3 turns
  → wav2vec2-er embedding + 6 cheap prosody numbers (pitch, energy, brightness)
  → small gated head → 7 MELD emotions
  → stance table → one-line reply
```

Text encoding: mean-pool the current utterance, mean-pool (history + current), then `0.75 * current + 0.25 * context`. Straight concat of the last three turns buries short replies. A lot of MELD is "No." / "Okay." / "What?".

Audio encoding: the SUPERB emotion wav2vec2 (layer-weighted, 256-d) plus log-energy, F0, crest, and high-frequency ratio. I am not doing pitch tracking because I enjoy it. Sitcom audio is messy; the extra six numbers are a cheap hedge when the SSL pooling gets confused by a laugh track.

Fusion: each side gets its own LayerNorm + MLP to 7 logits. A sigmoid gate (also from both vectors) scales the audio logits, then they add. Text-only / audio-only in the ablation table is just using one branch.

Stance: joy → with_you; anger/disgust → pushback; sadness/fear → caution; neutral → unclear. Surprise looks at whether joy or the negative emotions have more mass. Affirmative wording plus pushback gets a specific line ("the words say yes, but the tone is pushback") because that's the case a robot will actually mishandle.

## Parameter budget (6B cap)

Everything below is **total** loaded weights, frozen included. I did not count "active" or "trainable only."


| piece                            | source                                        | approx     |
| -------------------------------- | --------------------------------------------- | ---------- |
| text encoder                     | `distilroberta-base` (HF, frozen)             | ~82 M      |
| audio encoder                    | `superb/wav2vec2-base-superb-er` (HF, frozen) | ~95 M      |
| fusion head                      | trained here                                  | ~0.2 M     |
| ASR (mic only)                   | `openai/whisper-tiny.en` (HF, frozen)         | ~39 M      |
| **live path**                    |                                               | **~215 M** |
| offline / MELD path (no Whisper) |                                               | **~177 M** |


Call it well under 4% of the cap. `python meld_runtime.py params` prints the exact `numel()` if someone wants to audit it. I am not putting that number in this file.

## What I trained vs what I borrowed

Borrowed, frozen: the three HF checkpoints above. I looked at `j-hartmann/emotion-english-distilroberta-base` and did not use it — its card says it trained on MELD, so the test numbers would be junk.

Trained on official MELD train (9989) / picked on dev (1109) / scored once on test (2610):

- the gated head (`checkpoints/fusion_head.pt`)
- a linear concat baseline, for the table
- a tiny REINFORCE audio-weight policy, which I then left out of the runtime (see below)

Audio comes from `ajyy/MELD_audio` (16 kHz flac). Labels and transcripts are the official MELD CSVs in `data/meld/`.

## Test numbers

Full tables: `artifacts/metrics.json`. Rounded, on the official test split:


| setup                                     | weighted F1 | macro F1 | acc     |
| ----------------------------------------- | ----------- | -------- | ------- |
| always guess neutral                      | 0.31        | 0.09     | 48%     |
| audio branch only                         | 0.42        | 0.19     | 50%     |
| linear head, dump history into the string | 0.48        | 0.26     | 54%     |
| linear head, 0.75/0.25 current vs context | 0.58        | 0.35     | 63%     |
| gated head, text + audio + history        | **0.62**    | **0.42** | **64%** |
| same head, text branch only               | 0.60        | 0.41     | 63%     |
| gated, no dialogue history                | 0.62        | 0.43     | 64%     |


Three seeds of a fused head vs a head that never saw audio: fused ~0.618 ± 0.001 weighted F1, text-only head ~0.607 ± 0.004. So the lift is small but it isn't a one-seed accident.

Where audio actually moves the needle: it changes the predicted class on 250 / 2610 test turns. 100 of those go from wrong to right, 75 the other way. Net +25. On the 919 turns with ≤4 words, fused and text-only are a wash (both ~0.69). Fear and disgust stay bad — there are 50 and 68 test examples. I did not pretend otherwise.

I also tried a 7→16→1 REINFORCE policy that scales the audio residual per turn. Five seeds, scored against a constant audio weight of 1.0. Delta on test was noise (macro F1 +0.0005 ± 0.0008). So it is not in `process_turn`. The class is still in `meld_runtime.py` because `train_and_eval.py` uses it; that is leftover, not a product feature.

## What I finished / what I left out

Done:

- text + audio on MELD, local inference, under the 6B cap
- structured `{emotion, confidence, listener_stance}` plus a grounded one-liner
- streaming mic path (VAD → Whisper → same `process_turn`)
- a demo that shows one turn producing both outputs
- a small trained head and the obvious ablations

Left out on purpose:

- **Vision.** MELD cuts away from the speaker a lot. I had two days. A bad face crop would have made a worse system, not a more complete one.
- **A decoder LLM for the reply.** Even a 1B model eats a chunk of the budget and a few hundred ms of TTFT, and it will invent meeting advice. A table that quotes the utterance cannot do that.
- **Finetuning the 177 M encoders.** 10k sitcom lines is enough to memorize Friends voices and the laugh track. The 200k-parameter head trains in seconds and I can still swap the backbones.
- **The RL policy in production**, as above. No measured value, so it does not run.



## Limitations 

Audio-only is weak, and that is MELD, not just me. Laugh track, room mics, overlapping speech. The fusion head mostly trusts text and lets audio veto sometimes. That is the right bias and it also means a lot of turns look like a text classifier.

The energy VAD will false-trigger next to a TV. I would replace it with Silero if this left my laptop.

Replies are templates. They are grounded (they quote the line and name the predicted emotion) and they are not clever.

`test_runtime.py` pins the two demo rows to the checked-in head. Retrain and those asserts may fail; pick a new pair if that happens.

## Layout

```
meld_runtime.py       # VAD, encoders, head, CLI
train_and_eval.py     # feature cache, baselines, head, the RL experiment
test_runtime.py
download_meld.py
checkpoints/fusion_head.pt
artifacts/metrics.json
artifacts/bench.json
assets/dia107_utt11.flac    # Ross, "No."
assets/dia124_utt7.flac     # Chandler, "No!"
data/meld/{train,dev,test}.csv
```

