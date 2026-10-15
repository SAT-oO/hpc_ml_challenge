#!/usr/bin/env python3
"""Download MELD audio (16 kHz FLAC, ~1.5 GB) from the ajyy/MELD_audio dataset into data/meld/audio/."""

import tarfile

from huggingface_hub import hf_hub_download

for split in ["dev", "test", "train"]:
    path = hf_hub_download(repo_id="ajyy/MELD_audio", filename=f"archive/{split}.tar.gz", repo_type="dataset")
    with tarfile.open(path, "r:gz") as tar:
        tar.extractall("data/meld/audio", filter="data")
    print(f"{split}: extracted to data/meld/audio/{split}")
