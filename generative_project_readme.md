# Text-to-Image Diffusion on Flickr30k (Milestones 1–3)

A compact, reproducible project that fine-tunes a text→image workflow on a 5k-pair subset of Flickr30k, runs controlled generations with Stable Diffusion, and evaluates results with FID, Inception Score, and CLIP text–image alignment. It also includes a small interactive UI for prompt-based generation and a flow diagram of the system.

---

## Overview
We build a minimal pipeline that starts from captions, encodes text with CLIP, conditions a Stable Diffusion model (v1-5) with classifier-free guidance, and outputs images. We log every run and evaluate outputs with standard metrics. The goal is clarity and reproducibility rather than state-of-the-art scores.

- **Dataset:** Flickr30k (Kaggle mirror), 5k image–caption pairs
- **Text encoder:** CLIP (ViT-B/32)
- **Generator:** Stable Diffusion v1-5 (Diffusers) with DDIM / DPM-Solver schedulers
- **Controls:** steps, guidance scale (CFG), seed, size, optional negative prompt
- **Metrics:** FID (Inception-V3 features), Inception Score, CLIP cosine alignment
- **Artifacts:** plots, CSV logs, and comparison grids for different parameters

---

## Repository Structure (suggested)
```
.
├─ notebooks/
│  ├─ Milestone_1.ipynb                # data prep + text→embedding baseline
│  ├─ Milestone_2.ipynb                # baseline conditional generation
│  ├─ Milestone_2_image_comparison.ipynb
│  ├─ Milestone_3.ipynb                # FID / IS / CLIP evaluation
│  └─ UI_widgets.ipynb                 # small interactive UI in notebook
├─ app/
│  └─ app_server.py                    # optional Gradio app (lab/cluster only)
├─ data/
│  ├─ raw/                             # Kaggle downloads
│  └─ processed/
│     ├─ images/                       # copied 5k JPEGs used for experiments
│     └─ pairs.csv                     # image_path, caption
├─ outputs/
│  ├─ m2/                              # milestone 2 images + model choice json
│  ├─ m2_full/                         # full sweep images (steps / CFG / sched)
│  └─ m3/                              # plots for FID / IS / CLIP
├─ metrics/
│  ├─ m2_runs.csv                      # single/batch generation logs
│  ├─ m2_full_runs.csv                 # full sweep logs
│  ├─ m3_scores.csv                    # FID/IS by steps
│  ├─ m3_clip_scores.csv               # CLIP by steps
│  ├─ m3_all_scores.csv                # joined table (steps)
│  ├─ m3_cfg_all_scores.csv            # joined table (CFG)
│  └─ m3_sched_all_scores.csv          # joined table (scheduler)
├─ figures/
│  └─ flowchart.png                    # prompt→diffusion→image→evaluation
└─ README.md
```

> Paths above are suggestions. If you already have outputs under a different root, keep them; just update notebook variables accordingly.

---

## Environment
- **Python** ≥ 3.10 (tested on 3.12)
- **CPU or GPU:** works on CPU; GPU speeds things up

Install the core stack:
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install "torch" "torchvision"  \
            diffusers==0.30.0 transformers==4.45.0 accelerate  \
            pandas matplotlib pillow scikit-image  \
            kagglehub==0.3.13  # for dataset download
```
Optional (for the app):
```bash
pip install gradio==4.44.0
```
Set cache directories if desired:
```bash
export HF_HOME="$PWD/.cache/huggingface"
```

---

## Data: Flickr30k (5k subset)
Download via KaggleHub and export a balanced subset of pairs:
```python
import kagglehub, pandas as pd, shutil, os
from pathlib import Path

# 1) Download
kpath = kagglehub.dataset_download("adityajn105/flickr30k")
raw_dir = Path("data/raw/flickr30k")
raw_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(kpath, raw_dir, dirs_exist_ok=True)

# 2) Parse captions.txt -> pairs.csv (image_path, caption)
#    and copy a 5k-image subset into data/processed/images
```
The notebooks handle parsing, de-duplication, simple cleaning, and export to `data/processed/pairs.csv` and `data/processed/images/`.

**Citation:** Refer to the Flickr30k paper and Kaggle mirror terms. Respect the dataset license.

---

## Milestone 1 (setup + baseline)
Open `notebooks/Milestone_1.ipynb` and run cells:
1. Verify environment and caches.
2. Download + prepare 5k pairs.
3. Build a text→embedding baseline with CLIP (ViT-B/32).
4. Save embeddings and a small prompt list for later evaluation.

Artifacts: `pairs.csv`, `embeddings/*.pt` (if used), and a short proposal.

---

## Milestone 2 (conditional generation)
Use `notebooks/Milestone_2.ipynb` and `Milestone_2_image_comparison.ipynb`:
1. Load `runwayml/stable-diffusion-v1-5` in Diffusers.
2. Generate images for a fixed prompt set using DDIM or DPM-Solver schedulers.
3. Sweep **steps** and **guidance scale** (CFG). Optionally add a **negative prompt**.
4. Log each run to CSV (file path, prompt id, settings, seed, time).
5. Save comparison panels per prompt and parameter setting.

Artifacts live under `outputs/m2*` and `metrics/m2*.csv`.

**Tip:** When passing `prompt_embeds` directly, make sure positive and negative embeddings have the same length; the notebook includes a helper that pads/truncates negative tokens to match the positive sequence length.

---

## Milestone 3 (evaluation)
Run `notebooks/Milestone_3.ipynb`:
1. Compute Inception-V3 features on a real-image subset (cached as FID reference).
2. For generated images, compute FID (vs. reference), Inception Score, and CLIP alignment.
3. Aggregate by **steps**, **CFG**, and **scheduler**; export tables and plots.

Outputs include:
- `metrics/m3_all_scores.csv`, `m3_cfg_all_scores.csv`, `m3_sched_all_scores.csv`
- `outputs/m3/*.png` and `outputs/m3_cfg/*.png` (metric curves)

---

## Interactive UI (optional)
- **Notebook UI:** `notebooks/UI_widgets.ipynb` exposes sliders for scheduler, steps, CFG, seed, and size, and displays images inline.
- **Gradio app:** `app/app_server.py` can run a small web UI. On lab/JupyterHub systems where `localhost` is blocked, prefer the notebook UI or start Gradio with `share=True`.

---

## Flow Diagram
The figure below visualizes how inputs, encoders, scheduler, UNet, and VAE connect, and how evaluation consumes generated images. Include it in your report.

![Pipeline overview](figures/flowchart.png)

---

## Key Results (example)
- Increasing **steps** improved Inception Score but not FID on our CPU runs, while CLIP alignment stayed stable.
- Moderate **CFG** produced a good balance between alignment and diversity; higher CFG slightly increased CLIP alignment with small IS changes.
- **DDIM vs. DPM-Solver** showed similar CLIP alignment; DDIM gave a small IS edge in our settings.

Use the exported plots/tables for exact values.

---

## Limitations
- CPU-only runs are slow and influence sample size and metric stability.
- FID is computed against a subset of real images; results are indicative, not definitive.
- Prompts are short and generic; richer prompts and more samples would strengthen conclusions.

---

## Ethics
We include a safety checker option and a negative-prompt list to reduce harmful content, but automated filters are imperfect. Use datasets and models responsibly, respect licenses and consent, avoid generating misleading or sensitive images, and document model limits when sharing outputs.

---

## Reproducibility
- Every generation call logs **prompt id**, **scheduler**, **steps**, **CFG**, **seed**, **size**, and output path to CSV under `metrics/`.
- Seeds are set per prompt to make side-by-side comparisons fair.
- Cached references (`outputs/m3/cache/*.npz`) make FID runs repeatable.

---

## Acknowledgements
- Stable Diffusion v1-5 (runwayml / Stability AI ecosystem)
- Hugging Face Diffusers and Transformers
- Flickr30k authors and Kaggle mirror maintainers

> Also respect the Stable Diffusion license and the dataset license when sharing models, images, or apps.

