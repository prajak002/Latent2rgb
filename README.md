# Horizon Ladder

A diagnostic protocol for latent-only world models. Rolling a frozen state
predictor forward produces a latent for a future that never happened —
there's no real image to check it against, so no prior representation-
inversion method (Mahendran & Vedaldi, Nash et al., FAE, DPS, SPADE,
Neuralangelo, GeoDE) can ask whether that rolled-forward latent's pixel-space
error diverges from its representation-space drift, or whether a decode
error compounds once re-injected into the rollout.

Pixel decoding here is the **instrument**, not the deliverable: a
deliberately minimal decoder (single linear layer per token, plain pixel
loss, no perceptual/adversarial/diffusion loss) attached to a model that
ships no native decoder at all — [V-JEPA2](https://github.com/facebookresearch/vjepa2).
A stronger decoder would hallucinate detail the latent doesn't carry, hiding
exactly the failure this exists to surface.

Live results, methodology, and the full write-up: run the dashboard below,
or see `outputs/dashboard.html`.

## Status

Real results, on the real frozen V-JEPA2 ViT-L encoder/predictor, on two
disjoint datasets (Something-Something v2 and Kinetics-mini):

- **Stage D (pilot).** Under a causal masks_x/masks_y query the predictor was
  never trained for, `latent_drift` barely moves with horizon (2–3% from
  k=2→32) while `pixel_error` moves several times more, non-monotonically —
  on both datasets.
- **Stage E (decoder honesty).** Context-leakage and snapping checks both
  pass on real encoded latents — the decoder reads its input, it hasn't
  collapsed to a generic output.
- **Stage F (re-injection divergence).** Decode → re-encode → re-inject: the
  resulting error *shrinks* with further horizon rather than compounding, on
  both datasets — consistent with the predictor being largely insensitive to
  context under this out-of-distribution query.
- **Not run:** counterfactual faithfulness. The base V-JEPA2 predictor has no
  action input at all — this needs V-JEPA2-AC and a real action-conditioned
  (robot trajectory) dataset, not more compute on this setup.
- **Not attempted:** a second model family, which is what would turn "this is
  what V-JEPA2 does" into "this is what latent-only predictors do."

## Code map

Everything is written against three `Protocol` interfaces in
`latent2rgb/interfaces.py` (`Encoder`, `Decoder`, `StatePredictor`) plus a
`ClipSource` for data, so nothing downstream is tied to a specific model or
dataset:

- `latent2rgb/vjepa_adapter.py` — real V-JEPA2 ViT-L encoder/predictor
  adapter (causal masks_x/masks_y construction, built by reading the
  vendored source directly, not from summaries)
- `latent2rgb/video_source.py`, `ssv2.py` — generic video-directory
  `ClipSource` (webm/mp4)
- `latent2rgb/decoder.py` — the minimal per-token linear decoder
- `latent2rgb/data.py` — causal context/horizon rollout builder
- `latent2rgb/metrics.py` — latent_drift, pixel_error, floor, separation fit
- `latent2rgb/instruments.py` — leakage, snapping, counterfactual,
  re-injection instruments (generic; some run against the synthetic-era
  interface and need the tubelet adaptation shown in `scripts/stage_e_real.py`)

## Scripts

| Script | Stage |
|---|---|
| `scripts/smoke_test.py` | B — real shapes end to end |
| `scripts/train_decoder.py` | C — decoder floor |
| `scripts/pilot.py` | D — the decision point |
| `scripts/stage_e_real.py` | E — decoder honesty checks, real model |
| `scripts/stage_f_reinjection.py` | F — re-injection divergence, real model |
| `scripts/dump_visuals.py` | writes true/floor-recon/rollout PNGs for a gallery |
| `scripts/live_dashboard_runner.py` | runs Stage D live across both datasets, feeds `outputs/dashboard.html` |

## Reproducing

```
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install timm einops opencv-python-headless beartype pillow scikit-learn huggingface_hub

git clone --depth 1 https://github.com/facebookresearch/vjepa2.git vendor/vjepa2
curl -L -o checkpoints/vitl.pt https://dl.fbaipublicfiles.com/vjepa2/vitl.pt   # ~4.9GB
```

Stage the datasets under `data/ssv2/videos` (webm) and `data/kinetics_mini`
(mp4, e.g. the `nateraw/kinetics-mini` HuggingFace dataset) — not included in
this repo (`data/`, `checkpoints/`, and `vendor/` are gitignored: license
restrictions on SSv2, and the checkpoint/vendored repo are large and easily
re-fetched).

```
python scripts/smoke_test.py --checkpoint checkpoints/vitl.pt --video-dir data/ssv2/videos
python scripts/train_decoder.py   # writes floor.json, decoder.pt
python scripts/pilot.py           # writes stage_d_results.csv
python scripts/stage_e_real.py    # writes outputs/stage_e_results.json
python scripts/stage_f_reinjection.py  # writes outputs/stage_f_results.json

python scripts/live_dashboard_runner.py &
python -m http.server 8090 --directory outputs
# open http://localhost:8090/dashboard.html
```

## Website

`outputs/dashboard.html` — the live dashboard (research question, proposed
solution, dataset, methodology, mathematical background, and live-polling
results). `outputs/gallery.html` / `outputs/site.html` are earlier snapshot
pages kept for reference. `outputs/frames/` are the static sample frames
(true / floor-reconstruction / rollout-decode) shown in the gallery.
