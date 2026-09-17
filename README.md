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

## Contribution

1. **A protocol for a question prior methods can't ask.** Representation-
   inversion methods (Mahendran & Vedaldi, Nash et al., FAE, DPS, SPADE,
   Neuralangelo, GeoDE) invert the latent of an image that already exists —
   they're built to reconstruct as well as possible. Horizon Ladder inverts
   a *rolled-forward* latent, for a frame that was never observed, using a
   decoder deliberately kept too weak to hallucinate — so what comes out is
   evidence about the latent, not about the decoder's generative capacity.
   That trade (decode quality for decode honesty) is what lets this ask
   whether pixel-space error and representation-space drift move together
   at all, on a frozen, unmodified, real state predictor (V-JEPA2) with no
   fine-tuning and no native decoder to lean on.

2. **A measurement-gap finding, not just a pilot result.** Stage D (below)
   found `pixel_error` moves several times more than `latent_drift` under
   horizon. Experiments 1–6 (`outputs/experiment{1,3,4,5,6}_summary.json`)
   asked the obvious follow-up — can any cheap, latent-only signal
   substitute for the pixel-space check? — and stress-tested six candidate
   instability metrics (cosine distance, Mahalanobis distance, ensemble
   variance, effective rank, normalized L2 drift, and an entropy-rate
   metric ported from ecological crash detection, EBID) against real pixel
   divergence on real clips, alongside the raw latent-drift baseline
   already established in Stage D:
   - **Weak, not-significant correlation.** The best candidate (normalized
     L2) reaches Pearson r = 0.12 against `pixel_error`; raw latent drift
     itself reaches r = 0.013 (Experiment 1). Cluster-bootstrapped 95% CIs
     (resampled by clip, n=24, `outputs/stats_significance.json`) include
     zero for **all six candidates** — none is statistically distinguishable
     from no relationship at this sample size, let alone practically usable.
   - **No early warning.** Of 8 real crashes (excess-over-floor over
     threshold) across 24 clips, the entropy-rate metric gave lead-time
     warning on zero of them (Experiment 3); Wilson 95% CI on that rate is
     [0%, 32%] — even generously, better than roughly 1-in-3 is ruled out.
   - **Sign flips with an unrelated pipeline choice.** The entropy vs.
     perturbation-strength slope is negative and significant under a raw
     rollout (95% CI [-0.0075, -0.0039], r² up to 0.97) but its CI is fully
     disjoint from the positive-trending, not-quite-significant slope under
     a context-corrected rollout (95% CI [-0.0023, 0.0189]) — the metric's
     relationship to perturbation strength depends on a design choice
     orthogonal to the instability it's meant to measure (Experiment 4).
   - **No consistent ranking.** Across 96 (mode × perturbation × horizon)
     conditions and 4,026 decided pairwise comparisons, only 678 (17%)
     agree with a global Copeland ranking, and 10 explicit 3-cycles were
     found (Experiment 6) — a formal certificate of intransitivity, not a
     sampling artifact: no scalar ordering these conditions can be sorted
     by, full stop.

   Put together: for this frozen latent-only predictor, none of six
   plausible latent-space proxies — including one purpose-built for early
   crash warning in a different domain — can stand in for decoding against
   ground truth. That's the gap the protocol exists to expose, and it's
   why pixel-space checking (however minimal the decoder) isn't optional
   here.

3. **Per-clip evidence, not an averaged artifact.**
   `outputs/results_gallery.html` shows six real clips independently, each
   annotated with its own `latent_drift`/`pixel_error` horizon-range. The
   divergence isn't uniform across clips — some show pixel_error moving
   ~6x more than latent_drift across k=2→32, others show the two moving
   almost together — which is itself consistent with why no single
   latent-space threshold generalizes (Experiment 1): the clips a
   threshold would need to separate sit on both sides of it.

4. **A second, differently-architected predictor — and the gap changes
   shape, not just magnitude.** V-JEPA2's predictor is frozen, pretrained
   elsewhere, and queried causally out-of-distribution (see Status). To
   test whether the measurement gap is a property of *that regime* or of
   latent-only prediction generally, the identical protocol was run
   against a second family: a frozen, independently-pretrained DINOv2
   image encoder paired with a small predictor trained here, from scratch,
   by gradient descent on the same clips (`latent2rgb/dinov2_adapter.py`,
   `delta_predictor.py`, `scripts/train_second_family.py` —
   see Limitations for exactly what this is and isn't evidence for). The
   result is not "the gap disappears" — it's a different, informative
   shape:
   - `latent_drift` moves by 34% across k=2→32 (vs. V-JEPA2's ~2%); the
     separation-statistic residual — how much `pixel_error` exceeds what a
     low-horizon fit of `latent_drift` predicts — shrinks about 30x (0.0004
     vs. V-JEPA2's 0.0099). Here the direction inverts: latent space moves
     *more* than pixel space, not less (`stage_d_results_second_family.csv`).
   - Candidate-proxy correlations are higher (best r = 0.39, cosine
     distance, vs. 0.12) but at only 10 clips, none clear a 95% CI either
     (`outputs/experiment1_second_family_stats.json`).
   - Reading: a predictor trained end-to-end against the very tokens being
     measured naturally keeps latent and pixel error coupled. The gap this
     protocol exists to catch is sharpest exactly where deployment risk is
     highest — a frozen, pretrained-elsewhere model queried off-distribution
     — not a fixed property of "latent-only prediction" as a category. That
     refinement is itself only visible because a second architecture was
     run through the same protocol.

## Protocol (adopt this)

The concrete, reusable output of Experiments 1–6 isn't "these six metrics
failed" in isolation — it's a checklist for anyone evaluating a frozen,
latent-only state predictor with no native decoder, regardless of model
family. Skipping straight to step 3 without steps 1–2 is how a decoder
artifact gets reported as a predictor finding, or a plausible-looking
latent-space proxy gets trusted without ever being checked against ground
truth:

1. **Decoder-honesty gate first.** Attach a decoder trained only on real
   (non-rolled) tubelets — never on rollouts — and run the leakage and
   snapping checks (`latent2rgb/instruments.py`) before trusting anything
   downstream. If either fails, every pixel-space number below it is
   actually measuring the decoder, not the predictor.
2. **Don't substitute a latent-space proxy for step 3.** We tested six —
   cosine distance, Mahalanobis distance, ensemble variance, effective
   rank, normalized L2 drift, and an entropy-rate metric transplanted from
   ecological crash detection (EBID) — and all failed on this predictor
   (r ≤ 0.12 vs. real pixel divergence, zero crash lead-time, a sign flip
   across an unrelated pipeline choice, a non-transitive condition
   ranking). Treat any latent-only "confidence" signal as unverified on
   your model until it passes this same battery — Experiments 1, 3, 4, 6
   are the battery, not V-JEPA2-specific code.
3. **Decode and compare directly.** Once the decoder is honesty-gated,
   decode the rolled-forward latent and compare it to the real
   ground-truth frame. This is the only check in our results that
   reliably surfaces the divergence a latent-only proxy misses.
4. **Check re-injection compounding separately.** Decode → re-encode →
   re-inject and track whether divergence grows or shrinks with further
   horizon (Stage F) — this distinguishes one-shot decode noise from
   rollout drift that compounds, which steps 1–3 alone can't tell apart.

Nothing in this checklist is written against V-JEPA2 specifically — it's
implemented against the `Encoder`/`Decoder`/`StatePredictor` protocols in
`latent2rgb/interfaces.py`, so any model satisfying those interfaces runs
through the same four steps unmodified.

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
- **Experiments 1–6 (EBID measurement gap).** See Contribution, above, for
  the full breakdown: six candidate latent-only instability metrics
  (including an entropy-rate metric ported from ecological crash
  detection) all correlate weakly with real `pixel_error` (r ≤ 0.12, 95% CI
  includes zero for all six), give zero early-warning lead time on real
  crashes, flip sign under an unrelated pipeline choice, and produce a
  pairwise condition ranking that is mostly intransitive (17%
  Copeland-consistent, 10 explicit 3-cycles).
- **Second model family (DINOv2 + a predictor trained here).** See
  Contribution #4 and Limitations. The same protocol run against a
  differently-architected, differently-trained predictor does *not*
  reproduce the same gap shape: latent and pixel error stay more coupled
  (separation residual ~30x smaller), and the divergence direction
  inverts. Read as evidence the gap tracks *frozen, off-distribution
  querying*, not latent-only prediction as a category — not as evidence
  about DINOv2, or about video world models beyond V-JEPA2, generally.
- **Not run:** counterfactual faithfulness (rolling one state under two
  different actions). Needs V-JEPA2-AC plus a real action-conditioned
  (robot-trajectory) dataset; both are far outside this project's disk
  budget (the existing V-JEPA2 checkpoint alone is 4.8GB against ~18GB
  free, and the datasets this check is normally run against, e.g.
  DROID/Bridge, are 100GB+). Explicitly out of scope here — see
  Limitations — not attempted as a shortcut.

## Limitations

- **Counterfactual faithfulness is untested.** Every claim in this repo is
  about whether latent-space observables predict pixel-space divergence
  under a fixed, non-action-conditioned rollout. None of it says anything
  about whether the predictor responds *correctly* to different actions —
  that needs V-JEPA2-AC and a real action-conditioned dataset, and both
  are well outside what fits here (see Status). Treat this as a stated
  scope boundary, not an oversight: the faithfulness question is real and
  remains open.
- **The second model family is a lightweight stand-in, not a second SOTA
  world model.** DINOv2 was never pretrained for temporal dynamics; the
  temporal fusion (mean-pooling a tubelet's two frames) and the predictor
  (`delta_predictor.py`, a small MLP) are both built for this repo, trained
  on ~26 clips for 40 epochs — not independently pretrained, not
  state-of-the-art, not a research contribution in themselves. Its result
  (Contribution #4) answers "does the gap reproduce on a differently
  architected, differently trained latent-only predictor," not "does the
  gap hold for latent-only video world models in general." Confirming that
  broader claim still needs an independently-pretrained second SOTA model,
  which this project's disk/time budget didn't allow for.
- **Sample sizes are small.** 10–26 clips depending on the experiment (see
  `outputs/stats_significance.json` and `experiment1_second_family_stats.json`
  for the bootstrap confidence intervals behind every headline correlation
  and slope). Point estimates are reported alongside their CIs throughout
  Contribution and Status specifically so they aren't read as more certain
  than the sample size supports.

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
- `latent2rgb/ebid/` — the candidate latent-only instability metrics
  (cosine distance, Mahalanobis, ensemble variance, effective rank,
  entropy) stress-tested in Experiments 1–6; see Contribution, above
- `latent2rgb/dinov2_adapter.py` — the second model family's `Encoder`:
  frozen, independently-pretrained DINOv2 ViT-S/14, mean-pooled across a
  tubelet's 2 frames to land on the same token convention as
  `vjepa_adapter.py`; see Contribution #4 and Limitations
- `latent2rgb/delta_predictor.py` — the second family's `StatePredictor`:
  a small MLP trained here (DINOv2 ships no predictor at all)

## Scripts

| Script | Stage |
|---|---|
| `scripts/smoke_test.py` | B — real shapes end to end |
| `scripts/train_decoder.py` | C — decoder floor |
| `scripts/pilot.py` | D — the decision point |
| `scripts/stage_e_real.py` | E — decoder honesty checks, real model |
| `scripts/stage_f_reinjection.py` | F — re-injection divergence, real model |
| `scripts/experiment1_ebid_vs_baselines.py` | 1 — candidate metrics vs. pixel_error, real model |
| `scripts/experiment3_lead_time.py` | 3 — entropy-rate crash lead-time, real model |
| `scripts/experiment4_sweep.py` | 4 — horizon × perturbation × correction-mode sweep |
| `scripts/experiment5_interactions.py` | 5 — mode-gap and entropy-scaling fits over the sweep |
| `scripts/experiment6_transitivity.py` | 6 — pairwise condition ranking, Copeland/3-cycle test |
| `scripts/dump_visuals.py` | writes true/floor-recon/rollout PNGs for a gallery |
| `scripts/dump_rollout_videos.py` | writes true/rollout video pairs for the drag-compare gallery |
| `scripts/dump_rollout_metrics.py` | writes per-clip latent_drift/pixel_error for the gallery's video pairs |
| `scripts/live_dashboard_runner.py` | runs Stage D live across both datasets, feeds `outputs/dashboard.html` |
| `scripts/stats_significance.py` | cluster-bootstrap 95% CIs on Stage D / Experiments 1, 4 and Wilson CI on Experiment 3 |
| `scripts/train_second_family.py` | trains the DINOv2 predictor + decoder (second model family) |
| `scripts/pilot_second_family.py` | D equivalent, second family |
| `scripts/experiment1_second_family.py` | 1 equivalent, second family |

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

python scripts/experiment1_ebid_vs_baselines.py  # writes outputs/experiment1_{results.csv,summary.json}
python scripts/experiment3_lead_time.py          # writes outputs/experiment3_{results.csv,summary.json}
python scripts/experiment4_sweep.py              # writes outputs/experiment4_{results.csv,summary.json}
python scripts/experiment5_interactions.py       # writes outputs/experiment5_summary.json
python scripts/experiment6_transitivity.py       # writes outputs/experiment6_summary.json
python scripts/stats_significance.py             # writes outputs/stats_significance.json (bootstrap CIs)

python scripts/train_second_family.py       # writes predictor_dinov2.pt, decoder_dinov2.pt, floor_dinov2.json
python scripts/pilot_second_family.py       # writes stage_d_results_second_family.csv
python scripts/experiment1_second_family.py # writes outputs/experiment1_second_family_{results.csv,summary.json}

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
