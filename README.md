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
   - **Weak, not-significant correlation — honestly bounded.** The best
     candidate (normalized L2) reaches Pearson r = 0.12 against
     `pixel_error`; raw latent drift itself reaches r = 0.013 (Experiment
     1). Cluster-bootstrapped 95% CIs (resampled by clip, n=24,
     `outputs/stats_significance.json`) include zero for all six
     candidates, but at n=24 that alone only has ~80% power to detect
     |r|≈0.55 — so "CI includes zero" doesn't by itself rule out a
     moderate relationship. The CI upper bounds do the honest bounding:
     they range from ~0.12 (Mahalanobis, the tightest) to ~0.50 (ensemble
     variance, the loosest) — read per-candidate, not as one number.
   - **A second, power-independent check confirms it: conformal
     efficiency.** Correlation-test power is a real limitation at n=24, so
     `scripts/conformal_calibration.py` asks a differently-shaped question
     that isn't power-starved the same way: does conditioning a prediction
     interval for `pixel_error` on the candidate metric make that interval
     *narrower* than a proxy-free baseline that ignores the metric and
     predicts the marginal mean? Using CV+ (Barber et al. 2021,
     leave-one-clip-out folds, not literal functional conformal prediction
     — that's for curve-valued data, this is scalar per (clip, k), citing
     it here would overstate the connection to Foresight, arXiv
     2606.23085) at nominal 90% coverage (empirical: baseline 90.8%,
     guaranteed floor 80%): five of six candidates produce intervals
     *wider* than the proxy-free baseline (+0.1% to +4.9%); the sixth
     (normalized L2) is 1.3% narrower — noise, not a usable signal. None
     of the six make the interval tighter. This doesn't depend on
     detecting a correlation at all, so it isn't subject to the same
     power caveat as the bullet above.
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

3. **Per-clip evidence, not an averaged artifact.** The rollout gallery
   (`outputs/dashboard.html#gallery-sec`) shows six real clips independently, each
   annotated with its own `latent_drift`/`pixel_error` horizon-range. The
   divergence isn't uniform across clips — some show pixel_error moving
   ~6x more than latent_drift across k=2→32, others show the two moving
   almost together — which is itself consistent with why no single
   latent-space threshold generalizes (Experiment 1): the clips a
   threshold would need to separate sit on both sides of it.

4. **Two more, differently-trained predictors — and they agree with each
   other, not with V-JEPA2.** V-JEPA2's predictor is frozen, pretrained
   elsewhere, and queried causally out-of-distribution (see Status). To
   test whether the measurement gap is a property of *that regime* or of
   latent-only prediction generally, the identical protocol was run
   against two more families, differing from V-JEPA2 (and from each
   other) in what's frozen vs. trained and in encoder architecture:
   - **Family B**: a frozen, independently-pretrained DINOv2 image
     encoder paired with a small MLP predictor trained here by gradient
     descent (`latent2rgb/dinov2_adapter.py`, `delta_predictor.py`,
     `scripts/train_second_family.py`).
   - **Family C**: an action-free variant of LeWorldModel (Maes, Le Lidec,
     Scieur, LeCun, Balestriero, arXiv 2603.19312) — encoder *and*
     predictor trained jointly, end-to-end, from raw pixels, with no
     pretrained representation at all (a ViT-Tiny encoder + a causal
     transformer predictor + a SIGReg anti-collapse regularizer,
     `latent2rgb/lewm_adapter.py`, `lewm_predictor.py`,
     `scripts/train_lewm_family.py`). See Limitations for exactly what
     "action-free" changes and what this is/isn't evidence for.

   Neither result is "the gap disappears" — both are a different,
   informative shape, and the two agree with each other on the part that
   matters most:
   - **Direction inverts on both, and it's bootstrap-confirmed on both.**
     `latent_drift` moves far more than `pixel_error` on both trained
     families (Family B: 34% vs. 10% range across k=2→32; Family C: 66%
     vs. 4%) — the opposite of V-JEPA2 (~2% vs. ~12%). The divergence-ratio
     95% CIs (clip-level bootstrap): **[2.0, 14.0]** on V-JEPA2, **[0.20,
     0.41]** on Family B, **[0.026, 0.131]** on Family C. All three
     intervals are disjoint from each other and from 1.0; 100% (Family C)
     and 99.97% (V-JEPA2) of resamples land on the predicted side. Two
     independently-built, differently-trained predictors landing on the
     *same* side against V-JEPA2's frozen-OOD result is a real pattern,
     not a coincidence of one run
     (`stage_d_results_second_family.csv`, `stage_d_results_lewm_family.csv`).
   - **Separation-statistic residuals don't move together, though.**
     Family B's residual is ~30x smaller than V-JEPA2's (0.0004 vs.
     0.0099) — pixel_error is almost fully explained by the low-k
     latent_drift trend. Family C's residual (0.0131) is close to
     V-JEPA2's, *despite* sharing Family B's "latent moves more"
     direction — so residual size and divergence direction aren't the
     same axis; a proxy that looked safe on Family B's residual wouldn't
     have been safe on Family C's.
   - **Candidate-proxy correlations stay weak on Family C, unlike Family
     B.** Best |r| = 0.20 (ensemble variance, negative sign) at n=9 clips
     — closer to V-JEPA2's weak correlations than to Family B's elevated
     (if still not significant) r=0.39.
     (`outputs/experiment1_lewm_family_summary.json`)
   - **Collapse checked, not assumed.** Family C's encoder is trained
     from scratch — SIGReg exists specifically to stop it collapsing to a
     constant output, and a collapsed encoder would trivially minimize
     prediction loss too, which would look identical to "good coupling"
     in the numbers above. Checked directly: per-dimension embedding std
     ≈ 1.0 across all 192 dims (range 0.83–1.20) on 50 held-out tubelets
     from 10 different clips — matching SIGReg's isotropic-Gaussian
     target, not a collapsed one.
   - Reading: direction of divergence tracks *how the predictor relates
     to its query* — frozen-and-queried-out-of-distribution (V-JEPA2)
     vs. trained-toward-what's-being-measured (Families B and C) — not a
     fixed property of any one architecture. That's a more specific,
     better-supported claim than "the gap is V-JEPA2-specific," and it's
     only visible because two, not one, differently-built second families
     were run through the same protocol.

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
- **Two more model families (DINOv2+trained-MLP, and an action-free
  LeWM-style JEPA trained end-to-end from scratch).** See Contribution #4
  and Limitations. Neither reproduces V-JEPA2's gap shape, and — more
  informatively — they agree with *each other*: on both, `latent_drift`
  moves far more than `pixel_error` (the opposite of V-JEPA2), confirmed
  by non-overlapping bootstrap CIs on the divergence ratio across all
  three families. Read as evidence the gap's *direction* tracks
  frozen/off-distribution querying vs. trained-toward-the-query, not
  latent-only prediction as a category, and not as evidence about DINOv2
  or LeWorldModel specifically, or about video world models beyond
  V-JEPA2, generally.
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
- **Both extra model families are lightweight stand-ins, not second SOTA
  world models.** DINOv2 (Family B) was never pretrained for temporal
  dynamics; its predictor (`delta_predictor.py`, a small MLP) is built for
  this repo, trained on ~26 clips for 40 epochs. LeWM-style (Family C) is
  an *action-free variant* of LeWorldModel, not a reimplementation —
  SSv2/Kinetics-mini carry no action labels, so the paper's AdaLN action
  pathway is simply absent; call it that, not "LeWM." It's also trained at
  a much smaller scale than the paper (batch size 16 vs. the paper's 128 —
  SIGReg's per-batch normality estimate is noisier at this scale, though
  the direct collapse check in Contribution #4 passed; 300 steps on ~26
  clips vs. 10 epochs on 10,000-episode datasets in the paper). Neither
  family is independently pretrained, state-of-the-art, or a research
  contribution in themselves. Their result (Contribution #4) answers "does
  the gap reproduce on differently architected, differently trained
  latent-only predictors," not "does the gap hold for latent-only video
  world models in general." Confirming that broader claim still needs an
  independently-pretrained SOTA model at real scale, which this project's
  disk/time budget didn't allow for.
- **Sample sizes are small, and "CI includes zero" is not by itself strong
  evidence at n=24.** 10–26 clips depending on the experiment (see
  `outputs/stats_significance.json` and `experiment1_second_family_stats.json`
  for the bootstrap CIs behind every headline correlation and slope). At
  n=24, a correlation test has ~80% power only for |r|≈0.55 or larger — a
  CI including zero doesn't rule out a moderate relationship the study is
  underpowered to detect. Two things mitigate this rather than paper over
  it: (1) CI *upper bounds* are reported per-candidate rather than
  collapsed into one blanket "not significant" claim (see Contribution
  #2), and (2) `scripts/conformal_calibration.py` adds a differently-shaped,
  non-power-dependent check (conformal interval efficiency vs. a
  proxy-free baseline) that corroborates the same conclusion through a
  mechanism unaffected by this specific power limitation.
- **Only Stage D and Experiment 1 were run on the two extra families, not
  the full battery.** Experiments 3 (lead-time), 4/5 (sign-stability), and
  6 (transitivity) only ran against V-JEPA2. The n=9–10 clips available
  per extra family are too few for Experiment 6's pairwise-ranking design
  in particular (V-JEPA2's version used 26 clips for 4,026 pairs). Whether
  the sign-flip and intransitivity findings also hold on Families B/C is
  untested, not confirmed.

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
- `latent2rgb/lewm_adapter.py` — the third family's `Encoder`: a ViT-Tiny
  trained from scratch (no pretraining at all), per an action-free variant
  of LeWorldModel (arXiv 2603.19312); `tokens_per_tubelet=1` (one global
  embedding per tubelet, not a patch grid) — a real architectural
  difference from the other two adapters, not a bug
- `latent2rgb/lewm_predictor.py` — the third family's `StatePredictor`: a
  causal transformer trained jointly with the encoder, plus the SIGReg
  anti-collapse regularizer (Eq. EP/SIGReg in the paper, implemented
  exactly against the paper's formula)

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
| `scripts/conformal_calibration.py` | CV+ conformal interval efficiency vs. a proxy-free baseline, Experiment 1 |
| `scripts/train_second_family.py` | trains the DINOv2 predictor + decoder (second model family) |
| `scripts/pilot_second_family.py` | D equivalent, second family |
| `scripts/experiment1_second_family.py` | 1 equivalent, second family |
| `scripts/train_lewm_family.py` | trains the action-free LeWM-style encoder+predictor+decoder jointly, from scratch (third model family) |
| `scripts/pilot_lewm_family.py` | D equivalent, third family |
| `scripts/experiment1_lewm_family.py` | 1 equivalent, third family |

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
python scripts/conformal_calibration.py           # writes outputs/conformal_calibration.json

python scripts/train_second_family.py       # writes predictor_dinov2.pt, decoder_dinov2.pt, floor_dinov2.json
python scripts/pilot_second_family.py       # writes stage_d_results_second_family.csv
python scripts/experiment1_second_family.py # writes outputs/experiment1_second_family_{results.csv,summary.json}

python scripts/train_lewm_family.py         # writes lewm_encoder{,_vit}.pt, lewm_predictor.pt, decoder_lewm.pt, floor_lewm.json
python scripts/pilot_lewm_family.py         # writes stage_d_results_lewm_family.csv
python scripts/experiment1_lewm_family.py   # writes outputs/experiment1_lewm_family_{results.csv,summary.json}

python scripts/live_dashboard_runner.py &
python -m http.server 8090 --directory outputs
# open http://localhost:8090/dashboard.html
```

## Website

`outputs/dashboard.html` is the whole thing on one page — research question,
proposed solution, dataset, methodology, the full mathematical formulation
(every pipeline step with the equation it evaluates, KaTeX-rendered, plus a
system-design diagram), live-polling Stage D/E/F results, the drag-compare
rollout gallery, the measurement-gap results, the mathematical interpretation,
the protocol, the two extra model families, and limitations. A sticky section
nav runs down all fourteen sections.

`outputs/math.html` and `outputs/results_gallery.html` are now redirect stubs
into that page's `#design` and `#gallery-sec` anchors (they used to be separate
pages). `outputs/gallery.html` / `outputs/site.html` are earlier snapshot pages
kept for reference. `outputs/frames/` are the static sample frames (true /
floor-reconstruction / rollout-decode) shown in the live feed;
`outputs/videos/` are the gallery's clip triples.

Section 6 opens with an illustrated overview of the protocol
(`outputs/system_diagram.webp`, 146 KB; the full-resolution original is
`outputs/system_diagram.png`), followed by the same pipeline as an inline SVG
schematic carrying the equation each step evaluates.
`outputs/system_diagram_prompt.md` holds the image-generation prompts that
produced it, plus Graphviz (`outputs/system_diagram.dot`) and Mermaid sources.
