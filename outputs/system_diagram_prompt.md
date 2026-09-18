# System-design drawing — prompts and sources

Three ways to get the Horizon Ladder system diagram as an image. Pick by what you need.

1. **`prompt-a`** — paste into GPT-5 / GPT-4o image generation (or Nano Banana, Midjourney, Ideogram)
   for a polished, presentation-style figure.
2. **`prompt-b`** — a shorter, punchier variant for a slide or a paper teaser.
3. **Graphviz / Mermaid source** — deterministic, correct text. Use this when the labels must be
   exactly right; image models mangle subscripts, hats and Greek letters almost every time.

> The authoritative version is the inline SVG in `outputs/math.html` — it is already vector, theme-aware,
> and its labels are guaranteed correct. The prompts below are for when you want an *illustrated* figure
> rather than a schematic.

---

## prompt-a — full system diagram

```
Create a clean, technical system-architecture diagram for a machine-learning
research protocol called "Horizon Ladder". Flat vector illustration style,
no 3D, no gradients, no drop shadows, no clip art, no people. Thin 1.5px
rounded-rectangle boxes with generous whitespace, a single accent colour
(deep blue) used sparingly, everything else in near-black on a very light
warm-grey background. Typography: a clean grotesque sans for box titles,
a monospace face for the small formula lines under each title. Left-to-right
reading order. 16:9 landscape.

The diagram has one main horizontal chain across the middle, one branch
below it, one feedback loop, and two labelled dashed enclosures.

MAIN CHAIN, left to right, five boxes joined by solid arrows:
1. "Causal window" — subtitle "past frames only, x[0:L]". Show a small
   filmstrip icon of 4 video frames, the last one highlighted.
2. "Encoder E (frozen)" — subtitle "Z_t, token sequence". Draw a small
   snowflake glyph in the corner to mean frozen.
3. "Predictor P (frozen)" — subtitle "masked causal query, horizon k".
   Mark this box with a small warning tag reading "out of distribution".
4. "Minimal decoder D" — subtitle "one linear layer per token".
   Mark it with a small tag reading "deliberately weak".
5. "Compare" — a wider box listing three short lines:
   "latent drift", "pixel error", "excess over floor".

BRANCH BELOW THE CHAIN:
A wide box labelled "Ground truth at the target frame", subtitle
"the future frame the rollout claims to predict". A solid arrow goes from
box 1 down into it, and another solid arrow goes from it up into box 5.
Draw a single real video frame thumbnail inside this box.

DASHED ENCLOSURE, lower left, labelled "ADMISSIBILITY — run before trusting
any pixel number". It contains three small boxes:
- "Decoder floor" — subtitle "reconstruction error on real frames"
- "Context leakage check" — subtitle "does the decoder read its input?"
- "Snapping check" — subtitle "smooth interpolation, no jumps"

FEEDBACK LOOP, lower right: a box labelled "Re-injection divergence",
subtitle "decode, re-encode, resume the rollout". A dashed arrow runs down
from box 4 into it, and a small curved dashed arrow loops from it back up
toward box 3 to show the compounding test.

TOP STRIP, spanning the full width, inside its own dashed enclosure
labelled "CANDIDATE LATENT-ONLY PROXIES — tested, all failed":
four small boxes joined by arrows — "perturb the context", "ensemble
spread", "7 scalar proxies", and a final box "can a proxy replace the
pixel check?" with a red cross mark and the caption "no".

Keep every label short and legible. Do not invent additional boxes,
equations, axes or logos.
```

## prompt-b — one-slide teaser

```
Minimal flat vector diagram, 16:9, light warm-grey background, one deep-blue
accent, thin rounded boxes, clean sans + monospace labels, lots of whitespace.

A left-to-right pipeline of four boxes: "past frames" -> "frozen encoder" ->
"frozen predictor (queried out of distribution)" -> "deliberately weak
decoder". The fourth box outputs into a "compare" box that sits beside a
second, lower box labelled "real future frame", with both feeding the same
comparison.

Above the pipeline, a faded strip of small boxes labelled "cheap latent-only
shortcuts" ending in a red cross and the word "no".

Below the pipeline, a dashed frame labelled "checks that must pass first"
containing three tiny boxes: "decoder floor", "leakage", "snapping".

One dashed feedback arrow from the decoder back into the predictor labelled
"re-injection".

No people, no 3D, no logos, no extra text.
```

**Expect to iterate.** Image models mis-render subscripts and hats. Two things that help:
ask for "short labels only, no mathematical notation" (as above), then add the equations yourself;
or generate the figure, then overlay the real formulas from `outputs/math.html`.

---

## Graphviz DOT — exact labels, no model needed

```bash
dot -Tsvg outputs/system_diagram.dot -o outputs/system_diagram.svg
```

```dot
digraph horizon_ladder {
  rankdir=LR; bgcolor="#eef1f4"; splines=ortho; nodesep=0.45; ranksep=0.55;
  node [shape=box, style="rounded,filled", fillcolor="#dde3e9", color="#ccd4dc",
        fontname="Helvetica", fontsize=11, margin="0.18,0.12"];
  edge [color="#848f9c", arrowsize=0.7, fontname="Courier", fontsize=9];

  subgraph cluster_proxy {
    label="S7 · candidate latent-only proxies — all failed";
    fontname="Courier"; fontsize=10; color="#ccd4dc"; style=dashed;
    perturb [label="perturb context\nZ + εσ(Z)ξ"];
    spread  [label="ensemble spread\nSVD → H, eᴴ, Var, d_M"];
    proxies [label="7 scalar proxies g(·)"];
    verdict [label="can g replace the pixel check?\nno", fillcolor="#eef1f4"];
    perturb -> spread -> proxies -> verdict;
  }

  s1 [label="S1 · causal window\nL = 2⌊(t+1)/2⌋"];
  enc [label="E · encode (frozen)\nZ_t ∈ ℝ^(N×D)"];
  s3 [label="S3 · causal query\nẐ_k = P(Z_t; M_x, M_y)"];
  s5 [label="S5 · decode\nx̂_k = D(Ẑ_k)"];
  s6 [label="S6 · compare\nδ_k = ‖Ẑ_k − Z*_k‖\nε_k = RMSE(x̂_k, x*_k)\nρ = R[ε]/R[δ]", fillcolor="#eef1f4"];
  gt [label="S2/S4 · ground truth\ns_k = ⌊(t+k)/τ⌋ · Z*_k = E(x at s_k)"];

  subgraph cluster_gate {
    label="admissibility — run before trusting any ε";
    fontname="Courier"; fontsize=10; color="#ccd4dc"; style=dashed;
    floor [label="S6b · floor\nε_floor = E[RMSE(D(E(x)), x)]"];
    leak  [label="S8a · leakage\nΓ = ‖D(z_A)−x_B‖ − ‖D(z_A)−x_A‖"];
    snap  [label="S8b · snapping\nκ = max |e_{i+1} − 2e_i + e_{i−1}|"];
  }

  s9 [label="S9 · re-injection\nd_j = ‖P(Ẑ, j) − P(E(D(Ẑ)), j)‖\nrate = slope of d_j", fillcolor="#eef1f4"];

  s1 -> enc -> s3 -> s5 -> s6;
  s1 -> gt; gt -> s6;
  s5 -> s9 [style=dashed];
  s9 -> s3 [style=dashed, constraint=false, label="compounding?"];
}
```

## Mermaid — for a README or a wiki

```mermaid
flowchart LR
  subgraph PROXY["S7 · latent-only proxies — all failed"]
    direction LR
    PB[perturb context] --> ES[ensemble spread] --> GS[7 scalar proxies] --> V["replace the pixel check? — no"]
  end

  S1["S1 · causal window<br/><code>L = 2⌊(t+1)/2⌋</code>"] --> E["E · encode (frozen)<br/><code>Z_t</code>"]
  E --> S3["S3 · causal query<br/><code>Ẑ_k = P(Z_t; M_x, M_y)</code>"]
  S3 --> S5["S5 · decode<br/><code>x̂_k = D(Ẑ_k)</code>"]
  S5 --> S6["S6 · compare<br/><code>δ_k, ε_k, ρ = R[ε]/R[δ]</code>"]
  S1 --> GT["S2/S4 · ground truth<br/><code>Z*_k = E(x at s_k)</code>"]
  GT --> S6

  subgraph GATE["admissibility — run first"]
    F["S6b · floor"]
    L["S8a · leakage"]
    SN["S8b · snapping"]
  end

  S5 -.-> S9["S9 · re-injection<br/><code>d_j</code>, slope"]
  S9 -.->|compounding?| S3
```
