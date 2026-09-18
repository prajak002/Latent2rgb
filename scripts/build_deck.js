const P = require('pptxgenjs');
const NOTES = require('./deck_script.js');
const pres = new P();
pres.layout = 'LAYOUT_WIDE';            // 13.333 x 7.5 in
pres.title = 'Horizon Ladder';
pres.author = 'Horizon Ladder';

const W = 13.333, H = 7.5, M = 0.75, CW = W - 2 * M;
const DARK = '0E1116', PANEL = '1A2029', LIGHT = 'EEF1F4', CARD = 'F7F9FB';
const ACC  = '16406F';
const TXL = 'E7EBF0', DIML = '93A0B0', TXD = '161B22', DIMD = '5B6472';
const BLUE = '4A94EE', GREEN = '6FC791', RED = 'E88B83', AMBER = 'D69A4C';
const SERIF = 'Cambria', SANS = 'Calibri', MONO = 'Courier New';

let n = 0;
function slide(bg) {
  n++;
  const s = pres.addSlide();
  s.background = { color: bg };
  return s;
}
function eyebrow(s, t, c) {
  s.addText(t.toUpperCase(), { x: M, y: 0.52, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 11, color: c, charSpacing: 2, align: 'left' });
}
function title(s, t, c, size) {
  s.addText(t, { x: M, y: 0.95, w: CW, h: 1.15, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: size || 34, bold: false, italic: true, color: c,
    align: 'left', valign: 'top', lineSpacingMultiple: 1.05 });
}
function foot(s, label, light) {
  const c = light ? '7C8794' : '8794A4';
  s.addText(label, { x: M, y: H - 0.72, w: 6, h: 0.3, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 10, color: c, align: 'left' });
  s.addText(String(n), { x: W - M - 2, y: H - 0.72, w: 2, h: 0.3, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 10, color: c, align: 'right' });
}
function card(s, x, y, w, h, fill, line) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill },
    line: { color: line, width: 0.75 }, rectRadius: 0.08 });
}

/* ---------------------------------------------------------------- 1 cover */
{
  const s = slide(DARK);
  s.addText('A DIAGNOSTIC PROTOCOL FOR LATENT-ONLY WORLD MODELS', { x: M, y: 0.85, w: CW, h: 0.3,
    isTextBox: true, margin: 0, fontFace: MONO, fontSize: 12, color: DIML, charSpacing: 2 });
  s.addText('Horizon Ladder', { x: M, y: 1.7, w: CW, h: 1.6, isTextBox: true, margin: 0,
    fontFace: SERIF, fontSize: 72, italic: true, color: TXL, valign: 'middle' });
  s.addText('Rolling a frozen world model forward produces a latent for a future that never happened. We built the instrument that checks it anyway, and found that nothing cheaper works.',
    { x: M, y: 3.5, w: 9.6, h: 1.3, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 18,
      color: DIML, lineSpacingMultiple: 1.35, valign: 'top' });
  s.addText('prajak002.github.io/Latent2rgb', { x: M, y: H - 1.1, w: 5.2, h: 0.3, isTextBox: true,
    margin: 0, fontFace: MONO, fontSize: 12, color: BLUE });
  s.addText('V-JEPA2  /  DINOv2  /  LeWM-style JEPA', { x: W - M - 5.4, y: H - 1.1, w: 5.4, h: 0.3,
    isTextBox: true, margin: 0, fontFace: MONO, fontSize: 12, color: '6B7684', align: 'right' });
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 2 setup */
{
  const s = slide(LIGHT);
  eyebrow(s, 'The setup', DIMD);
  title(s, 'A world model predicts the future in latent space', TXD);
  const items = [
    ['01', 'Encode', 'Real frames go in. Out comes a sequence of latent tokens: the model’s compressed idea of what is happening.'],
    ['02', 'Roll forward', 'Ask the frozen predictor what the latent looks like k frames from now. No pixels are involved at any point.'],
    ['03', 'Act on it', 'Planning, control and evaluation all read that rolled-forward latent as if it were true.'],
  ];
  const cw = (CW - 0.6) / 3;
  items.forEach(([num, h, d], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.35, cw, 2.35, CARD, 'CCD4DC');
    s.addText(num, { x: x + 0.32, y: 2.6, w: cw - 0.64, h: 0.26, isTextBox: true, margin: 0,
      fontFace: MONO, fontSize: 11, color: BLUE });
    s.addText(h, { x: x + 0.32, y: 2.92, w: cw - 0.64, h: 0.4, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 18, bold: true, color: TXD });
    s.addText(d, { x: x + 0.32, y: 3.42, w: cw - 0.64, h: 1.1, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 13, color: DIMD, lineSpacingMultiple: 1.3 });
  });
  s.addText('Step 03 is the one that matters commercially. Every downstream decision trusts a prediction that was never checked against an image.',
    { x: M, y: 5.1, w: 10.8, h: 0.8, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 15,
      color: DIMD, lineSpacingMultiple: 1.35 });
  foot(s, 'Setup', true);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 3 problem */
{
  const s = slide(DARK);
  eyebrow(s, 'The problem', DIML);
  s.addText('There is no ground truth for a future that never happened.',
    { x: M, y: 1.0, w: 11.2, h: 1.7, isTextBox: true, margin: 0, fontFace: SERIF, fontSize: 40,
      italic: true, color: TXL, lineSpacingMultiple: 1.06, valign: 'top' });
  const colw = (CW - 0.7) / 2;
  [['What you can measure', 'How far the predicted latent sits from the real one. A number in representation space.'],
   ['What you actually care about', 'Whether the scene it describes is right. A number in pixel space, which nobody computes.']]
   .forEach(([h, d], i) => {
     const x = M + i * (colw + 0.7);
     s.addText(h.toUpperCase(), { x, y: 3.35, w: colw, h: 0.3, isTextBox: true, margin: 0,
       fontFace: MONO, fontSize: 11, color: RED, charSpacing: 1.5 });
     s.addText(d, { x, y: 3.75, w: colw, h: 1.1, isTextBox: true, margin: 0, fontFace: SANS,
       fontSize: 16, color: DIML, lineSpacingMultiple: 1.35 });
   });
  s.addText([{ text: 'The question this project asks: ', options: { bold: true, color: TXL } },
             { text: 'do those two numbers even move together?', options: { color: DIML } }],
    { x: M, y: 5.3, w: 11.2, h: 0.5, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 18 });
  foot(s, 'The problem', false);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 4 prior work */
{
  const s = slide(LIGHT);
  eyebrow(s, 'Why this was open', DIMD);
  title(s, 'Representation inversion cannot ask it', TXD);
  const colw = (CW - 0.9) / 2;
  const cols = [
    ['What prior methods do',
     'Mahendran & Vedaldi, Nash et al., FAE, DPS, SPADE, Neuralangelo, GeoDE all invert the latent of an image that already exists, and all are built to reconstruct it as well as possible.',
     'Their success criterion is fidelity. A stronger decoder is always better.'],
    ['What we need instead',
     'Invert a rolled-forward latent, for a frame that was never observed, on a frozen model that ships no decoder at all.',
     'Here a stronger decoder is worse. It would paint in detail the latent never carried, and hide the very failure we are looking for.'],
  ];
  cols.forEach(([h, a, b], i) => {
    const x = M + i * (colw + 0.9);
    s.addText(h, { x, y: 2.5, w: colw, h: 0.45, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 20, bold: true, color: TXD });
    s.addText(a, { x, y: 3.05, w: colw, h: 1.5, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 14, color: DIMD, lineSpacingMultiple: 1.4 });
    s.addText(b, { x, y: 4.6, w: colw, h: 1.0, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 14, color: i === 1 ? TXD : DIMD, bold: i === 1, lineSpacingMultiple: 1.4 });
  });
  foot(s, 'Prior work', true);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 5 the move */
{
  const s = slide(ACC);
  s.addText('THE MOVE', { x: M, y: 1.35, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 12, color: '9EC4EE', charSpacing: 2 });
  s.addText('Trade decode quality for decode honesty.', { x: M, y: 1.95, w: 11.6, h: 1.9,
    isTextBox: true, margin: 0, fontFace: SERIF, fontSize: 46, italic: true, color: 'EAF2FC',
    lineSpacingMultiple: 1.06, valign: 'middle' });
  s.addText('One linear layer per token. Plain pixel loss. No perceptual, adversarial or diffusion objective. Patch j of the output is a function of token j and nothing else, so there is nowhere for invented detail to come from.',
    { x: M, y: 3.9, w: 10.6, h: 1.4, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 18,
      color: 'C5DAF3', lineSpacingMultiple: 1.4 });
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 6 system */
{
  const s = slide(LIGHT);
  eyebrow(s, 'The system', DIMD);
  s.addText('One pass of the protocol', { x: M, y: 0.92, w: CW, h: 0.6, isTextBox: true,
    margin: 0, fontFace: SERIF, fontSize: 28, italic: true, color: TXD });
  s.addImage({ path: 'outputs/system_diagram_deck.png', x: M, y: 1.65, w: CW, h: 4.05,
    altText: 'The Horizon Ladder pipeline: causal window, frozen encoder, frozen predictor queried out of distribution, minimal decoder, compare; a ground truth branch, an admissibility box of decoder floor, leakage and snapping checks, a re-injection loop, and a strip of failed latent-only proxies.' });
  s.addText('Every stage is written against three interfaces, not against V-JEPA2, so any encoder, decoder and state predictor satisfying them runs the same protocol unmodified.',
    { x: M, y: 5.85, w: 11.2, h: 0.6, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 13,
      color: DIMD, lineSpacingMultiple: 1.35 });
  foot(s, 'The system', true);
  s.addNotes(NOTES[n]);
}


/* ---------------------------------------------------------------- 7 gate */
{
  const s = slide(DARK);
  eyebrow(s, 'Before any result', DIML);
  title(s, 'Prove the instrument, or the finding is about your decoder', TXL, 30);
  const items = [
    ['Check 1', 'Context leakage', 'Decode clip A and compare against clip A and an unrelated clip B. A decoder ignoring its input scores the same on both.', 'Passes: gap +0.15'],
    ['Check 2', 'Snapping', 'Slide between two real latents. A decoder that snaps to one memorised frame shows a plateau then a jump, not smooth decay.', 'Passes: no curvature spike'],
    ['Check 3', 'The floor', 'Measure what the decoder costs on real frames alone, with no rollout. Every later number is read against that floor.', 'Measured, not assumed'],
  ];
  const cw = (CW - 0.6) / 3;
  items.forEach(([tag, h, d, r], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 2.3, cw, 2.75, PANEL, '2C3440');
    s.addText(tag, { x: x + 0.3, y: 2.52, w: cw - 0.6, h: 0.26, isTextBox: true, margin: 0,
      fontFace: MONO, fontSize: 11, color: DIML });
    s.addText(h, { x: x + 0.3, y: 2.84, w: cw - 0.6, h: 0.36, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 17, bold: true, color: TXL });
    s.addText(d, { x: x + 0.3, y: 3.28, w: cw - 0.6, h: 1.2, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 12.5, color: DIML, lineSpacingMultiple: 1.3 });
    s.addText(r, { x: x + 0.3, y: 4.56, w: cw - 0.6, h: 0.32, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 13, bold: true, color: GREEN });
  });
  s.addText('Skip this and a decoder artefact gets reported as a finding about the predictor. It is the most common way this kind of result goes wrong.',
    { x: M, y: 5.45, w: 11.2, h: 0.7, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 15,
      color: DIML, lineSpacingMultiple: 1.35 });
  foot(s, 'Honesty gate', false);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 8 finding 1 */
{
  const s = slide(LIGHT);
  eyebrow(s, 'Finding 1', DIMD);
  title(s, 'The two numbers do not move together', TXD);
  const rows = [
    ['Latent drift, k = 2 to 32', 'moves 2.0%', DIMD],
    ['Pixel error, same rollout', 'moves 12.0%', TXD],
    ['Ratio, pixel over latent', '6.10x', BLUE],
    ['95% CI, clip-level bootstrap', '[2.03, 14.00]', DIMD],
  ];
  rows.forEach(([a, b, c], i) => {
    const y = 2.45 + i * 0.72;
    s.addText(a, { x: M, y, w: 3.5, h: 0.4, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 14, color: DIMD, valign: 'middle' });
    s.addText(b, { x: M + 3.5, y, w: 1.9, h: 0.4, isTextBox: true, margin: 0, fontFace: MONO,
      fontSize: 17, bold: true, color: c, align: 'right', valign: 'middle' });
    s.addShape(pres.ShapeType.line, { x: M, y: y + 0.5, w: 5.4, h: 0,
      line: { color: 'CCD4DC', width: 0.75 } });
  });
  const rx = M + 6.3;
  s.addText('Latent drift is almost flat across horizon. Pixel error is not.',
    { x: rx, y: 2.45, w: CW - 6.3, h: 0.7, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 18, bold: true, color: TXD, lineSpacingMultiple: 1.3 });
  s.addText('So the pixel movement cannot be explained by the latent simply drifting further from the truth. A latent-space check would see nothing happening while the decoded scene degrades.',
    { x: rx, y: 3.3, w: CW - 6.3, h: 1.3, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 14, color: DIMD, lineSpacingMultiple: 1.4 });
  s.addText('Both coordinates are compared as scale-free relative ranges, so the ratio is invariant to rescaling either one.',
    { x: rx, y: 4.7, w: CW - 6.3, h: 0.9, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 12.5, color: DIMD, lineSpacingMultiple: 1.4 });
  foot(s, 'Finding 1', true);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 9 objection */
{
  const s = slide(ACC);
  s.addText('THE OBVIOUS OBJECTION', { x: M, y: 1.3, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 12, color: '9EC4EE', charSpacing: 2 });
  s.addText('"Fine, but decoding is expensive. Surely some cheap latent-only signal tracks this?"',
    { x: M, y: 1.95, w: 11.2, h: 2.1, isTextBox: true, margin: 0, fontFace: SERIF, fontSize: 40,
      italic: true, color: 'EAF2FC', lineSpacingMultiple: 1.08 });
  s.addText('We took that seriously and tested seven, including one purpose-built for early crash warning in a different field entirely.',
    { x: M, y: 4.4, w: 10.4, h: 0.9, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 18,
      color: 'C5DAF3', lineSpacingMultiple: 1.4 });
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 10 battery */
{
  const s = slide(LIGHT);
  eyebrow(s, 'Finding 2', DIMD);
  title(s, 'Seven proxies, four independent tests, zero survivors', TXD);
  const tests = [
    ['Correlation', 'Best proxy reaches r = 0.12 against real pixel error. Raw latent drift reaches 0.013.', 'All seven CIs include zero'],
    ['Conformal efficiency', 'Does conditioning a prediction interval on the proxy make it narrower than ignoring the proxy?', 'Five of seven are wider'],
    ['Early warning', 'Of 8 real crashes across 24 clips, how many did the entropy-rate detector flag in advance?', 'Zero. Wilson CI [0%, 32%]'],
    ['Ranking consistency', 'Across 96 conditions and 4,026 decided pairwise comparisons, is there a consistent ordering?', '10 explicit 3-cycles'],
  ];
  tests.forEach(([t, d, r], i) => {
    const y = 2.3 + i * 0.83;
    card(s, M, y, CW, 0.7, CARD, 'CCD4DC');
    s.addText(t, { x: M + 0.3, y: y + 0.06, w: 2.5, h: 0.58, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 15, bold: true, color: TXD, valign: 'middle' });
    s.addText(d, { x: M + 2.95, y: y + 0.06, w: 5.6, h: 0.58, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 12, color: DIMD, valign: 'middle', lineSpacingMultiple: 1.2 });
    s.addText(r, { x: M + 8.7, y: y + 0.06, w: CW - 9.0, h: 0.58, isTextBox: true, margin: 0,
      fontFace: MONO, fontSize: 12.5, bold: true, color: RED, align: 'right', valign: 'middle' });
  });
  s.addText('Each test fails differently, so they do not inherit each other’s weaknesses. The conformal check does not depend on detecting a correlation at all, which is what makes it immune to the small-sample objection.',
    { x: M, y: 5.75, w: 11.4, h: 0.7, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 13,
      color: DIMD, lineSpacingMultiple: 1.35 });
  foot(s, 'The proxy battery', true);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 11 certificate */
{
  const s = slide(DARK);
  eyebrow(s, 'The strongest result', DIML);
  title(s, 'A certificate, not a p-value', TXL);
  const colw = (CW - 0.9) / 2;
  s.addText('Say condition A beats condition B when A wins on more clips than it loses. If any single score could rank these conditions, that relation could never loop back on itself.',
    { x: M, y: 2.45, w: colw, h: 1.3, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 15,
      color: DIML, lineSpacingMultiple: 1.4 });
  card(s, M, 3.85, colw, 0.75, PANEL, '2C3440');
  s.addText('A  >  B  >  C  >  A', { x: M, y: 3.85, w: colw, h: 0.75, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 20, bold: true, color: AMBER, align: 'center', valign: 'middle' });
  s.addText('We found ten of those loops. One is already enough.',
    { x: M, y: 4.8, w: colw, h: 0.6, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 16,
      bold: true, color: TXL, lineSpacingMultiple: 1.35 });
  const rx = M + colw + 0.9;
  s.addText('Why this one is different', { x: rx, y: 2.45, w: colw, h: 0.4, isTextBox: true,
    margin: 0, fontFace: SANS, fontSize: 19, bold: true, color: TXL });
  s.addText('Every other result here is a statistical estimate, and estimates can be underpowered. This is a proof by counterexample: it rules out the existence of a scalar stability index over these conditions outright.',
    { x: rx, y: 2.95, w: colw, h: 1.5, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 15,
      color: DIML, lineSpacingMultiple: 1.4 });
  s.addText('No sample-size caveat attaches to a refutation. The loops are in the data.',
    { x: rx, y: 4.55, w: colw, h: 0.8, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 15,
      color: DIML, lineSpacingMultiple: 1.4 });
  foot(s, 'Intransitivity', false);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 12 families */
{
  const s = slide(LIGHT);
  eyebrow(s, 'Finding 3', DIMD);
  title(s, 'Three predictor families, three disjoint intervals', TXD);
  const hdr = ['Family', 'Regime', 'Latent', 'Pixel', 'Ratio', '95% CI'];
  const fam = [
    ['V-JEPA2', 'frozen, queried out of distribution', '2.0%', '12.0%', '6.10', '[2.03, 14.00]', BLUE],
    ['DINOv2 + trained MLP', 'frozen encoder, predictor trained here', '34.3%', '9.9%', '0.29', '[0.20, 0.42]', GREEN],
    ['LeWM-style, from scratch', 'encoder and predictor trained jointly', '66.0%', '4.1%', '0.06', '[0.026, 0.132]', AMBER],
  ];
  const colW = [3.1, 3.5, 1.3, 1.3, 1.1, 1.53];
  const rows = [hdr.map((h, i) => ({
    text: h,
    options: { bold: true, color: DIMD, fontSize: 12, align: i < 2 ? 'left' : 'right',
               fill: { color: LIGHT }, valign: 'middle' } }))];
  fam.forEach(r => {
    rows.push([
      { text: r[0], options: { color: r[6], bold: true, align: 'left' } },
      { text: r[1], options: { color: DIMD, align: 'left' } },
      { text: r[2], options: { color: TXD, align: 'right' } },
      { text: r[3], options: { color: TXD, align: 'right' } },
      { text: r[4], options: { color: r[6], bold: true, align: 'right' } },
      { text: r[5], options: { color: DIMD, align: 'right' } },
    ]);
  });
  s.addTable(rows, { x: M, y: 2.35, w: CW, colW, fontFace: SANS, fontSize: 13,
    color: TXD, border: { type: 'solid', color: 'CCD4DC', pt: 0.5 },
    rowH: 0.52, valign: 'middle', margin: 0.08 });
  s.addText('All three intervals are disjoint from each other and from 1.0. The direction of the gap tracks how the predictor relates to its query, frozen and asked something it was never trained on, versus trained toward exactly what is being measured. It is not a property of any one architecture.',
    { x: M, y: 4.75, w: 11.5, h: 1.1, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 14,
      color: DIMD, lineSpacingMultiple: 1.4 });
  foot(s, 'Generalisation', true);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 13 limits */
{
  const s = slide(DARK);
  eyebrow(s, 'What we do not claim', DIML);
  title(s, 'The boundaries, stated before anyone asks', TXL);
  const lim = [
    ['Counterfactual faithfulness is untested', 'Everything here is about a fixed, non action-conditioned rollout. Whether the predictor responds correctly to different actions needs V-JEPA2-AC and a robot-trajectory dataset.'],
    ['The two extra families are stand-ins', 'Lightweight, trained here, at small scale. They answer whether the gap reproduces across regimes, not whether it holds for video world models in general.'],
    ['Sample sizes are small', 'Ten to 26 clips depending on the experiment. At n = 24 a correlation test only has 80% power for r around 0.55, which is exactly why the conformal check exists.'],
  ];
  lim.forEach(([t, d], i) => {
    const y = 2.4 + i * 1.0;
    card(s, M, y, CW, 0.85, PANEL, '2C3440');
    s.addText(t, { x: M + 0.35, y: y + 0.06, w: 3.9, h: 0.73, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 14.5, bold: true, color: TXL, valign: 'middle',
      lineSpacingMultiple: 1.2 });
    s.addText(d, { x: M + 4.45, y: y + 0.06, w: CW - 4.8, h: 0.73, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 12.5, color: DIML, valign: 'middle', lineSpacingMultiple: 1.25 });
  });
  s.addText('Each of these is a scope boundary we chose and documented, not an oversight someone found later. The confidence intervals behind every headline number are published alongside the code.',
    { x: M, y: 5.6, w: 11.4, h: 0.7, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 14,
      color: DIML, lineSpacingMultiple: 1.35 });
  foot(s, 'Limitations', false);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 14 protocol */
{
  const s = slide(LIGHT);
  eyebrow(s, 'The deliverable', DIMD);
  title(s, 'A four-gate protocol anyone can adopt', TXD);
  const gates = [
    ['01', 'Gate the instrument', 'Train the decoder on real frames only, never on rollouts, and pass the leakage and snapping checks first.', 'Fails: every pixel number below measures your decoder'],
    ['02', 'Do not substitute a proxy', 'Run the same four-test battery on your own model before trusting any latent-only confidence signal.', 'Fails: the proxy is unverified, so do not skip gate 3'],
    ['03', 'Decode and compare', 'Decode the rolled-forward latent against the real frame. Report both coordinates, their ratio and the residual.', 'The only check that reliably surfaces the gap'],
    ['04', 'Check compounding', 'Decode, re-encode, re-inject, and track whether error grows or shrinks with further horizon.', 'Fails: one-shot noise and compounding drift stay confused'],
  ];
  const cw = (CW - 0.66) / 4;
  gates.forEach(([num, t, d, f], i) => {
    const x = M + i * (cw + 0.22);
    card(s, x, 2.35, cw, 3.05, CARD, 'CCD4DC');
    s.addText(num, { x: x + 0.24, y: 2.55, w: cw - 0.48, h: 0.25, isTextBox: true, margin: 0,
      fontFace: MONO, fontSize: 11, color: BLUE });
    s.addText(t, { x: x + 0.24, y: 2.85, w: cw - 0.48, h: 0.62, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 15, bold: true, color: TXD, lineSpacingMultiple: 1.15 });
    s.addText(d, { x: x + 0.24, y: 3.52, w: cw - 0.48, h: 1.15, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 11.5, color: DIMD, lineSpacingMultiple: 1.3 });
    s.addText(f, { x: x + 0.24, y: 4.72, w: cw - 0.48, h: 0.6, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 10.5, color: RED, lineSpacingMultiple: 1.25 });
  });
  s.addText('Implemented against protocol interfaces, not against V-JEPA2. Any model exposing an encoder, a decoder and a state predictor runs all four gates unmodified.',
    { x: M, y: 5.6, w: 11.4, h: 0.6, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 14,
      color: DIMD, lineSpacingMultiple: 1.35 });
  foot(s, 'The protocol', true);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 15 who */
{
  const s = slide(DARK);
  eyebrow(s, 'Why it matters', DIML);
  title(s, 'Who needs this, and when', TXL);
  const who = [
    ['Labs shipping latent world models', 'You already report latent-space distance in your evaluations. This says what that number does and does not tell you about the scene.'],
    ['Robotics and planning teams', 'Anything that plans against a rolled-forward latent inherits this gap. A confidence signal that has not passed the battery is unverified on your model.'],
    ['Reviewers and evaluation work', 'A reusable, model-agnostic checklist, with the confidence intervals and the code published alongside every headline number.'],
  ];
  const cw = (CW - 0.7) / 3;
  who.forEach(([t, d], i) => {
    const x = M + i * (cw + 0.35);
    s.addShape(pres.ShapeType.rect, { x, y: 2.4, w: cw, h: 0.035, fill: { color: BLUE },
      line: { color: BLUE, width: 0 } });
    s.addText(t, { x, y: 2.62, w: cw, h: 0.75, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 17, bold: true, color: TXL, lineSpacingMultiple: 1.2 });
    s.addText(d, { x, y: 3.45, w: cw, h: 1.5, isTextBox: true, margin: 0, fontFace: SANS,
      fontSize: 13.5, color: DIML, lineSpacingMultiple: 1.4 });
  });
  s.addText('The bet the field is making is that latent-space evaluation is enough. On a real frozen predictor, on real video, it is not, and no cheap substitute we could construct closed the gap.',
    { x: M, y: 5.35, w: 11.4, h: 0.85, isTextBox: true, margin: 0, fontFace: SANS, fontSize: 16,
      bold: true, color: TXL, lineSpacingMultiple: 1.35 });
  foot(s, 'Why it matters', false);
  s.addNotes(NOTES[n]);
}

/* ---------------------------------------------------------------- 16 close */
{
  const s = slide(DARK);
  s.addText('EVERYTHING IS PUBLIC', { x: M, y: 0.95, w: CW, h: 0.3, isTextBox: true, margin: 0,
    fontFace: MONO, fontSize: 12, color: DIML, charSpacing: 2 });
  s.addText('One page: the protocol, the maths, the live results and the per-clip evidence.',
    { x: M, y: 1.6, w: 11.2, h: 1.7, isTextBox: true, margin: 0, fontFace: SERIF, fontSize: 38,
      italic: true, color: TXL, lineSpacingMultiple: 1.06 });
  s.addText('prajak002.github.io/Latent2rgb', { x: M, y: 3.5, w: 8, h: 0.5, isTextBox: true,
    margin: 0, fontFace: MONO, fontSize: 20, bold: true, color: BLUE });
  const cards = [
    ['Sections 5 to 7', 'Every pipeline step with the equation it evaluates, and a plain-language gloss'],
    ['Section 9', 'Drag-compare gallery: ground truth against decoded rollout, six real clips'],
    ['Sections 10 to 13', 'The four-test battery, the interpretation, and all three model families'],
  ];
  const cw = (CW - 0.6) / 3;
  cards.forEach(([a, b], i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, 4.45, cw, 1.35, PANEL, '2C3440');
    s.addText(a, { x: x + 0.28, y: 4.65, w: cw - 0.56, h: 0.26, isTextBox: true, margin: 0,
      fontFace: MONO, fontSize: 11, color: DIML });
    s.addText(b, { x: x + 0.28, y: 4.95, w: cw - 0.56, h: 0.72, isTextBox: true, margin: 0,
      fontFace: SANS, fontSize: 13, color: TXL, lineSpacingMultiple: 1.3 });
  });
  s.addNotes(NOTES[n]);
}

pres.writeFile({ fileName: 'outputs/horizon-ladder.pptx' })
  .then(f => console.log('wrote', f, '-', n, 'slides'));
