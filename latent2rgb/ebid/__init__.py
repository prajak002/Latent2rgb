"""
EBID (Entropy-Based Instability Dynamics) instrumentation, adapted from
github.com/HussainAther/pcc onto Horizon Ladder's real latent-token
rollouts.

PCC's EBID measures instability on a system's state distribution via
Shannon entropy and its rate of change (dS/dt), then asks whether an
entropy-rate threshold crossing gives earlier warning of a "crash" (their
term: population extinction / fixation) than the raw state variables do
(scripts/lead_time.py in that repo).

Here the "system" is a rollout's latent token set [N, D] at a given
horizon step, the "state distribution" is the spectral distribution of
that token set (see entropy.py), and the "crash" is excess pixel_error
over floor (Horizon Ladder's existing headline signal). Everything in
this module is generic across model families -- it operates on plain
Tensors, not on VJEPA2Encoder/-Predictor specifically -- so the same code
runs against a second StatePredictor (e.g. an adapted StreamDiffVSR) with
no changes.
"""
