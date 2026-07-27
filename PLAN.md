# nanoPELICAN → HGQ2 rewrite: plan, status, and known shortcomings

Goal: rewrite the nanoPELICAN QAT training stack (sibling repo `PELICAN-nano/`,
PyTorch + Brevitas) in Keras 3 with **HGQ2** (high-granularity quantization,
gradient-learned bitwidths + EBOP resource regularization), targeting the
hand-written Vitis HLS firmware in sibling repo `nPELICAN-fpga/`.

Decision context (2026-07-27): HGQ2 requires the model be built from its own
Keras layers — it cannot wrap the existing torch modules, so "run HGQ2 on the
Keras torch backend" is the same rewrite with a slower backend (eager, no XLA).
This repo therefore targets the **JAX backend** (`KERAS_BACKEND=jax`).

## Phases

| Phase | Content | Status |
|---|---|---|
| 1 | Port the exact nano math to Keras 3 (masked BN, 6-op 2→2, 2-op 2→0, normalize-late by N̄=49); float parity vs torch golden vectors | **done** — `tests/test_parity_float.py` passes at 1e-5 |
| 2 | HGQ2 quantization at the 7 Brevitas-mirror points; per-tensor datalane, per-element weights; RND_CONV/SAT | **done** — invariance/masking/granularity tests pass |
| 3 | Training loop (BCE≡torch CE, AUC, EBOP beta) on the toptag h5 files | **done** (smoke: val AUC 0.64 after 4 epochs on 8k sample events); full-dataset run pending |
| 4 | Export: learned (k,i,f) → `ap_fixed` contract + snapped weights | **stub done** (`export_reference.py` → contract.json/npz); emitting `weights.h`/`types_generated.h` inside nPELICAN-fpga + bit-exact C-sim gate is future work |
| 5 | Compare learned bitwidths + firmware resources vs the Brevitas 24-bit baseline; decide adoption | not started |

## Shortcomings of the original plan, found while implementing

1. **HGQ2's default granularity breaks permutation invariance.** Datalane
   (activation) quantizers default to a learned bitwidth per tensor *position* —
   i.e. per particle index (i, j). Any i,j-indexed scale violates the workspace
   invariant (permutation equivariance). Fixed by forcing
   `heterogeneous_axis=()` on every datalane quantizer;
   `tests/test_invariants.py::test_datalane_quantizers_are_per_tensor` guards it.
   Weights stay per-element (constants in the unrolled firmware — free).
2. **Rounding-contract risk is retired.** HGQ2 supports `RND_CONV` and `SAT`
   natively, matching the firmware's `AP_RND_CONV`/`AP_SAT` exactly. The
   original plan flagged a possible round-half-up vs round-half-even mismatch;
   there is none.
3. **EBOPs are blind to most of THIS design's cost.** EBOP counts Q-layer MACs;
   here those are two tiny denses (16 weights). The dot-product front-end
   (the known DSP floor), the 484-term aggregation adders, and the explicit BN
   are custom ops with no EBOP contribution. So β-regularization optimizes the
   mixing layers only; pmu/input bitwidths — the actual DSP lever — are NOT in
   the EBOP objective and must be swept/chosen by hand, and csynth stays the
   ground truth for resources.
4. **HGQ2's default kif constraints silently cap fractional bits at 10** — below
   the 24-bit Brevitas baseline (f up to 23). Widened to ±26 (`f_bound`) /
   ±16 (`i_bound`) in `HGQ2Config`; without this the model silently trains at
   ≤~12 useful bits and a parity comparison to the Brevitas run is meaningless.
5. **Torch-side defaults are a trap for golden vectors.** `PELICANNano`'s
   constructor default `activate_agg_out=True` (ReLU on the aggregated 2→0 ops)
   differs from the training default `False` (args.py) and from the firmware
   (no ReLU on R). Golden generation must use the *training* defaults.
6. **The sample h5 files are label-sorted.** Any head-slice subsample is
   single-class (AUC undefined); `load_split(limit=...)` takes a seeded random
   subsample instead.
7. **JAX stateless training vs BN moving stats.** Custom masked BN updates its
   moving statistics via `Variable.assign` inside `call`; this works under
   Keras 3's stateless JAX path but is easy to break —
   `test_fit_smoke_and_bn_updates` guards it.
8. **Bias policy carried over, unresolved at export.** Biases (b1, b1_diag, b2)
   and BN parameters stay float during training (Brevitas D6 policy); the
   export snaps weights but the firmware-side grid for biases/BN constants is
   decided by the (future) weights.h emitter, and bit-exactness will be gated
   there, not here.
9. **No hls4ml path — deliberately.** The polished HGQ2→hls4ml codegen flow is
   not usable for the custom equivariant firmware; this repo only exports the
   quantization contract. Per-element weight bitwidths therefore only pay off
   as array-type = max over elements unless the firmware emitter learns to use
   per-literal widths (HLS trims constant multipliers anyway).
10. **EBOP β needs a schedule.** With β=1e-6 the resource term dominates the
    train loss printout (offset ≈ 3.4); β-ramp (or `hgq.utils.beta_pid`) tuning
    is part of the Phase 5 study, not done.

## Open items / next actions

- Full-dataset training run (JAX on GPU preferred) + AUC vs the Brevitas
  24-bit baseline (0.952 at h=2).
- The gating experiment from the original plan still stands: a manual per-point
  bitwidth sweep in the Brevitas flow estimates the headroom HGQ2 can win;
  if that sweep shows the model already near its bitwidth floor, Phase 5
  should conclude "don't adopt".
- `nPELICAN-fpga` loader: consume `contract.json`/`contract.npz`, emit
  `weights.h` + `types_generated.h`, gate bit-exact C-sim vs this model's
  quantized logits.
