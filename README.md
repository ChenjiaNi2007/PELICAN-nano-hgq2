# PELICAN-nano-hgq2

nanoPELICAN (Lorentz- and permutation-invariant top tagger, ~21 params) rewritten
in **Keras 3 + HGQ2** with gradient-learned per-point bitwidths, targeting the
hand-written Vitis HLS firmware in the sibling repo `nPELICAN-fpga/`. The
PyTorch + Brevitas reference lives in sibling repo `PELICAN-nano/`; this port is
float-parity-tested against it. See `PLAN.md` for status and design caveats.

Requires `KERAS_BACKEND=jax` (the torch backend runs eager and defeats the
point; scripts set it themselves).

## Setup

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Use

```bash
# train (quantized, EBOP resource regularization; --float for the baseline)
python train.py --data-dir ../PELICAN-nano/data/sample_data --epochs 8 --beta 1e-6

# start capped at the hand-tuned w6a6i6p12 budget (12-bit pmu, 6 elsewhere) so
# bits can only shrink, and give every lane — including the standalone pmu/act
# quantizers EBOP cannot see — a real per-bit cost
python train.py --data-dir ../PELICAN-nano/data/toptag --epochs 24 \
    --preset w6p12c --bit-penalty 3e-4

# dump the learned quantization contract (ap_fixed types + snapped weights);
# --preset/--beta must match training, they change the variable set
python export_reference.py --weights model/hgq2_nano.weights.h5 --beta 1e-6 \
    --out model/contract

# tests (float parity vs torch goldens, invariance, masking, granularity)
python -m pytest tests/

# regenerate golden vectors (needs the sibling repo's torch venv)
../PELICAN-nano/.venv/bin/python scripts/make_golden.py
```

## Layout

- `pelican_hgq2/layers.py` — exact Keras ports: Minkowski dots, masked BN,
  the 6 equivariant 2→2 aggregators, the 2 2→0 aggregators (normalize-late,
  fixed N̄=49), bias+diag-bias.
- `pelican_hgq2/model.py` — `PelicanNanoHGQ` with HGQ2 quantizers at the same
  7 points as the Brevitas QAT model; per-tensor activation granularity
  (permutation invariance!), per-element weights, RND_CONV/SAT.
- `train.py` / `export_reference.py` — training CLI and contract dump.
- `tests/` — parity, invariance, masking, granularity, training smoke.

## Invariants (violating any is a failed change)

- Float path numerically matches the torch reference (`test_parity_float`).
- Activation quantizers are per-tensor; nothing learned is indexed by particle.
- Padded entries are exactly zero through every stage.
- Aggregation is sum-then-divide by the constant N̄ (normalize-late), never by
  the per-event multiplicity.
- BatchNorm stays explicit float; never folded into weights.
