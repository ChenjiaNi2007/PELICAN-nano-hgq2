"""Read the learned (k, i, f) off a trained model's quantizers.

Shared by export_reference.py (writes the firmware contract) and train.py
(prints where the bits landed at the end of a run).
"""
import numpy as np


def kif_of(quantizer_layer):
    """(k, i, f) numpy arrays of a hgq Quantizer (or QDense .iq/.kq)."""
    q = quantizer_layer.quantizer
    k = np.asarray(q.k, dtype=np.int32) if not callable(q.k) else np.asarray(q.k())
    i = np.rint(np.asarray(q.i)).astype(np.int32)
    f = np.rint(np.asarray(q.f)).astype(np.int32)
    return k, i, f


def quant_points(model):
    """Ordered {name: quantizer layer} for every quantization point."""
    points = {}
    if model.pmu_quant is not None:
        points['pmu'] = model.pmu_quant
    points.update({
        'input': model.input_quant,
        'post_agg_2to2': model.mixing_2to2.iq,
        'w_2to2': model.mixing_2to2.kq,
        'act': model.act_quant,
        'post_agg_2to0': model.mixing_2to0.iq,
        'w_2to0': model.mixing_2to0.kq,
        'output': model.output_quant,
    })
    return points


def contract_of(model):
    """{name: {k, i, f, bits, shape, ap_fixed}} using the widest element."""
    contract = {}
    for name, layer in quant_points(model).items():
        k, i, f = kif_of(layer)
        km, im, fm = int(k.max()), int(i.max()), int(f.max())
        contract[name] = {
            'k': km, 'i': im, 'f': fm, 'bits': km + im + fm,
            'shape': list(np.broadcast(k, i, f).shape),
            'ap_fixed': f'ap_{"" if km else "u"}fixed<{km + im + fm},{km + im}>',
        }
    return contract


def format_contract(contract, ref=None):
    """One line per lane; with `ref`, append the delta in total bits."""
    lines = []
    for name, c in contract.items():
        line = f"  {name:<14} {c['ap_fixed']:<22} bits={c['bits']:>3}"
        if ref is not None and name in ref:
            d = c['bits'] - ref[name]['bits']
            line += f"  ({d:+d} vs ref)"
        lines.append(line)
    return '\n'.join(lines)
