"""
Bootstrap convergence analysis.
Resamples per-file metric values at increasing n, computes JS divergence
against the fixed reference distribution, reports mean ± std across 500 trials.
"""
import json, numpy as np

rng = np.random.default_rng(0)
N_BOOT = 500
METRICS = ["pitch_class_entropy","pitch_range","note_density",
           "average_pitch","pitch_std","average_duration",
           "rhythmic_entropy","polyphony"]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def js_div(p_vals, q_vals, bins=30):
    lo = min(p_vals.min(), q_vals.min())
    hi = max(p_vals.max(), q_vals.max()) + 1e-9
    p_hist, _ = np.histogram(p_vals, bins=bins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, range=(lo, hi), density=True)
    p_hist += 1e-10; q_hist += 1e-10
    p_hist /= p_hist.sum(); q_hist /= q_hist.sum()
    m = 0.5*(p_hist + q_hist)
    kl = lambda a, b: np.sum(a * np.log(a/b))
    return 0.5*kl(p_hist, m) + 0.5*kl(q_hist, m)

def extract(records, key):
    return np.array([r[key] for r in records if r.get(key) is not None], dtype=float)

for label, path in [("BARPOS", "eval_results/barpos/metrics.json"),
                    ("Baseline", "eval_results/baseline/metrics.json")]:
    m = load_json(path)
    gen_records = m["generated"]   # list of dicts
    ref_records = m["reference"]

    gen_vals = {k: extract(gen_records, k) for k in METRICS}
    ref_vals = {k: extract(ref_records, k) for k in METRICS}
    n_total  = len(gen_records)

    print(f"\n=== {label}  (n_generated = {n_total}) ===")
    print(f"{'n':>5}  {'mean JS':>10}  {'std JS':>10}  {'CV%':>8}")

    for n in [5, 10, 15, 20, 30, 50]:
        if n > n_total:
            print(f"{n:>5}  -- need more samples --")
            continue
        js_samples = []
        for _ in range(N_BOOT):
            idx = rng.choice(n_total, size=n, replace=True)
            js_per_metric = []
            for k in METRICS:
                subset = gen_vals[k][idx]
                js_per_metric.append(js_div(subset, ref_vals[k]))
            js_samples.append(np.mean(js_per_metric))
        mu  = np.mean(js_samples)
        std = np.std(js_samples)
        cv  = 100*std/mu
        print(f"{n:>5}  {mu:>10.4f}  {std:>10.4f}  {cv:>7.1f}%")
