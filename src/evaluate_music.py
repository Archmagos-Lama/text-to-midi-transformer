"""
evaluate_music.py
比较生成MIDI和原曲在客观指标上的分布差距
用法：
    python evaluate_music.py \
        --generated  generated_midis/   \
        --reference  data/touhou_midi_collection/ \
        --n_ref      20 \
        --out        eval_results/
"""

import argparse
import random
import math
import os
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from miditoolkit import MidiFile
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm


# ───────────────────────────── 指标函数 ─────────────────────────────

def get_notes(midi: MidiFile):
    """把所有轨道的 note 打平成一个列表"""
    notes = []
    for inst in midi.instruments:
        if not inst.is_drum:
            notes.extend(inst.notes)
    return sorted(notes, key=lambda n: n.start)


def pitch_class_histogram(notes) -> np.ndarray:
    """12维音高类分布（归一化）"""
    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += 1
    s = hist.sum()
    return hist / s if s > 0 else hist


def pitch_class_entropy(notes) -> float:
    """音高类熵（bits），越高越多样"""
    hist = pitch_class_histogram(notes)
    hist = hist[hist > 0]
    return float(scipy_entropy(hist, base=2))


def pitch_range(notes) -> int:
    """最高音 - 最低音（半音数）"""
    if not notes:
        return 0
    pitches = [n.pitch for n in notes]
    return max(pitches) - min(pitches)


def note_density(midi: MidiFile, notes) -> float:
    """每拍平均音符数"""
    if not notes:
        return 0.0
    ticks_per_beat = midi.ticks_per_beat
    total_beats = (notes[-1].end - notes[0].start) / ticks_per_beat
    return len(notes) / total_beats if total_beats > 0 else 0.0


def average_pitch(notes) -> float:
    if not notes:
        return 0.0
    return float(np.mean([n.pitch for n in notes]))


def pitch_std(notes) -> float:
    if not notes:
        return 0.0
    return float(np.std([n.pitch for n in notes]))


def average_duration(midi: MidiFile, notes) -> float:
    """平均音符时值（拍）"""
    if not notes:
        return 0.0
    tpb = midi.ticks_per_beat
    durs = [(n.end - n.start) / tpb for n in notes]
    return float(np.mean(durs))


def rhythmic_entropy(midi: MidiFile, notes) -> float:
    """IOI（音符间隔）的熵，衡量节奏多样性"""
    if len(notes) < 2:
        return 0.0
    tpb = midi.ticks_per_beat
    iois = []
    for i in range(1, len(notes)):
        ioi = (notes[i].start - notes[i-1].start) / tpb
        if ioi > 0:
            iois.append(round(ioi, 2))   # 量化到0.01拍避免碎片化
    if not iois:
        return 0.0
    cnt = Counter(iois)
    total = sum(cnt.values())
    probs = np.array([v / total for v in cnt.values()])
    return float(scipy_entropy(probs, base=2))


def polyphony(notes) -> float:
    """平均同时发音音符数（粗估）"""
    if not notes:
        return 0.0
    events = []
    for n in notes:
        events.append((n.start, 1))
        events.append((n.end, -1))
    events.sort()
    cur = 0
    samples = []
    for _, delta in events:
        cur += delta
        samples.append(cur)
    return float(np.mean(samples))


def compute_metrics(midi_path: Path) -> dict | None:
    try:
        midi = MidiFile(midi_path)
        notes = get_notes(midi)
        if len(notes) < 10:
            return None
        return {
            "pitch_class_entropy":  pitch_class_entropy(notes),
            "pitch_range":          pitch_range(notes),
            "note_density":         note_density(midi, notes),
            "average_pitch":        average_pitch(notes),
            "pitch_std":            pitch_std(notes),
            "average_duration":     average_duration(midi, notes),
            "rhythmic_entropy":     rhythmic_entropy(midi, notes),
            "polyphony":            polyphony(notes),
        }
    except Exception as e:
        print(f"  skip {midi_path.name}: {e}")
        return None


# ───────────────────────────── 统计 & 可视化 ─────────────────────────────

METRIC_LABELS = {
    "pitch_class_entropy":  "Pitch Class Entropy (bits)",
    "pitch_range":          "Pitch Range (semitones)",
    "note_density":         "Note Density (notes/beat)",
    "average_pitch":        "Average Pitch (MIDI)",
    "pitch_std":            "Pitch Std Dev",
    "average_duration":     "Average Duration (beats)",
    "rhythmic_entropy":     "Rhythmic Entropy (bits)",
    "polyphony":            "Avg Polyphony",
}


def kl_divergence(a: np.ndarray, b: np.ndarray, bins=30) -> float:
    """用直方图估计KL散度（对称JS散度）"""
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max()) + 1e-9
    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    ha = ha + 1e-10
    hb = hb + 1e-10
    ha /= ha.sum()
    hb /= hb.sum()
    return float(jensenshannon(ha, hb))   # JS散度，0=完全相同，1=完全不同


def print_summary(gen_metrics, ref_metrics):
    print("\n" + "="*62)
    print(f"{'Metric':<25} {'Generated':>10} {'Reference':>10} {'JS-div':>8}")
    print("-"*62)
    for key, label in METRIC_LABELS.items():
        gv = np.array([m[key] for m in gen_metrics])
        rv = np.array([m[key] for m in ref_metrics])
        js = kl_divergence(gv, rv)
        print(f"{key:<25}  {gv.mean():>8.3f}   {rv.mean():>8.3f}   {js:>7.4f}")
    print("="*62)
    print("JS divergence: 0 = identical distribution, 1 = completely different")
    print()


def plot_distributions(gen_metrics, ref_metrics, out_dir: Path):
    n = len(METRIC_LABELS)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = axes.flatten()

    for idx, (key, label) in enumerate(METRIC_LABELS.items()):
        ax = axes[idx]
        gv = np.array([m[key] for m in gen_metrics])
        rv = np.array([m[key] for m in ref_metrics])

        lo = min(gv.min(), rv.min())
        hi = max(gv.max(), rv.max())
        bins = np.linspace(lo, hi, 25)

        ax.hist(rv, bins=bins, alpha=0.55, color="#4C8BE2", label="Reference", density=True)
        ax.hist(gv, bins=bins, alpha=0.55, color="#E2774C", label="Generated", density=True)

        js = kl_divergence(gv, rv)
        ax.set_title(f"{label}\nJS={js:.4f}", fontsize=9)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=8)

    # 隐藏多余子图
    for i in range(len(METRIC_LABELS), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Generated vs Reference MIDI — Feature Distributions", fontsize=12, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


def plot_radar(gen_metrics, ref_metrics, out_dir: Path):
    """雷达图：归一化后的均值对比"""
    keys = list(METRIC_LABELS.keys())
    labels = [METRIC_LABELS[k] for k in keys]

    gen_means = np.array([np.mean([m[k] for m in gen_metrics]) for k in keys])
    ref_means = np.array([np.mean([m[k] for m in ref_metrics]) for k in keys])

    # 用 ref 归一化
    scale = np.where(ref_means != 0, ref_means, 1.0)
    gen_norm = gen_means / scale
    ref_norm = ref_means / scale  # 全1

    angles = np.linspace(0, 2 * np.pi, len(keys), endpoint=False).tolist()
    angles += angles[:1]
    gen_norm = np.append(gen_norm, gen_norm[0])
    ref_norm = np.append(ref_norm, ref_norm[0])

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, ref_norm, "o-", color="#4C8BE2", linewidth=1.5, label="Reference")
    ax.fill(angles, ref_norm, alpha=0.15, color="#4C8BE2")
    ax.plot(angles, gen_norm, "o-", color="#E2774C", linewidth=1.5, label="Generated")
    ax.fill(angles, gen_norm, alpha=0.15, color="#E2774C")

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_title("Feature Radar (normalized to Reference)", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    out_path = out_dir / "radar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


# ───────────────────────────── main ─────────────────────────────

def collect_midis(directory: Path, n: int | None = None) -> list[Path]:
    files = []
    for root, _, fnames in os.walk(directory):
        for f in fnames:
            if f.lower().endswith((".mid", ".midi")):
                files.append(Path(root) / f)
    if n and len(files) > n:
        files = random.sample(files, n)
    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True,
                        help="生成MIDI所在目录")
    parser.add_argument("--reference", type=Path, required=True,
                        help="原曲MIDI所在目录")
    parser.add_argument("--n_ref", type=int, default=20,
                        help="从原曲目录随机采样多少首（默认20）")
    parser.add_argument("--n_gen", type=int, default=None,
                        help="从生成目录采样多少首（默认全部）")
    parser.add_argument("--out", type=Path, default=Path("eval_results"),
                        help="结果输出目录")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[gen]  collecting from {args.generated}")
    gen_files = collect_midis(args.generated, args.n_gen)
    print(f"       found {len(gen_files)} files")

    print(f"[ref]  collecting from {args.reference} (sample {args.n_ref})")
    ref_files = collect_midis(args.reference, args.n_ref)
    print(f"       found {len(ref_files)} files")

    print("\nComputing metrics for generated MIDIs...")
    gen_metrics = [r for f in tqdm(gen_files) if (r := compute_metrics(f)) is not None]

    print("Computing metrics for reference MIDIs...")
    ref_metrics = [r for f in tqdm(ref_files) if (r := compute_metrics(f)) is not None]

    print(f"\nValid:  generated={len(gen_metrics)},  reference={len(ref_metrics)}")

    if len(gen_metrics) < 3 or len(ref_metrics) < 3:
        print("Too few valid files, aborting.")
        return

    print_summary(gen_metrics, ref_metrics)
    plot_distributions(gen_metrics, ref_metrics, args.out)
    plot_radar(gen_metrics, ref_metrics, args.out)

    # 保存原始数据
    import json
    with open(args.out / "metrics.json", "w") as f:
        json.dump({"generated": gen_metrics, "reference": ref_metrics}, f, indent=2)
    print(f"[saved] {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
