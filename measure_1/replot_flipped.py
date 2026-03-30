"""
replot_flipped.py — Quick replot of saved LIF results with flipped directions

Loads the lif_results_*.npy files from a folder, applies (360 - predicted) % 360
to all predictions, recalculates errors, and generates corrected plots.

NO Brian2 simulation needed — just loads and replots.

Usage:
    python replot_flipped.py                                         # auto-detect
    python replot_flipped.py recordings/Goldberg_3sec/Foci           # specific folder
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MEASURE_DIR = Path(__file__).parent

PARAM_SETS = [
    ("optimized",      58,   0.80),
    ("respeaker",      55,   1.00),
    ("tight_leak",     40,   0.80),
    ("loose_leak",    100,   0.80),
    ("low_thresh",     58,   0.50),
    ("high_thresh",    58,   1.20),
]

def load_and_flip(folder):
    """Load all lif_results_*.npy, flip + remove offset, return corrected results."""
    folder = Path(folder)
    sound_name = folder.name

    # --- Step 1: Find the calibration offset from the first available param set ---
    # The LIF reports some raw angle at true=0°. That raw angle is our offset.
    cal_offset = None
    for pname, _, _ in PARAM_SETS:
        npy_path = folder / f"lif_results_{pname}.npy"
        if not npy_path.exists():
            continue
        data = np.load(npy_path, allow_pickle=True).item()
        raw_results = data['results']
        # Find the entry with true_angle = 0
        for ta, bp, _, _, _ in raw_results:
            if ta == 0 and bp is not None:
                cal_offset = bp
                break
        if cal_offset is not None:
            break

    if cal_offset is None:
        print("  WARNING: Could not find calibration offset (no data at 0°)")
        print("  Using offset = 180° as fallback")
        cal_offset = 180.0

    print(f"  Calibration: LIF reports {cal_offset:.0f}° at true 0°")
    print(f"  Correction: corrected = ({cal_offset:.0f} - raw_pred) % 360\n")

    # --- Step 2: Apply correction to all param sets ---
    all_results = {}
    for pname, tau, vt in PARAM_SETS:
        npy_path = folder / f"lif_results_{pname}.npy"
        if not npy_path.exists():
            print(f"  Skipping {pname} (no .npy file)")
            continue

        data = np.load(npy_path, allow_pickle=True).item()
        raw_results = data['results']

        corrected = []
        for ta, bp, _, nf, ni in raw_results:
            if bp is not None:
                bp_corr = (cal_offset - bp) % 360
                # Map to nearest valid range for 0-180 measurement
                if bp_corr > 180:
                    bp_corr = bp_corr - 360
                err = abs(bp_corr - ta)
                if err > 180:
                    err = 360 - err
                corrected.append((ta, bp_corr, err, nf, ni))
            else:
                corrected.append((ta, None, None, nf, ni))

        all_results[pname] = corrected

        valid = [err for _, _, err, _, _ in corrected if err is not None]
        if valid:
            print(f"  {pname:16s}: mean={np.mean(valid):5.1f}°  max={np.max(valid):5.1f}°  "
                  f"success={len(valid)}/{len(corrected)}")
        else:
            print(f"  {pname:16s}: no valid data")

    return all_results, sound_name


def plot_comparison(all_results, sound_name, param_sets, save_dir):
    """Replot with flipped data."""
    # Get all true angles from first available result set
    for pname in all_results:
        true_angles = [ta for ta, _, _, _, _ in all_results[pname]]
        break
    max_angle = max(true_angles)

    n_params = len(param_sets)
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))

    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    fig.suptitle(f"LIF SNN Parameter Comparison — {sound_name} (direction corrected)",
                 fontsize=16, fontweight='bold')

    # 1. Tracking
    ax1 = axes[0, 0]
    ax1.plot([0, max_angle], [0, max_angle], 'k--', lw=2, alpha=0.4, label='Perfect')
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        valid = [(ta, bp) for ta, bp, _, _, _ in all_results[pname] if bp is not None]
        if valid:
            ax1.plot([t for t, _ in valid], [p for _, p in valid], 'o-',
                     color=colors[i], markersize=4, label=f"{pname} (τ={tau}, Vt={vt})")
    ax1.set_xlabel('True Angle (°)')
    ax1.set_ylabel('LIF Predicted Angle (°)')
    ax1.set_title('Tracking Accuracy')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. Error per angle
    ax2 = axes[0, 1]
    bar_w = 0.8 / n_params
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        valid = [(ta, err) for ta, _, err, _, _ in all_results[pname] if err is not None]
        if valid:
            x = np.array([t for t, _ in valid]) + i * bar_w
            ax2.bar(x, [e for _, e in valid], width=bar_w, color=colors[i],
                    alpha=0.7, label=pname)
    ax2.set_xlabel('True Angle (°)')
    ax2.set_ylabel('Absolute Error (°)')
    ax2.set_title('Error at Each Position')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Box plot
    ax3 = axes[1, 0]
    error_data, box_labels, box_colors = [], [], []
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        valid = [err for _, _, err, _, _ in all_results[pname] if err is not None]
        if valid:
            error_data.append(valid)
            box_labels.append(f"{pname}\nτ={tau},Vt={vt}")
            box_colors.append(colors[i])
    if error_data:
        bp = ax3.boxplot(error_data, tick_labels=box_labels, patch_artist=True)
        for patch, c in zip(bp['boxes'], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
    ax3.set_ylabel('Error (°)')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='x', labelsize=7)

    # 4. Mean error bar chart
    ax4 = axes[1, 1]
    means, srs, bar_labels, bar_colors = [], [], [], []
    for i, (pname, tau, vt) in enumerate(param_sets):
        if pname not in all_results:
            continue
        res = all_results[pname]
        valid = [err for _, _, err, _, _ in res if err is not None]
        if valid:
            means.append(np.mean(valid))
            srs.append(100 * len(valid) / len(res))
        else:
            means.append(90)
            srs.append(0)
        bar_labels.append(f"{pname}\nτ={tau}")
        bar_colors.append(colors[i])

    x_pos = np.arange(len(bar_labels))
    bars = ax4.bar(x_pos, means, color=bar_colors, alpha=0.7)
    for bar, me, sr in zip(bars, means, srs):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{me:.1f}°\n({sr:.0f}%)', ha='center', fontsize=8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(bar_labels, fontsize=7)
    ax4.set_ylabel('Mean Error (°)')
    ax4.set_title('Mean Error & Success Rate')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    plot_path = save_dir / f"lif_comparison_{sound_name}_flipped.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n  Plot saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    recordings_root = MEASURE_DIR / "recordings"

    if len(sys.argv) > 1:
        folders = [Path(sys.argv[i]) for i in range(1, len(sys.argv))]
    else:
        # Auto-detect folders with lif_results_*.npy
        folders = []
        for session in sorted(recordings_root.iterdir()):
            if session.is_dir():
                if list(session.glob("lif_results_*.npy")):
                    folders.append(session)
                else:
                    for sub in sorted(session.iterdir()):
                        if sub.is_dir() and list(sub.glob("lif_results_*.npy")):
                            folders.append(sub)

    if not folders:
        print("No folders with lif_results_*.npy found!")
        sys.exit(1)

    for folder in folders:
        print(f"\n{'='*60}")
        print(f" Replotting: {folder.name} (flipped)")
        print(f"{'='*60}")

        all_results, sound_name = load_and_flip(folder)
        if all_results:
            plot_comparison(all_results, sound_name, PARAM_SETS, folder)

    print("\nDone.")
