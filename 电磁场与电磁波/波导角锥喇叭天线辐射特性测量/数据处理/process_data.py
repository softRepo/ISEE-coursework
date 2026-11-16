from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "原始数据"
OUT = ROOT / "结果"
FIG = OUT / "figures"


def read_two_column(path: Path):
    xs, ys = [], []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            try:
                x = float(row[0].strip())
                y = float(row[1].strip())
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def read_multi_pair(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        labels = [h.strip().replace("GHz", "") for h in header[0::2]]
        for row in reader:
            if len(row) < 6:
                continue
            vals = []
            ok = True
            for i in range(0, 6, 2):
                try:
                    vals.append((float(row[i].strip()), float(row[i + 1].strip())))
                except ValueError:
                    ok = False
                    break
            if ok:
                rows.append(vals)
    out = {}
    for col, label in enumerate(labels):
        angle = np.array([r[col][0] for r in rows], dtype=float)
        level = np.array([r[col][1] for r in rows], dtype=float)
        out[float(label)] = (angle, level)
    return out


def nearest(xs, ys, x0):
    idx = int(np.argmin(np.abs(xs - x0)))
    return xs[idx], ys[idx]


def gamma_from_vswr(vswr):
    return (vswr - 1.0) / (vswr + 1.0)


def return_loss(gamma):
    if gamma <= 0:
        return float("inf")
    return -20.0 * math.log10(gamma)


def hpbw(angle, level_db):
    """Return main-beam direction and half-power beamwidth from dB samples."""
    idx0 = int(np.argmax(level_db))
    peak_angle = angle[idx0]
    rel = level_db - level_db[idx0]
    thr = -3.0

    left = idx0
    while left > 0 and rel[left] >= thr:
        left -= 1
    right = idx0
    while right < len(rel) - 1 and rel[right] >= thr:
        right += 1

    def interp(i1, i2):
        x1, y1 = angle[i1], rel[i1]
        x2, y2 = angle[i2], rel[i2]
        if y2 == y1:
            return x1
        return x1 + (thr - y1) * (x2 - x1) / (y2 - y1)

    left_cross = interp(left, left + 1) if left < idx0 else angle[left]
    right_cross = interp(right - 1, right) if right > idx0 else angle[right]
    return peak_angle, right_cross - left_cross, left_cross, right_cross


def plot_s_parameters(freq_ghz, vswr, s21_freq_ghz, s21_db):
    gamma = np.array([gamma_from_vswr(v) for v in vswr])
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True)
    axes[0].plot(freq_ghz, vswr, color="#2563eb", lw=1.5)
    axes[0].set_ylabel("VSWR")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Measured S11 and S21 Sweep")
    ax0b = axes[0].twinx()
    ax0b.plot(freq_ghz, gamma, color="#dc2626", lw=1.0, alpha=0.75)
    ax0b.set_ylabel("|Gamma|")

    axes[1].plot(s21_freq_ghz, s21_db, color="#059669", lw=1.5)
    axes[1].set_xlabel("Frequency (GHz)")
    axes[1].set_ylabel("S21 (dB)")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "s_parameter_sweep.png", dpi=220)
    plt.close(fig)


def plot_pattern(patterns, filename, title, min_db):
    fig = plt.figure(figsize=(6.4, 5.8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetamin(-135)
    ax.set_thetamax(135)
    colors = {6.5: "#2563eb", 7.5: "#dc2626", 8.5: "#7c7c00"}
    for freq, (angle, level) in patterns.items():
        rel = level - np.max(level)
        rel = np.clip(rel, min_db, 0)
        radius = rel - min_db
        ax.plot(np.deg2rad(angle), radius, lw=1.5, label=f"{freq:.1f} GHz", color=colors.get(freq))
    ticks = np.arange(0, abs(min_db) + 1, 10)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{min_db + t:.0f}" for t in ticks])
    ax.set_rlim(0, abs(min_db))
    ax.set_title(title, pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False)
    ax.grid(True, alpha=0.35)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / filename, dpi=220)
    plt.close(fig)


def plot_polarization(normal, grid):
    angle_n, amp_n = normal
    angle_g, amp_g = grid
    db_n = 20 * np.log10(np.maximum(amp_n, 1e-6))
    db_g = 20 * np.log10(np.maximum(amp_g, 1e-6))
    theory = 20 * np.log10(np.maximum(np.cos(np.deg2rad(angle_n)), 1e-4))
    theory = np.clip(theory, -24, 1)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(angle_n, db_n, label="Normal measurement", color="#2563eb", lw=1.6)
    ax.plot(angle_g, db_g, label="With polarizing grid", color="#dc2626", lw=1.4)
    ax.plot(angle_n, theory, "--", label="20log10(cos alpha)", color="#111827", lw=1.1)
    ax.set_xlim(0, 90)
    ax.set_ylim(-20, 1)
    ax.set_xlabel("Polarization angle alpha (deg)")
    ax.set_ylabel("Normalized S21 (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "polarization_curves.png", dpi=220)
    plt.close(fig)


def plot_distance(distance):
    x, amp = distance
    measured_db = 20 * np.log10(np.maximum(amp, 1e-6))
    r0 = 0.55
    r = r0 + x
    theory = {n: -10 * n * np.log10(r / r0) for n in (1, 2, 3)}

    fit_mask = x > 0.05
    log_ratio = np.log10((r[fit_mask]) / r0)
    n_fit = float(-np.dot(measured_db[fit_mask], log_ratio) / (10 * np.dot(log_ratio, log_ratio)))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, measured_db, "o-", label="Measured", color="#2563eb", lw=1.5, ms=3.8)
    styles = {1: ":", 2: "--", 3: "-."}
    for n, db in theory.items():
        ax.plot(x, db, styles[n], label=f"1/R^{n}", lw=1.2)
    ax.set_xlabel("Rail displacement x (m), R = 0.55 m + x")
    ax.set_ylabel("Relative received power (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.text(0.03, 0.06, f"least-squares exponent n = {n_fit:.2f}", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG / "distance_power_fit.png", dpi=220)
    plt.close(fig)
    return r0, n_fit


def main():
    global OUT, FIG
    parser = argparse.ArgumentParser(description="Process horn antenna measurement data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "结果")
    args = parser.parse_args()
    OUT = args.output_dir
    FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    f_mhz, vswr = read_two_column(DATA / "S11.csv")
    s21_mhz, s21_db = read_two_column(DATA / "S21.csv")
    freq = f_mhz / 1000.0
    s21_freq = s21_mhz / 1000.0
    plot_s_parameters(freq, vswr, s21_freq, s21_db)

    h_patterns = read_multi_pair(DATA / "方向图.csv")
    e_patterns = read_multi_pair(DATA / "垂直方向图.csv")
    plot_pattern(h_patterns, "h_plane_patterns.png", "Horizontal Plane Patterns (normalized dB)", -45)
    plot_pattern(e_patterns, "e_plane_patterns.png", "Vertical Plane Patterns (normalized dB)", -35)

    normal = read_two_column(DATA / "常规极化.csv")
    grid = read_two_column(DATA / "栅网极化.csv")
    plot_polarization(normal, grid)
    r0, n_fit = plot_distance(read_two_column(DATA / "距离.csv"))

    key_freqs = [6.5, 7.5, 8.0, 8.5]
    rows = []
    for f0 in key_freqs:
        _, v = nearest(freq, vswr, f0)
        _, s = nearest(s21_freq, s21_db, f0)
        g = gamma_from_vswr(v)
        rows.append((f0, v, g, return_loss(g), s))

    min_idx = int(np.argmin(vswr))
    max_s21_idx = int(np.argmax(s21_db))

    hp_rows = []
    for f0 in sorted(h_patterns):
        h_ang, h_lvl = h_patterns[f0]
        e_ang, e_lvl = e_patterns[f0]
        h_peak, h_bw, h_l, h_r = hpbw(h_ang, h_lvl)
        e_peak, e_bw, e_l, e_r = hpbw(e_ang, e_lvl)
        gain_bw = 10 * math.log10(32400.0 / (h_bw * e_bw))
        hp_rows.append((f0, h_peak, h_bw, e_peak, e_bw, gain_bw))

    pol_angles = [0, 30, 45, 60, 80, 90]
    pol_rows = []
    angle_n, amp_n = normal
    angle_g, amp_g = grid
    for a in pol_angles:
        _, an = nearest(angle_n, amp_n, a)
        _, ag = nearest(angle_g, amp_g, a)
        pol_rows.append((a, an, 20 * math.log10(max(an, 1e-9)), ag, 20 * math.log10(max(ag, 1e-9))))

    lam = 3e8 / 8.0e9
    tx_de, tx_dh = 0.037, 0.082
    rx_de, rx_dh = 0.105, 0.141

    def aperture_gain(de, dh, eta=1.0):
        return 10 * math.log10(eta * 4 * math.pi * de * dh / lam**2)

    def bw_h(dh):
        return math.degrees(1.18 * lam / dh)

    def bw_e(de):
        return math.degrees(0.89 * lam / de)

    far_tx = 2 * max(tx_de, tx_dh) ** 2 / lam
    far_rx = 2 * max(rx_de, rx_dh) ** 2 / lam

    summary = OUT / "processed_summary.txt"
    with summary.open("w", encoding="utf-8") as f:
        f.write("S11/S21 key points\n")
        f.write("freq_GHz,VSWR,abs_gamma,return_loss_dB,S21_dB\n")
        for row in rows:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")
        f.write(
            f"\nMinimum VSWR: {vswr[min_idx]:.4f} at {freq[min_idx]:.4f} GHz; "
            f"|Gamma|={gamma_from_vswr(vswr[min_idx]):.4f}; "
            f"RL={return_loss(gamma_from_vswr(vswr[min_idx])):.2f} dB\n"
        )
        f.write(f"Maximum S21: {s21_db[max_s21_idx]:.3f} dB at {s21_freq[max_s21_idx]:.4f} GHz\n")

        f.write("\nPattern HPBW and beamwidth gain\n")
        f.write("freq_GHz,H_peak_deg,H_HPBW_deg,E_peak_deg,E_HPBW_deg,GBW_dBi\n")
        for row in hp_rows:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")

        f.write("\nPolarization selected points\n")
        f.write("angle_deg,normal_amp,normal_dB,grid_amp,grid_dB\n")
        for row in pol_rows:
            f.write(",".join(f"{v:.4f}" for v in row) + "\n")

        f.write("\nDistance fit\n")
        f.write(f"R0_m={r0:.3f}, fitted_power_exponent={n_fit:.3f}\n")

        f.write("\nTheoretical values at 8.00 GHz, k=1\n")
        f.write(f"lambda_m={lam:.5f}\n")
        f.write(f"Tx_gain_dBi={aperture_gain(tx_de, tx_dh):.2f}, Tx_H_HPBW_deg={bw_h(tx_dh):.2f}, Tx_E_HPBW_deg={bw_e(tx_de):.2f}\n")
        f.write(f"Rx_gain_dBi={aperture_gain(rx_de, rx_dh):.2f}, Rx_H_HPBW_deg={bw_h(rx_dh):.2f}, Rx_E_HPBW_deg={bw_e(rx_de):.2f}\n")
        f.write(f"Far_field_tx_m={far_tx:.3f}, Far_field_rx_m={far_rx:.3f}\n")


if __name__ == "__main__":
    main()
