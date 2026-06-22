import matplotlib
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)

ZP_MIN = 0.85
ZP_MAX_DESIRED = 2.85
N_RANDOM_SAMPLES = 12
KAPPA_TOTAL = 2.7 + 2.9


def madau_dickinson_sfr(z, alpha, beta, zp_target, psi_0=0.015):
    alpha = np.maximum(1e-6, alpha)
    beta = np.maximum(1e-6, beta)
    pivot = (1 + zp_target) * (beta / alpha) ** (1 / (alpha + beta))
    numerator = (1 + z) ** alpha
    denominator = 1 + ((1 + z) / pivot) ** (alpha + beta)
    return psi_0 * (numerator / denominator)


def get_alpha_center(zp):
    slope = (2.1 - 3.6) / (2.4 - 1.4)
    intercept = 2.7 - slope * 1.9
    return slope * zp + intercept


def calculate_beta_from_alpha(alpha, kappa_total=KAPPA_TOTAL):
    beta = kappa_total - alpha
    return np.maximum(1e-6, beta)


psi_0 = 0.015
zp_fid, a_fid, b_fid = 1.9, 2.7, 2.9
target_peak_h = madau_dickinson_sfr(zp_fid, a_fid, b_fid, zp_fid, psi_0)

alpha_min_effective = get_alpha_center(ZP_MAX_DESIRED)
alpha_max_effective = get_alpha_center(ZP_MIN)
beta_min_effective = calculate_beta_from_alpha(alpha_max_effective)
beta_max_effective = calculate_beta_from_alpha(alpha_min_effective)

print(f"zp range: [{ZP_MIN:.6f}, {ZP_MAX_DESIRED:.6f}]")
print(f"alpha range: [{alpha_min_effective:.6f}, {alpha_max_effective:.6f}]")
print(f"beta range: [{beta_min_effective:.6f}, {beta_max_effective:.6f}]")

boundary_samples = []
for zp_b in [ZP_MIN, ZP_MAX_DESIRED]:
    alpha_b = get_alpha_center(zp_b)
    beta_b = calculate_beta_from_alpha(alpha_b)
    boundary_samples.append({"zp": zp_b, "alpha": alpha_b, "beta": beta_b})

random_samples = []
for _ in range(N_RANDOM_SAMPLES):
    zp_s = np.random.uniform(ZP_MIN, ZP_MAX_DESIRED)
    alpha_s = get_alpha_center(zp_s)
    beta_s = calculate_beta_from_alpha(alpha_s)
    random_samples.append({"zp": zp_s, "alpha": alpha_s, "beta": beta_s})

z_range = np.linspace(0, 8, 1000)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)

colors_map = plt.cm.Set3(np.linspace(0, 1, N_RANDOM_SAMPLES))

for i, sample in enumerate(random_samples):
    sfr_curve = madau_dickinson_sfr(z_range, sample["alpha"], sample["beta"], sample["zp"])
    label_str = (
        f"$z_p$:{sample['zp']:.2f}, "
        f"$\\alpha$:{sample['alpha']:.2f}, "
        f"$\\beta$:{sample['beta']:.2f}"
    )
    ax.plot(
        z_range,
        np.log10(sfr_curve),
        color=colors_map[i],
        alpha=0.5,
        lw=1.5,
        label=label_str,
        zorder=2,
    )

b_colors = ["#48D1CC", "#FFB6C1"]
for i, sample in enumerate(boundary_samples):
    sfr_curve = madau_dickinson_sfr(z_range, sample["alpha"], sample["beta"], sample["zp"])
    label_str = (
        f"BOUND $z_p$:{sample['zp']:.2f}, "
        f"$\\alpha$:{sample['alpha']:.2f}, "
        f"$\\beta$:{sample['beta']:.2f}"
    )
    ax.plot(
        z_range,
        np.log10(sfr_curve),
        color=b_colors[i],
        lw=4,
        ls="-",
        zorder=10,
        label=label_str,
    )

sfr_fid = madau_dickinson_sfr(z_range, a_fid, b_fid, zp_fid)
ax.plot(
    z_range,
    np.log10(sfr_fid),
    color="#555555",
    linestyle="--",
    linewidth=2,
    zorder=11,
    label=f"FIDU $z_p$:{zp_fid}, $\\alpha$:{a_fid}, $\\beta$:{b_fid}",
)

ax.axhline(np.log10(target_peak_h), color="#DDDDDD", linestyle=":", zorder=1)

ax.set_title("SFH Samples with Updated zp-alpha_z-beta_z Mapping", fontsize=16, pad=20)
ax.set_xlabel("Redshift ($z$)", fontsize=13)
ax.set_ylabel("$\\log_{10} \\psi(z)$", fontsize=13)
ax.set_xlim(0, 8)
ax.set_ylim(-2.5, -0.5)
ax.grid(True, which="both", linestyle="-", color="#F5F5F5", zorder=0)

ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    borderaxespad=0,
    title="Model Parameters",
    fontsize=8.5,
    title_fontsize=10,
    frameon=True,
)

plt.subplots_adjust(right=0.72)
output_dir = Path(__file__).with_name("zpFig")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "getZp.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved figure to {output_path}")
