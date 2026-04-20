import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# --- 模型函数 ---
def madau_dickinson_sfr(z, alpha, beta, zp_target, psi_0=0.015):
    alpha = max(1e-6, alpha)
    beta = max(1e-6, beta)
    pivot = (1 + zp_target) * (beta / alpha) ** (1 / (alpha + beta))
    numerator = (1 + z) ** alpha
    denominator = 1 + ((1 + z) / pivot) ** (alpha + beta)
    return psi_0 * (numerator / denominator)

# --- 参数计算函数 ---
def calculate_beta_to_match_peak(zp, alpha, target_peak_height, psi_0=0.015):
    alpha = max(1e-6, alpha)
    numerator_at_peak = psi_0 * (1 + zp) ** alpha
    if (numerator_at_peak / target_peak_height) <= 1:
        return 10.0
    beta = alpha / ((numerator_at_peak / target_peak_height) - 1)
    return beta

def get_alpha_center(zp):
    a_center_slope = (2.1 - 3.6) / (2.4 - 1.4)
    a_center_intercept = 3.6 - a_center_slope * 1.4
    return a_center_slope * zp + a_center_intercept

# --- 准备数据 ---
psi_0 = 0.015
zp_fid, a_fid, b_fid = 1.9, 2.7, 2.9
target_peak_h = madau_dickinson_sfr(zp_fid, a_fid, b_fid, zp_fid, psi_0)

# 1. 边界样本 (zp = 1.4 和 2.4)
boundary_samples = []
for zp_b in [1.4, 2.4]:
    alpha_b = get_alpha_center(zp_b)
    beta_b = calculate_beta_to_match_peak(zp_b, alpha_b, target_peak_h, psi_0)
    boundary_samples.append({'zp': zp_b, 'alpha': alpha_b, 'beta': beta_b})

# 2. 随机采样 (含扰动)
n_samples = 12
random_samples = []
for _ in range(n_samples):
    zp_s = np.random.uniform(1.4, 2.4)
    alpha_center = get_alpha_center(zp_s)
    # 1. 不加一个随机扰动，则alpha_s完全由zp计算得来
    # alpha_s = alpha_center
    # 2. 加一个随机扰动，则alpha_s完全由zp计算的基础上添加一个随机性
    alpha_s = alpha_center + np.random.uniform(-0.1, 0.1)
    beta_s = calculate_beta_to_match_peak(zp_s, alpha_s, target_peak_h, psi_0)
    random_samples.append({'zp': zp_s, 'alpha': alpha_s, 'beta': beta_s})

# --- 绘图 ---
z_range = np.linspace(0, 8, 1000)
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)

# 使用比较清淡的颜色映射（Pastel色系）
colors_map = plt.cm.Set3(np.linspace(0, 1, n_samples))

# 绘制随机线
for i, s in enumerate(random_samples):
    sfr_curve = madau_dickinson_sfr(z_range, s['alpha'], s['beta'], s['zp'])
    label_str = f"$z_p$:{s['zp']:.2f}, $\\alpha$:{s['alpha']:.2f}, $\\beta$:{s['beta']:.2f}"
    ax.plot(z_range, np.log10(sfr_curve), color=colors_map[i], alpha=0.5, lw=1.5, label=label_str, zorder=2)

# 绘制边界线 - 换成浅色系
# zp=1.4: MediumTurquoise (浅青); zp=2.4: HotPink 的淡化版
b_colors = ['#48D1CC', '#FFB6C1'] # 浅青绿 和 浅粉红
for i, s in enumerate(boundary_samples):
    sfr_curve = madau_dickinson_sfr(z_range, s['alpha'], s['beta'], s['zp'])
    label_str = f"BOUND $z_p$:{s['zp']:.1f}, $\\alpha$:{s['alpha']:.2f}, $\\beta$:{s['beta']:.2f}"
    ax.plot(z_range, np.log10(sfr_curve), color=b_colors[i], lw=4, ls='-', zorder=10, label=label_str)

# 绘制 Fiducial 基准线
sfr_fid = madau_dickinson_sfr(z_range, a_fid, b_fid, zp_fid)
ax.plot(z_range, np.log10(sfr_fid), color='#555555', linestyle='--', linewidth=2, zorder=11,
         label=f"FIDU $z_p$:{zp_fid}, $\\alpha$:{a_fid}, $\\beta$:{b_fid}")

# 辅助线
ax.axhline(np.log10(target_peak_h), color='#DDDDDD', linestyle=':', zorder=1)

# 图表修饰
ax.set_title('SFH Samples with Light Colors & Alpha Perturbations', fontsize=16, pad=20)
ax.set_xlabel('Redshift ($z$)', fontsize=13)
ax.set_ylabel('$\log_{10} \psi(z)$', fontsize=13)
ax.set_xlim(0, 8)
ax.set_ylim(-2.5, -0.5)
ax.grid(True, which='both', linestyle='-', color='#F5F5F5', zorder=0) # 极浅的网格线

# 外置图例
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0,
          title="Model Parameters", fontsize=8.5, title_fontsize=10, frameon=True)

plt.subplots_adjust(right=0.72)
plt.show()