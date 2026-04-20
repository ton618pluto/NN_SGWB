import numpy as np

import numpy as np


def madau_dickinson_sfr(z, alpha, beta, zp_target, psi_0=0.015):
    """
    支持向量化计算的 Madau-Dickinson SFRD 公式。
    """
    # 使用 np.maximum 替换原生 max，支持数组逐元素对比
    alpha = np.maximum(1e-6, alpha)
    beta = np.maximum(1e-6, beta)

    # 所有的数学运算 (+, *, **, /) 在 NumPy 中默认支持数组
    pivot = (1 + zp_target) * (beta / alpha) ** (1 / (alpha + beta))

    numerator = (1 + z) ** alpha
    denominator = 1 + ((1 + z) / pivot) ** (alpha + beta)

    return psi_0 * (numerator / denominator)


def calculate_beta_to_match_peak(zp, alpha, target_peak_height, psi_0=0.015):
    """
    支持向量化计算的 Beta 解析解。
    """
    alpha = np.maximum(1e-6, alpha)

    numerator_at_peak = psi_0 * (1 + zp) ** alpha

    # 计算比例
    ratio = numerator_at_peak / target_peak_height

    # 使用 np.where 替换 if-else 逻辑
    # 当 ratio > 1 时执行公式计算，否则返回默认值 10.0
    beta = np.where(
        ratio > 1.0,
        alpha / (ratio - 1.0),
        10.0
    )

    # 再次确保 beta 为正数
    return np.maximum(1e-6, beta)

def sample_joint_hyperparameters_separate(n_samples=10000, seed=None):
    """
    联合采样超参数：
    - Madau–Dickinson 红移模型
    - PowerLaw + Peak 质量模型

    返回
    -------
    红移超参数（numpy数组）:
        alpha_z, beta_z, zp

    质量超参数（numpy数组）:
        alpha_m, beta_q, m_min, m_max,
        lambda_peak, delta_m, mu_m, sigma_m
    """

    if seed is not None:
        np.random.seed(seed)

    # ---------- 红移超参数 ----------
    # alpha_z = np.random.uniform(2.1, 3.6, n_samples)
    # beta_z  = np.random.uniform(2.5, 3.3, n_samples)
    zp      = np.random.uniform(1.4, 2.4, n_samples)
    slope = (2.1 - 3.6) / (2.4 - 1.4)  # 结果为 -1.5
    intercept = 3.6 - slope * 1.4  # 结果为 5.7
    alpha_z = slope * zp + intercept

    zp_fid, a_fid, b_fid = 1.9, 2.7, 2.9
    psi_0=0.015
    target_peak_height = madau_dickinson_sfr(zp_fid, a_fid, b_fid, zp_fid, psi_0)
    beta_z = calculate_beta_to_match_peak(zp, alpha_z, target_peak_height, psi_0)
    # r0      = np.random.uniform(250,370,n_samples)

    # ---------- 质量超参数 ----------
    # alpha_m = np.random.uniform(-3.0, 7.0, n_samples)
    alpha_m = np.random.uniform(3.0, 3.5, n_samples)
    m_max = np.random.uniform(10.0, 100.0, n_samples)

    delta_m = np.random.uniform(0.0, 10.0, n_samples)
    m_min =np.random.uniform(2.0,10.0,n_samples)

    # lambda_peak = np.random.uniform(0.07, 0.18, n_samples)
    lambda_peak = np.random.uniform(0.02, 0.06, n_samples)
    mu_m        = np.random.uniform(33.2, 35.4, n_samples)
    # sigma_m     = np.random.uniform(0.1, 0.9, n_samples)
    sigma_m     = np.random.uniform(3.52, 3.60, n_samples)
    beta_q = np.random.uniform(-5.0, 5.0, n_samples)

    return (
        alpha_z, beta_z, zp,
        alpha_m, m_max,
        delta_m,m_min,
        lambda_peak,mu_m,sigma_m,beta_q
    )

# --------- 使用示例 ----------
if __name__ == "__main__":
    n_samples=100
    alpha_z, beta_z, zp,alpha_m, m_max,delta_m, m_min,lambda_peak, mu_m, sigma_m, beta_q = sample_joint_hyperparameters_separate(n_samples, seed=42)

    print("采样的超参数如下：")
    print("红移模型 (Madau–Dickinson)：")
    print("  alpha_z：红移上升的幂指数 (2.4-3)", alpha_z)
    print("  beta_z：红移下降的幂指数 (1.95-4.5)", beta_z)
    print("  zp：星形成率峰值红移 (1.4-2.5)", zp)
    # print("  r0：", r0)
    print("质量模型 (PowerLaw + Peak)：")
    print("  alpha_m：主星质量幂律指数 (-3-7)", alpha_m)
    print("  m_max：黑洞最大截断质量范围(10-100)",m_max)

    print("  delta_m：平滑宽度 (0-10 Msun)", delta_m)
    print("  m_min：黑洞最小截断质量范围(2-10)", m_min)

    print("  lambda_peak：高斯峰占比 (0.07-0.18)", lambda_peak)
    print("  mu_m：高斯峰均值 (33.2-35.4 Msun)", mu_m)
    print("  sigma_m：高斯峰标准差 (0.1-0.9 Msun)", sigma_m)
    print("  beta_q：质量比幂律指数 (-5-5)", beta_q)

    # 保存为 npz 文件
    save_path = "./joint_hyperparams_train/v2/joint_hyperparams.npz"
    np.savez(save_path,
             alpha_z=alpha_z, beta_z=beta_z, zp=zp,
             alpha_m=alpha_m, m_max=m_max,
             delta_m=delta_m,m_min=m_min,
             lambda_peak=lambda_peak, mu_m=mu_m, sigma_m=sigma_m,beta_q=beta_q)
    print(f"[成功] 超参数已保存到 '{save_path}'")
