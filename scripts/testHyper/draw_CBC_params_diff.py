import sys, os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from astropy.cosmology import Planck18
from tqdm import tqdm  # 推荐加上，方便观察进度
from jax.numpy import sqrt, pi

# 假设环境已配置好 gwfast
sys.path.append('./gwfast/')
from gwfast.population.popdistributions import massdistribution, ratedistribution, spindistribution
from gwfast.population import POPmodels
from utils import *


# ------------------核心物理函数----------------------
@jit
def psi_z(z, gamma, kappa, z_peak):
    const = 1 / (1 + (1 / (1 + z_peak)) ** kappa)
    return const * (1 + z) ** gamma / (1 + ((1 + z) / (1 + z_peak)) ** kappa)


@jit
def dN(z, gamma, kappa, z_peak, R_0, dVdz_grid):
    return R_0 * psi_z(z, gamma, kappa, z_peak) * dVdz_grid * 4 * pi / (1 + z)


@partial(jit, static_argnums=(5,))  # static: z_grid, z_grid_n, dVdz_grid
def compute_redshift_prob_det(redshift_vals, gamma, kappa, z_peak, z_grid, z_grid_n, dVdz_grid):
    """计算红移的概率密度函数（PDF）并进行归一化"""
    z_pdf_vals = psi_z(z_grid, gamma, kappa, z_peak) * dVdz_grid / (1 + z_grid)
    normalization = jnp.trapezoid(z_pdf_vals, z_grid)
    return psi_z(redshift_vals, gamma, kappa, z_peak) * fast_interp_uniform(redshift_vals, 0.0, z_grid[-1], dVdz_grid, z_grid_n) / (1 + redshift_vals) / normalization


# ------------------封装采样函数----------------------
def run_parameter_sampling(
        output_dir="./parameter_sampling",
        sample_id=0,
        T_obs_month=12,
        # 红移分布超参
        R0=320.0,
        alpha_z=2.7,
        beta_z=2.9,
        zp=1.9,
        # 质量分布超参
        alpha_m=3.4,
        m_max=86.85,

        delta_m=4.83,
        m_min=5.08,

        lambda_peak=0.04,
        mu_m=33.73,  # 峰值位置建议也作为变量或固定
        sigma_m=3.56,
        beta_q=1.08,
        seed=3333
):
    if seed is not None:
        np.random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 宇宙学背景计算
    z_grid_n = 2048
    zmax = 20.0
    z_grid_np = np.linspace(0.0, zmax, z_grid_n, dtype=np.float32)
    dVdz_grid_np = (Planck18.differential_comoving_volume(z_grid_np).value / 1e9)
    z_grid = jnp.asarray(z_grid_np)
    dVdz_grid = jnp.asarray(dVdz_grid_np)

    # 2. 计算 4 个月内的期望事件数
    T_obs_year = T_obs_month / 12.0
    _, z_w = trapz_weights(z_grid_n, 0.0, zmax)
    kappa = alpha_z + beta_z
    expected_N = jnp.dot(dN(z_grid, alpha_z, kappa, zp, R0, dVdz_grid), z_w) * T_obs_year
    Ninj = np.random.poisson(float(expected_N))

    if Ninj == 0:
        return 0

    # 3. 构建群体模型 (使用传入的动态超参)
    mass_dist = massdistribution.PowerLawPlusPeak_MassDistribution(
        hyperparameters={
            'alpha_m': alpha_m,
            'beta_q': beta_q,
            'm_min': m_min,
            'm_max': m_max,
            'lambda_peak': lambda_peak,
            'delta_m': delta_m,
            'mu_m': mu_m,
            'sigma_m': sigma_m
        },
        priorlims_parameters={'m1_src': (m_min, m_max), 'm2_src': (m_min, m_max)}
    )

    spin_dist = spindistribution.SameFlatNonPrecessing_SpinDistribution(
        hyperparameters={'abs_chi': 0.99},
        priorlims_parameters={'chi1z': (-1., 1.), 'chi2z': (-1., 1.)}
    )

    redshift_dist = ratedistribution.MadauDickinson_RateDistribution(
        hyperparameters={'alpha_z': alpha_z, 'beta_z': beta_z, 'zp': zp},
        priorlims_parameters={'z': (0., 20.)}
    )

    full_pop = POPmodels.MassSpinRedshiftIndependent_PopulationModel(
        mass_function=mass_dist,
        spin_function=spin_dist,
        rate_function=redshift_dist,
    )

    # 采样源参数
    injpars = full_pop.sample_population(Ninj)

    # 4. 生成辅助观测参数
    # T_seconds = T_obs_year * 365.25 * 24 * 3600
    T_seconds = T_obs_year * 365.25 * 24 * 3600
    tc = np.sort(np.random.uniform(0, T_seconds, size=Ninj))
    ra = np.random.uniform(0, 2 * np.pi, Ninj)
    psi = np.random.uniform(0, 2 * np.pi, Ninj)
    phi_c = np.random.uniform(0, 2 * np.pi, Ninj)
    iota = np.arccos(np.random.uniform(-1, 1, Ninj))
    dec = np.arcsin(np.random.uniform(-1, 1, Ninj))
    dL = np.asarray(Planck18.luminosity_distance(injpars['z']))

    # 5. 保存结果与对应的真值标签
    filename = os.path.join(output_dir, f"CBC_params_example{sample_id:05d}.npz")
    np.savez(filename,
             tc=tc, m1=injpars['m1'], m2=injpars['m2'],
             dL=dL, z=injpars['z'], ra=ra, dec=dec,
             psi=psi, phi_c=phi_c, iota=iota,
             # 神经网络训练所需的 Labels
             true_R0=R0,
             true_alpha_z=alpha_z,
             true_beta_z=beta_z,
             true_zp=zp,

             true_alpha_m=alpha_m,
             true_m_max=m_max,

             true_delta_m=delta_m,
             true_m_min=m_min,

             true_lambda_peak=lambda_peak,
             true_mu_m=mu_m,
             true_sigma_m=sigma_m,
             true_beta_q=beta_q
             )

    return Ninj


# ---------------------------------------------------------
# 批量循环采样
# ---------------------------------------------------------


output_path = "./parameter_sampling_train"
# output_path = "./parameter_sampling_val"
# output_path = "./parameter_sampling_test"
dataHyperPara = np.load('./joint_hyperparams_train/joint_hyperparams_train.npz')
keys = dataHyperPara.files
num_datasets = len(dataHyperPara[keys[0]])

for i in range(0,1):
    np.random.seed(i)
    # --- 随机生成超参数 (根据 LIGO O3/O4 常见的先验范围设定) ---
    # 红移相关
    # r_R0 = np.random.uniform(250, 370)
    # r_R0 = dataHyperPara['r0'][i]
    # r_zp = np.random.uniform(1.8, 2.4)
    r_zp = dataHyperPara['zp'][i]
    # r_alpha_z = np.random.uniform(2.3, 2.8)
    r_alpha_z = dataHyperPara['alpha_z'][i]
    # r_beta_z = np.random.uniform(2.5, 3.2)
    r_beta_z = dataHyperPara['beta_z'][i]

    # 质量相关
    # r_alpha_m = np.random.uniform(3.0, 4.0)
    r_alpha_m = dataHyperPara['alpha_m'][i]
    r_m_max = dataHyperPara['m_max'][i]

    # r_delta_m = np.random.uniform(3.0, 6.0)
    r_delta_m = dataHyperPara['delta_m'][i]
    r_m_min = dataHyperPara['m_min'][i]

    r_lambda_peak = dataHyperPara['lambda_peak'][i]
    r_mu_m = dataHyperPara['mu_m'][i]
    # r_sigma_m = np.random.uniform(2.0, 5.0)
    r_sigma_m = dataHyperPara['sigma_m'][i]
    r_beta_q = dataHyperPara['beta_q'][i]

    # r_lambda_peak = np.random.uniform(0.02, 0.08)

    print(f"第{i}组数据的超参数如下：")
    print("  alpha_z：红移上升的幂指数 (2.1-3.6)", r_alpha_z)
    print("  beta_z：红移下降的幂指数 (由alpha_z和zp计算)", r_beta_z)
    print("  zp：星形成率峰值红移 (1.4-2.4)", r_zp)
    # print("  r0：", r0)
    print("质量模型 (PowerLaw + Peak)：")
    print("  alpha_m：主星质量幂律指数 (-3-7)", r_alpha_m)
    print("  m_max：黑洞最大截断质量范围(10-100)", r_m_max)

    print("  delta_m：平滑宽度 (0-10 Msun)", r_delta_m)
    print("  m_min：黑洞最小截断质量范围(2-10)", r_m_min)

    print("  lambda_peak：高斯峰占比 (0.07-0.18)", r_lambda_peak)
    print("  mu_m：高斯峰均值 (33.2-35.4 Msun)", r_mu_m)
    print("  sigma_m：高斯峰标准差 (0.1-0.9 Msun)", r_sigma_m)
    print("  beta_q：质量比幂律指数 (-5-5)", r_beta_q)

    num_events = run_parameter_sampling(
        sample_id=i,
        output_dir=output_path,
        # T_obs_month=1,
        T_obs_month=1/30,
        zp=r_zp,
        alpha_z=r_alpha_z,
        beta_z=r_beta_z,
        alpha_m=r_alpha_m,
        m_max=r_m_max,
        delta_m=r_delta_m,
        m_min=r_m_min,
        lambda_peak=r_lambda_peak,
        mu_m=r_mu_m,
        sigma_m=r_sigma_m,
        beta_q=r_beta_q,
        seed=i
    )
    print(f"完成！生成事件总数: {num_events}")
    print()
