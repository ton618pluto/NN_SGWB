import sys, os

sys.path.append('./gwfast/')
import gwfast.gwfastGlobals as glob
import gwfast.gwfastUtils as gwutils
import gwfast.gwfastGlobals as gwglob
from gwfast.population.popdistributions import massdistribution, ratedistribution, spindistribution
from gwfast.population import POPmodels
import h5py

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import jax
import jax.numpy as jnp
from jax import jit
from jax.numpy import sqrt, pi
# from astropy.constants import pc, c
from astropy.cosmology import Planck18
from utils import *


# ------------------Defining redshift model----------------------
@jit
# 这里的函数主要用于描述不同距离下引力波源出现的密度分布
def psi_z(z, gamma, kappa, z_peak):
    """描述红移分布形状的函数"""
    const = 1 / (1 + (1 / (1 + z_peak)) ** kappa)
    return const * (1 + z) ** gamma / (1 + ((1 + z) / (1 + z_peak)) ** kappa)


@jit
def dN(z, gamma, kappa, z_peak, R_0, dVdz_grid):
    """计算单位红移内的事件数量：dN/dz"""
    return R_0 * psi_z(z, gamma, kappa, z_peak) * dVdz_grid * 4 * pi / (1 + z)


@partial(jit, static_argnums=(5,))  # static: z_grid, z_grid_n, dVdz_grid
def compute_redshift_prob_det(redshift_vals, gamma, kappa, z_peak, z_grid, z_grid_n, dVdz_grid):
    """计算红移的概率密度函数（PDF）并进行归一化"""
    z_pdf_vals = psi_z(z_grid, gamma, kappa, z_peak) * dVdz_grid / (1 + z_grid)
    normalization = jnp.trapezoid(z_pdf_vals, z_grid)
    return psi_z(redshift_vals, gamma, kappa, z_peak) * fast_interp_uniform(redshift_vals, 0.0, z_grid[-1], dVdz_grid, z_grid_n) / (1 + redshift_vals) / normalization


# Injection distribution, something flatish in masses, flat in spins, Madau-Dickinson in redshift but with high tail
# 这里是做一个比较平的分布用来做injections
# 1. 质量分布：使用 PowerLawPlusPeak 模型（幂律+峰值模型）
massDistr_inj = massdistribution.PowerLawPlusPeak_MassDistribution(
    # 这是PLPP model！
    # LIGO O3 的最好的 model 是 PLPP，目前根据 O4 最好的 model 是 BP2P model，GWFAST暂时应该还没有这个 model，不过可以自己写...
    hyperparameters={
        'alpha_m': 3.4,
        'beta_q': 1.08,
        'm_min': 5.08,
        'm_max': 86.85,
        'lambda_peak': 0.04,
        'delta_m': 4.83,
        'mu_m': 33.73,
        'sigma_m': 3.56
    },
    priorlims_parameters={'m1_scr': (5.08, 86.85), 'm2_src': (5.08, 86.85)}
)

# 2. 自旋分布：这里采用简单的非进动平展分布（简化模型）
spinDistr_inj = spindistribution.SameFlatNonPrecessing_SpinDistribution(
    # 我们没有必要去考虑 spin
    hyperparameters={'abs_chi': 0.99},
    priorlims_parameters={'chi1z': (-1., 1.), 'chi2z': (-1., 1.)}
)

# 3. 红移（距离）分布：使用 Madau-Dickinson 模型
redshiftDistr_inj = ratedistribution.MadauDickinson_RateDistribution(
    # # MD 分布的redshift
    hyperparameters={'alpha_z': 2.7, 'beta_z': 2.9, 'zp': 1.9},
    priorlims_parameters={'z': (0., 20.)}  # z_max 一般取到 10 就可以了
)

# 4. 组合成完整的群体模型：将质量、自旋、红移模型独立组合
fullPop_inj = POPmodels.MassSpinRedshiftIndependent_PopulationModel(
    mass_function=massDistr_inj,
    spin_function=spinDistr_inj,
    rate_function=redshiftDistr_inj,
)

np.random.seed(3333)
## Now let's generate the injections
# 这里虽然是固定了生成 N_inj 个样本，但是最好的做法是首先选定一个 population 模型，然后去考虑 T_obs的观测时间。
# 根据 popualtion 的具体参数以及 T_obs 去计算CBC总数的一个期望值。
# 然后去随机采样一个 Poisson 变量作为真正出现的Ninj: Ninj~Poisson(N|T_obs,Lambda)
z_grid_n = 2048
zmax = 20.0
# arrays for np
z_grid_np = np.linspace(0.0, zmax, z_grid_n, dtype=np.float32)  # np.array!
dVdz_grid_np = (Planck18.differential_comoving_volume(z_grid_np).value / 1e9)  # [Gpc^3 / sr / dz]
# Arrays for JAX
z_grid = jnp.asarray(z_grid_np, dtype=jnp.float32)
dVdz_grid = jnp.asarray(dVdz_grid_np, dtype=jnp.float32)
# 计算 N_tot
_, z_w = trapz_weights(z_grid_n, 0.0, zmax)
N_tot = jnp.dot(dN(z_grid, 2.7, 2.7 + 2.9, 1.9, 320, dVdz_grid), z_w)  # 假设观测时长为 1 年，期望出现的 CBC 的数量
Ninj = np.random.poisson(N_tot)
injpars = fullPop_inj.sample_population(Ninj)
print(f"一共有 {Ninj} 个 CBC 事件！")

dL = np.asarray(Planck18.luminosity_distance(injpars['z']))  # 单位是Mpc,百万秒差距，1Mpc=326万光年

T = 365.25 * 24 * 3600
tc_unsorted = np.random.uniform(0, T, size=Ninj)
tc = np.sort(tc_unsorted)
ra = np.random.uniform(0, 2 * pi, Ninj)  # 随机赤经
psi = np.random.uniform(0, 2 * pi, Ninj)  # 随机偏振角
phi_c = np.random.uniform(0, 2 * pi, Ninj)  # 随机并合相位
u = np.random.uniform(-1, 1, Ninj)  # 用于均匀球面上采样倾角
v = np.random.uniform(-1, 1, Ninj)  # 用于均匀球面上采样赤纬
iota = np.arccos(u)  # 轨道倾角
dec = np.arccos(v)  # 赤纬

np.savez("CBC_params_example18", tc=tc,
         m1=injpars['m1'],
         m2=injpars['m2'],
         dL=dL,
         z=injpars['z'],
         ra=ra,
         dec=dec,
         psi=psi,
         phi_c=phi_c,
         iota=iota)
