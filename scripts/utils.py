import numpy as np
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit


jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", False) ## THIS IS A MUST

# ---------- 线性等间距插值 (比 jnp.interp 快，而且能处理边界) ----------
@partial(jax.jit, static_argnums=(4,))  # n 是静态
def fast_interp_uniform(x, xmin, xmax, fp, n):
    # fp 形状为 (n,) ，定义在等距网格 xmin..xmax
    # 超出范围返回 0
    # 映射到索引空间
    t = (x - xmin) * (n - 1) / (xmax - xmin)
    i0 = jnp.floor(t).astype(jnp.int32)
    i1 = jnp.clip(i0 + 1, 0, n - 1)
    w  = jnp.clip(t - i0, 0.0, 1.0)
    i0 = jnp.clip(i0, 0, n - 1)
    y  = (1.0 - w) * fp[i0] + w * fp[i1]
    # 区外设为 0
    y = jnp.where((x < xmin) | (x > xmax), 0.0, y)
    return y

# ---------- 预计算梯形权重（等距 z 网格） ----------
def trapz_weights(n, a, b):
    dz = (b - a) / (n - 1)
    w = jnp.ones((n,)) * dz
    w = w.at[0].set(dz / 2).at[-1].set(dz / 2)
    z = jnp.linspace(a, b, n)
    return z, w



EPS = 1e-12

def inverse_cdf_1d_np(u, cdf, grid):
    u = np.clip(u, 0.0, 1.0 - EPS)
    # 反演：interp 的 xp= cdf, fp= grid
    return np.interp(u, cdf, grid)

def inverse_cdf_rows_np(u, cdf_rows, x_grid):
    N, Q = cdf_rows.shape
    idx = np.sum(cdf_rows < u[:, None], axis=1)
    idx = np.clip(idx, 1, Q - 1)
    i0  = idx - 1; i1 = idx

    c0 = cdf_rows[np.arange(N), i0]
    c1 = cdf_rows[np.arange(N), i1]
    x0 = x_grid[i0]
    x1 = x_grid[i1]

    t  = (u - c0) / (c1 - c0 + EPS)
    return x0 + t*(x1 - x0)

def sample_events_np(N, rng,
                     m1_grid, cdf_m1,
                     q_grid,  cdf_q,     # (M1,Q)
                     z_grid,  cdf_z):
    u1 = rng.random(N)
    m1 = inverse_cdf_1d_np(u1, cdf_m1, m1_grid)

    idx_m1 = np.searchsorted(m1_grid, m1, side='right')
    idx_m1 = np.clip(idx_m1, 1, len(m1_grid) - 1)
    i0 = idx_m1 - 1
    t  = (m1 - m1_grid[i0]) / (m1_grid[i0+1] - m1_grid[i0] + EPS)
    cdf_rows = (1.0 - t)[:, None] * cdf_q[i0] + t[:, None] * cdf_q[i0+1]

    u2 = rng.random(N)
    q  = inverse_cdf_rows_np(u2, cdf_rows, q_grid)

    u3 = rng.random(N)
    z  = inverse_cdf_1d_np(u3, cdf_z, z_grid)

    m2 = q*m1
    Theta = np.random.beta(2,4, size=N)
    return m1, m2, q, z, Theta