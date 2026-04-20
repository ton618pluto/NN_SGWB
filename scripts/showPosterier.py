import numpy as np
import corner
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'，取决于你安装了哪个

# 1. 模拟模型输出的后验采样点 (Posterior Samples)
# 在实际应用中，这些 samples 是由你的神经网络（如 Normalizing Flow）生成的
# 这里我们用高斯分布模拟一组已经“收敛”的采样结果
n_samples = 10000
labels = [r"$\alpha_m$", r"$\mu_m$", r"$z_p$", r"$\lambda_{peak}$"]

# 假设真实值（Truths）就是你代码里的 O3 参数
truths = [3.4, 33.73, 1.9, 0.04]

# 生成一些随机采样数据，中心在真实值附近
# 模拟神经网络推断出的不确定性
samples = np.random.multivariate_normal(
    mean=truths,
    cov=np.diag([0.2, 1.5, 0.1, 0.01])**2, # 模拟每个参数的推断标准差
    size=n_samples
)

print(f'真实值：{truths}')
print(f'神经网络的输出：{samples}')
# 2. 绘制 Corner Plot
fig = corner.corner(
    samples,
    labels=labels,
    truths=truths,
    truth_color="#1f77b4",
    show_titles=True,
    title_fmt=".2f",
    # 显式添加下面这一行，提供三个分位数值
    title_quantiles=[0.16, 0.5, 0.84],
    # 绘图使用的分位数通常保持一致
    quantiles=[0.16, 0.5, 0.84],
    smooth=1.0,
    fill_contours=True,
    plot_datapoints=False
)

# 3. 设置标题并显示
fig.suptitle("CBC-SGWB Hyperparameter Posterior Inference", fontsize=16)
plt.show()