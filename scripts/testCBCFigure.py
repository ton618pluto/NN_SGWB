from gwpy.timeseries import TimeSeries
import matplotlib.pyplot as plt
import os

# =====================
# 文件路径
# =====================
pop_num=28
frames_num=3
h1_file = f"./training_set/H1/H1-pop{pop_num:05d}_sample{frames_num:05d}.gwf"
l1_file = f"./training_set/L1/L1-pop{pop_num:05d}_sample{frames_num:05d}.gwf"

# =====================
# 通道名
# =====================
h1_channel = "H1:TEST_INJ"
l1_channel = "L1:TEST_INJ"

# =====================
# 读取数据
# =====================
h1_ts = TimeSeries.read(h1_file, h1_channel)
l1_ts = TimeSeries.read(l1_file, l1_channel)

# =====================
# 转 numpy
# =====================
h1_strain = h1_ts.value
l1_strain = l1_ts.value

times_h1 = h1_ts.times.value
times_l1 = l1_ts.times.value

# =====================
# 保存目录
# =====================
out_dir = "plots_SGWB"
os.makedirs(out_dir, exist_ok=True)

# =====================
# 画图并保存（不显示）
# =====================
plt.figure(figsize=(10, 4))
plt.plot(times_h1, h1_strain, label="H1")
plt.plot(times_l1, l1_strain, label="L1", alpha=0.7)

plt.xlabel("GPS time [s]")
plt.ylabel("Strain")
plt.legend()
plt.tight_layout()

out_path = os.path.join(out_dir, f"H1_L1_SGWB_strain-pop{pop_num:05d}_sample{frames_num:02d}.png")
plt.savefig(out_path, dpi=300)
plt.close()

print(f"图像已保存到: {out_path}")
