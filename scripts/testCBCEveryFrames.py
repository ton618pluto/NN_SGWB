import numpy as np
import os

duration=2048
jobNumber=1
minimum_frequency = 10
def estimation_t0(tc, m1, m2):
    MTSUN_SI = 4.925491025543576e-06
    f0 = minimum_frequency - 1
    mc = (m1 * m2) ** (3. / 5) / (m1 + m2) ** (1. / 5)
    C = 5 ** (3. / 8) / (8 * np.pi)
    real_t0 = tc - (f0 / C) ** (-8. / 3) * (mc * MTSUN_SI) ** (-5. / 3)
    return real_t0

st = (jobNumber - 1) * duration  # 开始时间
et = st + duration               # 结束时间 (duration=2048)

param_file = './parameter_sampling_val/CBC_params_example00000.npz'
#param_file = './parameter_sampling_train/CBC_params_example00000.npz'
list_cbc = np.load(param_file)
tc = list_cbc['tc']
t0=estimation_t0(tc,list_cbc['m1'],list_cbc['m2'])
# 这行代码筛选出了落在 [st, et] 范围内的所有事件索引
list_cbc_mask = (tc > st) * (t0 < et)
# 提取所有符合掩码条件的属性
tc_all = list_cbc['tc'][list_cbc_mask]
# ... 提取其他参数 ...

# 计算事件总数
num_events = len(tc_all)

# 打印结果
print(f"📂 找到 {num_events} 个事件")
