import numpy as np

# 文件路径
file_path = "./CBC_params_example.npz"
# file_path = "./CBC_params_example00002.npz"

# 字典，用于存储键名及其对应的中文名称
parameter_names = {
    'tc': '并合时间 (Coalescence Time)',
    'm1': '主质量 (Primary Mass)',
    'm2': '次质量 (Secondary Mass)',
    'dL': '光度距离 (Luminosity Distance)',
    'z': '红移 (Redshift)',
    'ra': '赤经 (Right Ascension)',
    'dec': '赤纬 (Declination)',
    'psi': '偏振角 (Polarization Angle)',
    'phi_c': '并合相位 (Coalescence Phase)',
    'iota': '轨道倾角 (Inclination Angle)',
}


# 加载 .npz 文件
list_cbc = np.load(file_path)

print("--- 🌌 CBC 参数文件内容摘要 ---")

# 获取文件中所有的键名
keys_in_file = list_cbc.files

for key in keys_in_file:
    # 尝试获取中文名称，如果找不到则显示英文键名
    chinese_name = parameter_names.get(key, f"未知参数 ({key})")

    # 打印键名、中文名称以及对应的数组形状
    data_shape = list_cbc[key].shape
    print(f"✅ {chinese_name}： 数组形状 (Shape): {data_shape} 数组详情：{list_cbc[key]}")

print("--------------------------------")

# 定义当前任务编号
jobNumber=15409
# 定义数据段的持续时间（秒）
duration = 2048
# 计算当前时间段的起始 GPS 时间
st = (jobNumber-1)*duration
# 计算当前时间段的结束 GPS 时间
et = st + duration
# 定义最小分析频率（Hz）
minimum_frequency = 10

def estimation_t0(tc, m1, m2):
    """
    使用牛顿近似法估算啁啾信号频率等于 f0 时的时刻。
    """
    # 太阳质量转换为秒的常数 G*M_sun/c^3
    MTSUN_SI = 4.925491025543576e-06
    # 定义估算时的起始频率 f0（略低于最小分析频率）
    f0 = minimum_frequency -1
    # 计算啁啾质量（Chirp Mass）：mc = (m1*m2)^(3/5) / (m1+m2)^(1/5)
    mc = (m1*m2)**(3./5)/(m1+m2)**(1./5)
    # 牛顿近似公式中的常数部分
    C = 5**(3./8) / (8*np.pi)
    # 根据牛顿近似公式，估算啁啾频率达到 f0 时的信号起始时刻 t0
    real_t0 = tc-(f0/C)**(-8./3)*(mc*MTSUN_SI)**(-5./3)
    return real_t0

# 从数据集中加载并合时间
tc = list_cbc['tc']
m1 = list_cbc['m1']
m2 = list_cbc['m2']
z  = list_cbc['z']

# 将源系质量转换为探测器系质量：m_det = m_src * (1+z)
m1, m2 = m1*(1+z), m2*(1+z)
# 估算每个事件的信号起始时间 t0
t0 = estimation_t0(tc, m1, m2)

# 筛选出哪些事件落在了当前任务的时间段内！
# 筛选条件：
# 1. 并合时间 (tc) 发生在当前时间段起始时间 (st) 之后
# 2. 信号起始时间 (t0) 发生在当前时间段结束时间 (et) 之前
list_cbc_mask = (tc > st) * (t0 < et)

# 打印布尔掩码的长度，即总事件数
print(len(list_cbc_mask))
# 统计 True 的个数（符合时间条件的事件）
num_true = np.sum(list_cbc_mask)
print('true个数：',num_true)

# 统计 False 的个数（不符合时间条件的事件）
num_false = np.sum(~list_cbc_mask)
print('false个数：',num_false)

# 打印布尔掩码数组，其中 True 表示该事件符合筛选条件
print(list_cbc_mask)
