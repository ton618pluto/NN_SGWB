import pandas as pd
from sklearn.datasets import fetch_california_housing


def explore_regression_dataset():
    # 1. 加载加州住房数据集
    # as_frame=True 会直接返回 pandas DataFrame 格式，方便查看
    data = fetch_california_housing(as_frame=True)

    df = data.frame

    print("--- 数据集概览 ---")
    print(f"特征名称: {data.feature_names}")
    print(f"目标变量 (Target): {data.target_names}")
    print(f"数据量 (行, 列): {df.shape}")

    print("\n--- 前 5 行数据预览 ---")
    # MedHouseVal 是我们要预测的目标值，单位为 100,000 美元
    print(df.head())

    print("\n--- 统计摘要 ---")
    print(df.describe().loc[['mean', 'std', 'min', 'max']])


if __name__ == "__main__":
    explore_regression_dataset()