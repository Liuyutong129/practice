import os
import numpy as np
import pandas as pd

np.random.seed(42)

n = 300
t = np.arange(n)

# 正常数据
x1 = np.sin(t / 20) + 0.05 * np.random.randn(n)
x2 = np.cos(t / 25) + 0.05 * np.random.randn(n)
x3 = 0.3 * np.random.randn(n)
x4 = 0.5 * x1 - 0.2 * x2 + 0.05 * np.random.randn(n)

# 注入异常：180~220 区间
x1[180:220] += 1.2
x3[180:220] += np.linspace(0, 1.5, 40)

df = pd.DataFrame({
    "time": t,
    "sensor_1": x1,
    "sensor_2": x2,
    "sensor_3": x3,
    "sensor_4": x4
})

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/demo_timeseries.csv", index=False, encoding="utf-8-sig")

print("Demo data saved to: data/raw/demo_timeseries.csv")
print(df.head())