"""Generate synthetic test data mimicking the 700.xlsx format for testing."""
import pandas as pd
import numpy as np

np.random.seed(42)
N = 200

data = pd.DataFrame(np.nan, index=range(N + 2), columns=range(80))

data.iloc[0, 0] = "随机号"
data.iloc[0, 4] = "年龄"
data.iloc[0, 6] = "体重"
data.iloc[0, 9] = "甲状腺重"
data.iloc[0, 10] = "超声"
data.iloc[0, 14] = "疗效"
data.iloc[0, 16] = "FT3_0"
data.iloc[0, 17] = "FT4_0"
data.iloc[0, 18] = "TSH_0"
data.iloc[0, 21] = "TRAb"
data.iloc[0, 22] = "24h吸碘"
data.iloc[0, 25] = "剂量"
data.iloc[1, :] = "单位行"

for i in range(N):
    row = i + 2
    data.iloc[row, 0] = i + 1
    data.iloc[row, 1] = f"H{i+1:04d}"
    data.iloc[row, 2] = f"患者{i+1}"
    data.iloc[row, 3] = np.random.choice(["男", "女"])
    data.iloc[row, 4] = np.random.randint(20, 70)
    data.iloc[row, 5] = np.random.randint(150, 185)
    weight = np.random.randint(45, 95)
    data.iloc[row, 6] = weight
    height = float(data.iloc[row, 5])
    data.iloc[row, 7] = round(weight / ((height / 100) ** 2), 1)
    data.iloc[row, 8] = np.random.choice([0, 1, 2])
    thyroid_w = round(np.random.uniform(15, 80), 1)
    data.iloc[row, 9] = thyroid_w
    data.iloc[row, 10] = np.random.choice(["未见", "增多", "丰富", "极丰富", "略增多"])
    data.iloc[row, 11] = round(np.random.uniform(1, 10), 1)
    data.iloc[row, 12] = np.random.choice([1, 2, 3])
    data.iloc[row, 14] = np.random.choice([1, 2, 3], p=[0.4, 0.25, 0.35])

    ft3_base = round(np.random.uniform(3, 30), 2)
    ft4_base = round(np.random.uniform(10, 50), 2)
    tsh_base = round(np.random.uniform(0.01, 5), 3)
    data.iloc[row, 16] = ft3_base
    data.iloc[row, 17] = ft4_base
    data.iloc[row, 18] = tsh_base

    data.iloc[row, 19] = round(np.random.uniform(0, 500), 1)
    data.iloc[row, 20] = round(np.random.uniform(0, 300), 1)
    trab = round(np.random.uniform(0.5, 40), 2)
    data.iloc[row, 21] = trab
    data.iloc[row, 22] = round(np.random.uniform(20, 80), 1)
    data.iloc[row, 23] = round(np.random.uniform(30, 90), 1)
    data.iloc[row, 24] = round(np.random.uniform(2, 8), 1)
    data.iloc[row, 25] = round(np.random.uniform(3, 15), 1)

    time_offsets = [29, 30, 31, 38, 39, 40, 47, 48, 49, 56, 57, 58, 65, 66, 67, 74, 75, 76]
    for j, col in enumerate(time_offsets):
        base_val = [ft3_base, ft4_base, tsh_base][j % 3]
        noise = np.random.uniform(0.7, 1.3)
        data.iloc[row, col] = round(base_val * noise, 2)

data.to_excel("/workspace/700.xlsx", index=False, header=False, engine="openpyxl")
print(f"Generated synthetic 700.xlsx with {N} patient records")
