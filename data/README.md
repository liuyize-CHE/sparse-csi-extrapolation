# 初始空间 CSI 数据集

该目录包含为任务 D 生成的小规模合成复数 CSI 数据集。数据由
`experiments/generate_dataset.py` 完整生成，无需下载外部原始数据。

## 数据规模

- 传播场景：10 个
- 空间区域：4 m × 4 m
- 空间网格：101 × 101，间距 0.04 m
- 子载波：8 个
- 中心频率：3 GHz
- 总带宽：20 MHz
- 每个场景：1 条直达径 + 3 条反射径
- CSI 类型：`complex64`
- CSI 张量形状：`(10, 8, 101, 101)`

`0.04 m` 的空间间距小于最高频率对应的半波长 `0.04983 m`，通过了空间
Nyquist 检查，避免先前 `64 × 64` 网格造成的空间相位混叠。

## 文件说明

- `processed/initial_csi_dataset.npz`：完整数据数组。
- `processed/initial_csi_dataset_metadata.json`：配置、统计量、划分与 SHA256。
- `figures/initial_dataset_scene0_overview.png`：场景 0 的幅度、相位和几何结构。
- `figures/initial_dataset_subcarrier_diversity.png`：不同子载波的幅度地图。

## NPZ 中的数组

| 数组 | 形状 | 含义 |
|---|---:|---|
| `csi` | `(10, 8, 101, 101)` | 多场景、多子载波复数 CSI |
| `x_m` | `(101,)` | x 方向坐标，单位 m |
| `y_m` | `(101,)` | y 方向坐标，单位 m |
| `frequencies_hz` | `(8,)` | 子载波频率，单位 Hz |
| `tx_positions_m` | `(10, 2)` | 每个场景的 Tx 坐标 |
| `scatterer_positions_m` | `(10, 3, 2)` | 三个反射点坐标 |
| `reflection_coefficients` | `(10, 3)` | 复数反射系数 |
| `path_loss_exponents` | `(10,)` | 每个场景的路径损耗指数 |
| `valid_masks` | `(10, 101, 101)` | 排除 Tx 近场后的有效位置 |
| `train_scene_indices` | `(6,)` | 训练场景编号 |
| `validation_scene_indices` | `(2,)` | 验证场景编号 |
| `test_scene_indices` | `(2,)` | 测试场景编号 |

## 读取示例

```python
import numpy as np

data = np.load("data/processed/initial_csi_dataset.npz")

csi = data["csi"]
train_scenes = data["train_scene_indices"]

print(csi.shape)       # (10, 8, 101, 101)
print(csi.dtype)       # complex64
print(train_scenes)    # [0 2 3 4 5 9]

# 场景 0、子载波 0 的完整复数 CSI 地图
h = csi[0, 0]
amplitude = np.abs(h)
phase = np.angle(h)
```

## 重新生成

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe .\experiments\generate_dataset.py
```

固定随机种子为 2026。当前数据文件的 SHA256 为：

```text
aa195b3b9b4437ed642a4674c9a96089f36ae28b0416aa24cc8ec9f59f477939
```

生成脚本会自动检查：

1. 张量形状和复数类型；
2. 是否存在 NaN 或无穷大；
3. 训练、验证、测试场景是否互斥且完整；
4. 不同场景是否确实不同；
5. 相邻子载波 CSI 是否确实不同；
6. 空间网格是否满足最高载波频率的半波长采样条件。
