# Sparse CSI Extrapolation

一个基于稀疏空间测量恢复多子载波复数 CSI 的实验项目，包含合成数据生成、复数线性插值、Fourier MLP 和 Tx-aware Fourier MLP。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
python -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux/macOS
source .venv/bin/activate
```

然后安装依赖：

```bash
python -m pip install -e .
```

## 运行方法

### 1. 生成数据集

```bash
python experiments/generate_dataset.py
```

### 2. 运行复数线性插值基线

```bash
python experiments/run_baseline.py --self-test
python experiments/run_baseline.py --run-experiment
```

### 3. 运行普通 Fourier MLP

```bash
python experiments/run_fourier_mlp.py --self-test --device auto
python experiments/run_fourier_mlp.py --tune-validation --device auto
python experiments/run_fourier_mlp.py --run-test --device auto
```

### 4. 运行 Tx-aware Fourier MLP

```bash
python experiments/run_tx_aware_mlp.py --self-test --device auto
python experiments/run_tx_aware_mlp.py --tune-validation --device auto
python experiments/run_tx_aware_mlp.py --run-test --device auto
```

没有 CUDA 时可将 `--device auto` 改为 `--device cpu`。生成的数据位于 `data/`，实验结果位于 `results/`。
