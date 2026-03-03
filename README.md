# costSensitive-new：从数据集获取到推理测试的完整运行流程

本文档给出本项目的**端到端运行步骤**：

1. 获取/放置原始 ISCX 数据集（pcap/pcapng）
2. 预处理为 PNG + IDX(MNIST 格式)
3. 训练 CNN 模型
4. 推理并导出预测结果

---

## 0. 项目目录约定

以下说明默认你在仓库根目录：

`costSensitive-new/`

训练与推理脚本位于：

`costSensitive/`

建议所有命令在该目录下执行：

```powershell
cd costSensitive
```

---

## 1. 环境准备

## 1.1 Python 版本

推荐 Python 3.9 ~ 3.11（Windows）。

## 1.2 创建虚拟环境（可选但推荐）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

## 1.3 安装依赖

> 训练脚本里导入了 `cv2`、`matplotlib`，即使未显式使用也需要安装，避免启动时报 ImportError。

```powershell
pip install numpy pillow dpkt torch torchvision opencv-python matplotlib
```

如果你使用 CUDA 版 PyTorch，请按官方命令替换 `torch/torchvision` 安装方式。

---

## 2. 获取并放置数据集

本项目使用 ISCX VPN / NonVPN 流量数据（pcap 或 pcapng）。

将原始文件放到：

`costSensitive/dataset/iscx/`

并保持如下子目录前缀（脚本靠前缀识别类别域）：

- `NonVPN-*`（如 `NonVPN-PCAPs-01`）
- `VPN-*`（如 `VPN-PCAPS-01` / `VPN-PCAPs-02`）

你的工作区已经包含示例结构，可直接复用。

---

## 3. 预处理（pcap -> PNG -> IDX）

在 `costSensitive/` 目录执行：

```powershell
python preprocess_full.py --dataset-root dataset/iscx --output-root processed_full --train-ratio 0.8 --seed 42 --max-per-class 10000 --clean
```

参数说明：

- `--dataset-root`：原始 pcap 数据根目录
- `--output-root`：输出目录（默认 `processed_full`）
- `--train-ratio`：训练集比例
- `--max-per-class`：每类最多抽取样本数
- `--clean`：预处理前删除旧输出

预处理成功后会生成：

- `processed_full/png_all/`：按类别汇总的中间 PNG
- `processed_full/png/train|test/`：训练/测试拆分后的 PNG
- `processed_full/mnist/`：
	- `train-images-idx3-ubyte(.gz)`
	- `train-labels-idx1-ubyte(.gz)`
	- `t10k-images-idx3-ubyte(.gz)`
	- `t10k-labels-idx1-ubyte(.gz)`
- `processed_full/reports/`：统计报表（类分布、失败文件、summary 等）

---

## 4. 训练模型

在 `costSensitive/` 目录执行：

```powershell
python RSG_pytroch_CNN784.py
```

训练脚本默认：

- 自动使用 `cuda`（可用时）否则 `cpu`
- 训练 6 个 epoch
- 读取 `processed_full/mnist/*.gz`

训练输出：

- `pytorch_model/convnet.pth`：模型权重
- `pytorch_model/train_log.csv`：每 epoch 日志（loss/acc/time）

---

## 5. 推理测试

在 `costSensitive/` 目录执行：

```powershell
python inference.py
```

推理脚本会：

1. 加载 `pytorch_model/convnet.pth`
2. 在 `processed_full/mnist/t10k-*` 测试集上推理
3. 输出整体准确率
4. 保存逐样本结果到 `pytorch_model/predictions.csv`

`predictions.csv` 列含义：

- `index`：样本序号
- `pred`：预测类别 ID
- `label`：真实类别 ID
- `confidence`：最大 softmax 置信度

---

## 6. 一键复现实验顺序（最常用）

```powershell
cd costSensitive
python preprocess_full.py --dataset-root dataset/iscx --output-root processed_full --train-ratio 0.8 --seed 42 --max-per-class 10000 --clean
python RSG_pytroch_CNN784.py
python inference.py
```

---

## 7. 常见问题排查

## 7.1 `模型文件不存在: pytorch_model/convnet.pth`

先执行训练：

```powershell
python RSG_pytroch_CNN784.py
```

## 7.2 预处理后样本数很少或为 0

- 检查 `dataset/iscx` 下是否确实有 `.pcap/.pcapng`
- 检查文件名前缀是否符合脚本内置映射规则（如 `facebook_chat`、`email`、`bittorrent`、`vpn_...` 等）
- 查看 `processed_full/reports/failed_files.csv`

## 7.3 显存不足 / 训练慢

- 降低 batch（需改 `RSG_pytroch_CNN784.py` 中 `batch_size`）
- 使用 CPU 可运行但更慢

---

## 8. 已有处理结果的快速验证（可选）

若仓库内已存在 `processed_full/` 与 `pytorch_model/convnet.pth`，可直接跳过预处理与训练，只做推理：

```powershell
cd costSensitive
python inference.py
```

