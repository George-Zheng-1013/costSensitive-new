# Encryption-Traffic-Detection (Byte-Session Refactor)

本项目已完成底层重构：从旧的 `784->28x28` 图像式分类，切换为 **字节级包编码 + 会话级时序建模**。

## 新主干特性

- 输入：`[B, num_packets, packet_len]` 的字节张量（默认 `[B, 12, 256]`）
- 包掩码：`packet_mask [B, num_packets]`
- 模型：`PacketEncoder + SessionEncoder(BiGRU) + Embedding Head + Classifier`
- 推理：支持分类预测与 embedding 导出
- 实时：基于流组装会话后进行实时推理（pcap 模式）

## 关键文件

- `costSensitive/session_data.py`
  - 构建会话样本与 manifest
  - `ByteSessionDataset` 输出统一字典：
    - `bytes: Tensor[num_packets, packet_len]`
    - `packet_mask: Tensor[num_packets]`
    - `label: int`
    - `flow_id: str`
- `costSensitive/realtime/model_def.py`
  - `PacketEncoder`
  - `SessionEncoder`
  - `ByteSessionClassifier`
- `costSensitive/train_byte_session.py`
  - 新训练入口
- `costSensitive/inference.py`
  - 离线推理与 embedding 导出
- `costSensitive/main_realtime.py`
  - 实时 pcap 推理入口

## 快速开始

在仓库根目录下执行：

```powershell
cd costSensitive
```

### 1) 构建会话数据（可选单独执行）

```powershell
python session_data.py --dataset-root dataset/iscx --output-root processed_full/sessions --num-packets 12 --packet-len 256 --train-ratio 0.8 --flow-timeout-s 10 --seed 42
```

### 2) 训练新模型

```powershell
python train_byte_session.py --manifest processed_full/sessions/manifest.csv --epochs 8 --batch-size 32
```

若还未生成 manifest，可直接让训练脚本自动构建：

```powershell
python train_byte_session.py --rebuild-sessions --dataset-root dataset/iscx --session-root processed_full/sessions
```

### 3) 离线推理与 embedding 导出

```powershell
python inference.py --manifest processed_full/sessions/manifest.csv --model-path pytorch_model/byte_session_classifier.pth --split test
```

输出：

- `pytorch_model/session_predictions.csv`
- `pytorch_model/session_embeddings.npy`
- `pytorch_model/session_eval_report.json`

### 4) 实时 pcap 推理

```powershell
python main_realtime.py --mode pcap --source dataset/iscx --model-path pytorch_model/byte_session_classifier.pth --label-map processed_full/sessions/label_map.json --output-dir pytorch_model/realtime_sessions
```

输出：

- `pytorch_model/realtime_sessions/realtime_predictions.csv`
- `pytorch_model/realtime_sessions/realtime_predictions.jsonl`

## 说明

- 旧链路（图像化预处理、MNIST IDX、ConvNet 训练脚本）已移除。
- 当前实时入口支持 `pcap` 模式。

## 全链路启动（FastAPI + 入库引擎）

后端已支持一键联动：启动 FastAPI 时自动启动 `backend_engine` 写库线程。

```powershell
python run_full_chain.py --host 127.0.0.1 --port 8000 --reload
```

Windows 下也可直接执行：

```powershell
start_full_chain.bat
```

可选环境变量：

- `NETGUARD_AUTOSTART_ENGINE=0`：仅启 API，不自动拉起入库引擎。
- `NETGUARD_ENGINE_AUTOCAPTURE=1`：自动拉起实时抓包推理子进程。
- `NETGUARD_CAPTURE_SOURCE`：抓包网卡源。
- `NETGUARD_CAPTURE_DEVICE`：推理设备（`cpu`/`cuda`）。

前端开发模式：

```powershell
cd web-app
npm run dev
```
