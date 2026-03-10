# Encryption-Traffic-Detection

本项目面向加密流量检测与研判，主链路已从旧图像化方案切换为 `字节级包编码 + 会话级时序建模`，并打通了 `实时推理 -> 入库 -> FastAPI -> Vue 可视化`。

## 当前能力概览

- 模型输入：`[B, num_packets, packet_len]`（默认 `[B, 12, 256]`）
- 编码结构：`PacketEncoder + SessionEncoder(BiGRU) + Embedding + Classifier`
- Unknown 检测：基于 centroid distance 的多级状态（known/suspected/unknown）
- 实时链路：抓包推理结果写入 JSONL，再由 `backend_engine` 入库 SQLite
- 分析接口：提供概览、告警、地理热力图、Unknown Cluster、XAI、AI 洞察
- 前端看板：Overview / Alerts / XAI / AI Insights 四个页面

## 项目结构（核心目录）

- `costSensitive/`
  - 训练、推理、实时抓包、未知类聚类等主逻辑
- `services/`
  - `api_server.py` FastAPI 服务
  - `backend_engine.py` JSONL -> SQLite 入库引擎
- `web-app/`
  - Vue3 + Vite 前端
- `项目概要/`
  - Notebook 说明与全流程记录

## 关键脚本

- `costSensitive/session_data.py`：构建会话级样本和 manifest
- `costSensitive/train_byte_session.py`：训练字节会话模型
- `costSensitive/inference.py`：离线推理 + embedding 导出
- `costSensitive/main_realtime.py`：实时抓包推理
- `costSensitive/cluster_unknown.py`：对 confirmed unknown 做聚类
- `run_full_chain.py`：一键启动 API 与后台入库链路

## 环境准备

```powershell
pip install -r requirements.txt
```

前端依赖：

```powershell
cd web-app
npm install
```

## 快速开始

### 1) 构建会话数据（可选）

```powershell
cd costSensitive
python session_data.py --dataset-root dataset/iscx --output-root processed_full/sessions --num-packets 12 --packet-len 256 --train-ratio 0.8 --flow-timeout-s 10 --seed 42
```

### 2) 训练模型

```powershell
python train_byte_session.py --manifest processed_full/sessions/manifest.csv --max_epoch 8 --batch-size 32
```

若 manifest 尚未生成，可用自动重建：

```powershell
python train_byte_session.py --rebuild-sessions --dataset-root dataset/iscx --session-root processed_full/sessions
```

### 3) 离线推理与 embedding 导出

```powershell
python inference.py --manifest processed_full/sessions/manifest.csv --model-path pytorch_model/byte_session_classifier.pth --split test
```

输出文件：

- `costSensitive/pytorch_model/session_predictions.csv`
- `costSensitive/pytorch_model/session_embeddings.npy`
- `costSensitive/pytorch_model/session_eval_report.json`

### 4) Unknown 聚类

```powershell
python cluster_unknown.py --pred-csv pytorch_model/session_predictions.csv --embeddings-npy pytorch_model/session_embeddings.npy
```

输出文件：

- `costSensitive/pytorch_model/unknown_clusters.json`
- `costSensitive/pytorch_model/unknown_cluster_history.json`
- `costSensitive/pytorch_model/unknown_cluster_assignments.csv`

### 5) 全链路启动（推荐）

在仓库根目录执行：

```powershell
python run_full_chain.py --host 127.0.0.1 --port 8000 --reload
```

Windows 下也可直接运行：

```powershell
start_full_chain.bat
```

常用参数：

- `--no-engine`：仅启 API，不启动后台入库引擎
- `--from-start`：从 JSONL 文件开头回放入库
- `--no-auto-capture`：不自动拉起实时抓包子进程
- `--capture-source`：指定网卡源（`\Device\NPF_{...}`）
- `--capture-device`：推理设备（`cpu`/`cuda`）

## 前端启动

```powershell
cd web-app
npm run dev
```

## API 目录（主要）

- `/api/health`
- `/api/overview`
- `/api/alerts`
- `/api/geo/source-heatmap`
- `/api/geo/source-drilldown`
- `/api/unknown/clusters/summary`
- `/api/unknown/clusters/trend`
- `/api/unknown/clusters/rebuild`
- `/api/unknown/clusters/ai-hints`
- `/api/xai/samples`
- `/api/xai/detail/{alert_id}`
- `/api/xai/explain/{alert_id}`
- `/api/ai/insights`
- `/api/model/metrics`

## XAI 页面当前交互

- 进入页面时仅显示工作台（不自动加载样本详情）
- 选择具体流量后，才显示包级贡献、字节热力图、可疑模式聚类、AI 解读和对比区

## 常见问题

- `npm run dev` 失败：先执行 `npm install`
- `run_full_chain.py` 失败：确认已安装 `requirements.txt` 依赖，并检查 Python 环境可用 `sqlite3`
- 热力图/聚类无内容：先确认已有告警数据和 XAI 字段入库，再检查 `/api/xai/samples`、`/api/xai/detail/{id}` 返回值
