# NetGuard Web App (Vue3 + Vite)

前端负责展示加密流量检测的在线看板，连接后端 `FastAPI` 的 `/api/*` 接口。

## 页面说明

- `Overview`：总体态势、地理来源热力图、Unknown Cluster 监控
- `Alerts`：告警表格与筛选
- `XAI`：样本级可解释分析（包级贡献、字节热力图、可疑模式聚类、AI 解释）
- `AI Insights`：规则与模型输出的文本化洞察

## 目录结构

- `src/views/`：页面级视图
- `src/components/charts/`：ECharts 图表组件
- `src/stores/dashboard.js`：全局状态与 API 调用编排
- `src/services/api.js`：后端接口封装
- `src/layout/AppShell.vue`：主布局与导航

## 开发启动

```bash
npm install
npm run dev
```

默认 Vite 地址通常为 `http://localhost:5173`。

## 构建与检查

```bash
npm run build
npm run lint
```

## 交互约定

- XAI 页面默认仅显示工作台；选择具体流量样本后才展示详细分析内容。
- Unknown Cluster 区域支持手动刷新与重建聚类。
- 地理来源支持全局/中国范围切换及区域钻取。
