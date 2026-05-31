# 🎯 目标检测系统（双模型版）

基于深度学习的目标检测系统，支持 **YOLOv8** + **Grounding DINO** 双模型，集成 GPU 加速与开放词汇检测能力。

---

## 🎨 UI 设计

**主题风格**：蓝绿清新配色
- 主背景：浅灰蓝渐变 `#D3DFDD → #A6C9CE`
- 主色调：蓝绿色 `#66AAB7`
- 辅助色：薄荷绿 `#9DD2C7`
- 强调色：橄榄绿 `#727665`
- 替补色：`#857B74`, `#A6C9CE`, `#0E4D66`

**配色表**：
| 元素 | 颜色代码 | 说明 |
|------|----------|------|
| 主背景 | #D3DFDD | 浅灰蓝 |
| 强调色 | #66AAB7 | 蓝绿色 |
| 辅助色 | #9DD2C7 | 薄荷绿 |
| 文字色 | #727665 | 橄榄绿 |
| 替补色 | #857B74, #0E4D66 | 灰棕、深青 |

---

## 🚀 快速启动

### ⭐ 最简单方式：双击 start.bat

直接双击 `start.bat` 文件即可启动服务。

### 手动启动
```powershell
cd d:\program\0429\AIProjects\project3_image_classifier
python app.py
```

### 访问地址
- **本地**：`http://localhost:5000`
- **局域网**：`http://你的电脑IP:5000`（手机同 WiFi 可访问）

### 分享给其他人
| 方式 | 说明 |
|------|------|
| ngrok 内网穿透 | `ngrok http 5000` 生成公网链接 |
| Docker 云部署 | `docker-compose up -d` 部署到服务器 |
| 同一 WiFi | 查看电脑 IP，手机浏览器访问 |

---

## 功能特性

- 🎯 **双模型架构**：
  - **YOLOv8s (GPU)**：COCO 80类快速检测，GPU 加速
  - **Grounding DINO (CPU)**：自定义提示词，开放词汇检测
- 📊 **智能切换**：空提示词自动使用 YOLOv8，自定义提示词切换到 Grounding DINO
- 📷 **实时可视化**：检测框、类别标签、置信度百分比
- 💻 **GPU 加速**：支持 NVIDIA CUDA (RTX 系列优化)
- 🔧 **灵活配置**：支持自定义提示词、阈值调节

## 检测策略

| 场景 | 模型 | 设备 | 说明 |
|------|------|------|------|
| 空提示词 | YOLOv8s | GPU | 快速检测 COCO 80 类 |
| 自定义提示词 | Grounding DINO | CPU | 开放词汇检测任意物体 |

## 技术栈

| 类别 | 技术 |
|------|------|
| 深度学习 | PyTorch + Ultralytics YOLOv8 + Grounding DINO |
| 后端 | Flask Web 服务 |
| 前端 | 原生 HTML/CSS/JavaScript |
| 加速 | CUDA (NVIDIA GPU) |

## 项目结构

```
project3_image_classifier/
├── app.py                    # Flask 主服务
├── requirements.txt          # Python 依赖
├── .env                      # 环境配置
├── .gitignore               # Git 忽略规则
├── README.md                 # 本文件
├── templates/
│   └── index.html           # 前端检测页面
├── uploads/                  # 临时上传目录
├── models/                   # 模型存储目录
├── weights/
│   └── groundingdino_swint_ogc.pth  # Grounding DINO 权重
├── yolov8n.pt/s.pt/m.pt     # YOLOv8 三个版本
├── GroundingDINO/            # Grounding DINO 子模块
└── venv/                     # Python 虚拟环境
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --user
```

### 2. 环境配置

创建 `.env` 文件（或修改现有配置）：

```env
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=False
MAX_CONTENT_LENGTH=16777216
BOX_THRESHOLD=0.35
TEXT_THRESHOLD=0.25
```

### 3. 运行服务

```bash
python app.py
```

### 4. 访问应用

访问地址：`http://localhost:5000`

- 上传图像进行目标检测
- 不填提示词：使用 YOLOv8 检测 COCO 80 类
- 填写提示词（如"猫.狗.汽车"）：使用 Grounding DINO 检测

## API 接口

| 端点 | 方法 | 功能 |
|------|------|------|
| `/` | GET | 首页 |
| `/detect` | POST | 图像检测（支持 multipart/form-data） |
| `/gpu_status` | GET | GPU 状态查询 |
| `/load_yolo` | POST | 重新加载 YOLOv8 |
| `/unload_yolo` | POST | 卸载 YOLOv8 释放显存 |
| `/health` | GET | 服务健康检查 |

## 模型说明

### YOLOv8 版本选择

| 版本 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|---------|
| yolov8n | 3.2M | 最快 | 基础 | 实时应用/演示 |
| **yolov8s** | 11.2M | 快 | 中等 | **默认/一般应用** |
| yolov8m | 25.9M | 中等 | 较高 | 生产环境 |
| yolov8l | 53.7M | 较慢 | 高 | 高精度需求 |
| yolov8x | 68.2M | 最慢 | 最高 | 最高精度 |

当前默认使用 **yolov8s**（平衡版，21.5MB）。

### Grounding DINO

支持开放词汇检测，可检测预训练模型未涵盖的任意物体类别。

权重文件：`weights/groundingdino_swint_ogc.pth`（662MB）

下载链接：https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

## GPU 要求

- 推荐：NVIDIA RTX 系列（6GB+ 显存）
- 最低：GTX 1060 6GB
- 依赖：CUDA 11.x + cuDNN

## 适合岗位

- 计算机视觉工程师
- AI 算法工程师
- 深度学习工程师
- 图像处理工程师

## 项目亮点

1. ✅ 双模型架构实战经验（YOLOv8 + Grounding DINO）
2. ✅ GPU 加速部署与 CUDA 优化
3. ✅ 开放词汇目标检测能力
4. ✅ Web 服务化与 API 开发
5. ✅ CV 领域核心技能展示

## 常见问题与解答

### Q1: Grounding DINO 模型加载失败？
**A:** 确保权重文件存在于 `weights/groundingdino_swint_ogc.pth`，如不存在请从以下链接下载：
```
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

### Q2: 提示词格式怎么写？
**A:** 使用英文句点 `.` 分隔不同类别，例如：`cat . dog . car . bicycle`
Grounding DINO 会将句点理解为类别分隔符。

### Q3: 如何调整检测灵敏度？
**A:** 修改 `.env` 中的阈值参数：
- `BOX_THRESHOLD=0.35` - 框阈值，越低越容易检出（默认0.35）
- `TEXT_THRESHOLD=0.25` - 文本阈值，越低越容易匹配（默认0.25）

### Q4: GPU 显存不足怎么办？
**A:** 
1. 使用较小的 YOLOv8 模型（yolov8n.pt 而非 yolov8m.pt）
2. 调用 `/unload_yolo` 释放显存
3. 降低图像分辨率

### Q5: 如何部署到服务器？
**A:** 支持 Docker 部署：
```bash
docker-compose up -d
```
详细配置见 `docker-compose.yml`。

### Q6: HuggingFace 下载慢？
**A:** 项目已配置国内镜像 `HF_ENDPOINT=https://hf-mirror.com`，Grounding DINO 依赖会自动使用镜像加速。

### Q7: 检测结果为空？
**A:** 可能原因：
- 图像中无目标物体
- 自定义提示词与图像内容不匹配
- 阈值设置过高 → 适当降低 `BOX_THRESHOLD`

### Q8: numpy 版本冲突？
**A:** Grounding DINO 部分依赖需要 `numpy<2.0`，requirements.txt 已做版本约束：
```
numpy>=1.23.0,<2.0
```
请勿手动升级 numpy 至 2.x。
