# 🩺基于贝叶斯神经网络的可解释性黑素瘤诊断系统
本项目是一个基于 **Flask** 框架开发的医学图像智能诊断平台，融合了 **贝叶斯神经网络（Bayesian Neural Network）** 进行皮肤病图像的分类与不确定性预测。前端支持用户上传图像、查看诊断结果，后端支持模型训练与推理，使用了公开皮肤病图像数据集 **HAM10000**。

## 📦 项目结构
```
flask_test/
├── app/                  # 前端Web应用（Flask）
│   ├── models/           # Flask层模型定义
│   ├── routes/           # 路由配置
│   ├── static/           # 静态资源（CSS、JS、图片、用户上传的图片）
│   └── templates/        # HTML模板文件
├── backend/              # 后端深度学习模块
│   ├── config/           # 参数配置模块
│   ├── core/             # 核心代码（模型、损失、数据处理）
│   ├── data/             # 数据集与元数据
│   ├── results/          # 模型输出结果（如日志、模型文件）
│   ├── scripts/          # 辅助脚本（如训练脚本）
│   ├── util/             # 工具函数与辅助模块
│   ├── test.py           # 测试脚本入口
│   └── train.py          # 训练脚本入口
├── config.py             # flask全局配置入口
├── run.py                # 启动 Flask 应用主程序
├── requirements.txt      # Python依赖列表
├── LICENSE.txt           # 授权协议
├── .gitattributes.txt    # 定义 Git 对文件的处理规则
├── .gitignore.txt        # 指定哪些文件不被 Git 跟踪和提交
└── README.md             # reademe文件
```

## 🚀 快速开始

### 1️⃣ 安装环境依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 下载数据集

- 下载 [HAM10000 数据集](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- 解压到 `backend/data/` 目录下

### 3️⃣ 模型训练

```bash
python train.py
```

### 4️⃣ 启动 Web 服务

```bash
python run.py
```

## 🧠 模型说明

- `ResNet50_Self_Attn`：带自注意力机制的 ResNet-50 网络
- `MLP_Skin`：贝叶斯线性变换层（用于不确定性估计）
- `EDLClassifier`：证据深度学习分类器（输出类别 + 置信度 + 不确定性）

## ✅ 功能特性

- ✔️ 图像上传诊断
- ✔️ 不确定性输出
- ✔️ 模型训练与评估
- ✔️ 前后端分离结构

## 🏆 最佳实验结果展示

以下为模型在训练过程中的最佳评估结果（截取自训练日志）：

| 指标                    | 数值        |
| ----------------------- | ----------- |
| 最高准确率    | 89.82%      |
| 最高置信度  | 84.93%      |
| 对应精确度  | 16.97%       |


