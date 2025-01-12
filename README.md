# TransRouter

TransRouter 是一个实时语音翻译工具，使用 Google Gemini 大模型进行中英文实时翻译。它可以直接与 Zoom 等会议软件集成，实现实时的语音翻译。

## 功能特点

- 实时语音翻译
- 中=》英翻译
- 自动语音合成
- 与 Zoom 等会议软件无缝集成
- 低延迟的流式处理

## 系统要求

- Python 3.8 或更高版本
- macOS 系统
- BlackHole 虚拟音频设备（用于音频路由）
- 稳定的网络连接
- Google Gemini API 密钥

## 安装步骤

1. 克隆仓库：

```bash
git clone https://github.com/notedit/TransRouter.git
cd TransRouter
```


2. 创建并激活虚拟环境：

Mac:
```bash
python -m venv venv
./venv/bin/activate
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```


4. 配置环境变量：
   - 复制 `.env.example` 为 `.env`
   - 填入您的 API 密钥：

```bash
GOOGLE_API_KEY=your_gemini_api_key
```


## 音频设备配置

### macOS
1. 安装 BlackHole：

```bash
brew install blackhole-2ch
```

2. 系统设置：
   - 打开系统偏好设置 > 声音
   - 确认可以看到 BlackHole 2ch 设备


### Zoom 配置
1. 打开 Zoom 设置 > 音频
2. 麦克风：选择系统默认麦克风
3. 扬声器：选择 "BlackHole 2ch"


## 使用说明

1. 查看可用设备：

```bash
python main.py --list-devices
```

2. 基本使用

```bash
python main.py
```

3. 指定输入输出设备

```bash
python main.py --input-device "输入设备名称" --output-device "输出设备名称"
```

4. 使用 BlackHole 作为输出设备：
```bash
python main.py --blackhole
```


## 技术实现

- 音频采集：使用 sounddevice 进行实时音频采集
- 语音翻译：使用 Google Gemini 大模型进行音频翻译
- 音频输出：使用异步音频流实现低延迟播放


## 常见问题

1. 找不到音频设备：
   - 检查 BlackHole 是否正确安装
   - 运行程序时查看打印的设备列表
   - 确认系统音频设置中可以看到虚拟设备

2. 翻译延迟：
   - 检查网络连接
   - 可能是 API 调用限制
   - 检查音频队列长度

3. 音频问题：
   - 确认采样率设置（输入16kHz，输出24kHz）
   - 检查音频设备路由
   - 验证 Zoom 音频设置

## 注意事项

1. API 使用：
   - 注意 API 调用限制和计费
   - 保护好 API 密钥

2. 音频设置：
   - 输入采样率 16kHz
   - 输出采样率 24kHz
   - 单声道音频
   - PCM 16bit 格式

3. 系统要求：
   - 确保 Python 环境正确
   - 安装必要的音频驱动
   - 保持充足的系统资源
