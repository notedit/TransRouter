import asyncio
import logging
import os
from datetime import datetime
from scipy.io import wavfile
import numpy as np
from pathlib import Path
from device import AudioRecorder, AudioPlayer
from gemini_transcriber import GeminiTranscriber


def setup_logging():
    """设置日志配置"""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('logs', f'translator_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('TransRouter')


class AudioTranslator:
    def __init__(self, source_lang="zh-CN"):
        self.logger = setup_logging()

        # 音频设备配置
        self.input_device = None  # 使用系统默认麦克风
        self.output_device = None

        # 音频参数
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000
        self.channels = 1
        self.dtype = np.int16

        # 创建录音器和播放器
        self.recorder = AudioRecorder(
            device=self.input_device,
            channels=self.channels,
            samplerate=self.input_sample_rate,
            dtype=self.dtype
        )

        self.player = AudioPlayer(
            device=self.output_device,
            channels=self.channels,
            samplerate=self.output_sample_rate,
            dtype=self.dtype
        )

        # 初始化 Gemini 转录器
        self.transcriber = GeminiTranscriber()

        # 创建录音和合成音频的保存目录
        self.recordings_dir = Path("recordings")
        self.synthesis_dir = Path("synthesis")
        self.recordings_dir.mkdir(exist_ok=True)
        self.synthesis_dir.mkdir(exist_ok=True)

        # 用于保存音频数据的列表
        self.recording_buffer = []

        # 创建事件用于控制程序停止
        self.running = True

    def save_wav(self, audio_data, directory: Path, prefix="", sample_rate=None):
        """保存音频数据为WAV文件"""
        if audio_data is None or len(audio_data) == 0:
            self.logger.warning("没有音频数据")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = directory / f"{prefix}_{timestamp}.wav"

        wavfile.write(
            filename, sample_rate or self.input_sample_rate, audio_data)
        self.logger.info(f"音频已保存: {filename}")
        return filename

    async def process_audio(self, audio_data: bytes):
        """处理音频数据"""
        try:
            # 保存原始音频数据
            audio_array = np.frombuffer(audio_data, dtype=self.dtype)
            self.recording_buffer.append(audio_array)

            # 发送到转录器
            await self.transcriber.transcribe_audio(audio_data)

        except Exception as e:
            self.logger.error(f"处理音频时出错: {e}")

    async def start_streaming(self):
        """开始音频流处理"""
        try:
            # 启动播放任务
            playback_task = asyncio.create_task(
                self.player.play(self.transcriber.audio_in_iterator)
            )

            # 开始录音并处理
            self.logger.info("开始录音...（按 Ctrl+C 停止）")
            async for audio_data in self.recorder.start():
                if not self.running:
                    break
                await self.process_audio(audio_data)

        except Exception as e:
            self.logger.error(f"发生错误: {e}")
        finally:
            # 保存录音
            if self.recording_buffer:
                audio_data = np.concatenate(self.recording_buffer)
                self.save_wav(audio_data, self.recordings_dir, "recording")
                self.recording_buffer = []

            # 停止所有任务
            await self.stop()

            # 等待播放任务完成
            if playback_task:
                try:
                    await playback_task
                except asyncio.CancelledError:
                    pass

    async def run(self):
        """运行翻译器"""
        try:
            await self.start_streaming()
        except KeyboardInterrupt:
            self.running = False
            self.logger.info("程序已停止")

    async def stop(self):
        """停止翻译器"""
        self.running = False

        # 停止录音和播放
        await self.recorder.stop()
        await self.player.stop()

        # 停止转录器
        if hasattr(self, 'transcriber'):
            await self.transcriber.stop_session()


if __name__ == "__main__":
    translator = AudioTranslator()
    asyncio.run(translator.run())
