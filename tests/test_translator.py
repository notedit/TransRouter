import asyncio
import logging
import numpy as np
from pathlib import Path
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if True:
    from translator import AudioTranslator
    from device import AudioRecorder, AudioPlayer


async def generate_test_audio():
    """生成测试音频数据"""
    duration = 2  # 2秒
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    chunk_size = 1600  # 100ms
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        yield chunk.tobytes()
        await asyncio.sleep(0.1)


class MockGeminiTranscriber:
    """模拟 GeminiTranscriber"""

    def __init__(self):
        self.audio_in = asyncio.Queue()
        self.received_audio = []

    async def transcribe_audio(self, audio_data: bytes):
        await self.audio_in.put(audio_data)
        self.received_audio.append(audio_data)

    async def stop_session(self):
        pass


async def test_basic_translation():
    """测试基本翻译功能"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Test.Translator')

    # 创建临时目录
    test_dir = Path("test_output")
    test_dir.mkdir(exist_ok=True)

    translator = AudioTranslator()
    # 注入模拟的转录器
    translator.transcriber = MockGeminiTranscriber()

    try:
        # 启动翻译任务
        translate_task = asyncio.create_task(translator.run())

        # 等待启动
        await asyncio.sleep(0.5)

        # 运行3秒后停止
        await asyncio.sleep(3)
        await translator.stop()

        # 等待任务完成
        await translate_task

        # 验证录音文件是否创建
        recordings = list(translator.recordings_dir.glob("recording_*.wav"))
        assert len(recordings) > 0, "应该生成录音文件"

        logger.info("基本翻译测试通过")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise
    finally:
        await translator.stop()


async def test_audio_processing():
    """测试音频处理功能"""
    logger = logging.getLogger('Test.Translator')

    translator = AudioTranslator()
    translator.transcriber = MockGeminiTranscriber()

    # 测试音频处理
    test_audio = bytes([0] * 1600)  # 模拟音频数据
    await translator.process_audio(test_audio)

    # 验证音频是否被正确缓存
    assert len(translator.recording_buffer) > 0, "音频数据应该被添加到缓冲区"

    # 验证音频是否被发送到转录器
    assert len(translator.transcriber.received_audio) > 0, "音频数据应该被发送到转录器"

    logger.info("音频处理测试通过")


async def test_device_configuration():
    """测试设备配置"""
    logger = logging.getLogger('Test.Translator')

    # 测试默认配置
    translator = AudioTranslator()
    assert translator.input_sample_rate == 16000, "输入采样率应该是16000"
    assert translator.output_sample_rate == 24000, "输出采样率应该是24000"
    assert translator.channels == 1, "应该是单声道"

    # 测试录音器配置
    assert isinstance(translator.recorder,
                      AudioRecorder), "应该创建AudioRecorder实例"
    assert translator.recorder.samplerate == translator.input_sample_rate, "录音器采样率配置错误"

    # 测试播放器配置
    assert isinstance(translator.player, AudioPlayer), "应该创建AudioPlayer实例"
    assert translator.player.samplerate == translator.output_sample_rate, "播放器采样率配置错误"

    logger.info("设备配置测试通过")


async def test_error_handling():
    """测试错误处理"""
    logger = logging.getLogger('Test.Translator')

    translator = AudioTranslator()

    # 测试无效音频数据处理
    await translator.process_audio(None)
    await translator.process_audio(bytes())

    # 测试保存空音频
    result = translator.save_wav(None, translator.recordings_dir)
    assert result is None, "空音频数据应该返回None"

    # 测试异常停止
    await translator.stop()  # 应该能正常调用，不抛出异常

    logger.info("错误处理测试通过")


async def test_file_operations():
    """测试文件操作"""
    logger = logging.getLogger('Test.Translator')

    translator = AudioTranslator()

    # 测试目录创建
    assert translator.recordings_dir.exists(), "应该创建recordings目录"
    assert translator.synthesis_dir.exists(), "应该创建synthesis目录"

    # 测试音频保存
    test_audio = np.zeros(16000, dtype=np.int16)  # 1秒的静音
    filename = translator.save_wav(
        test_audio, translator.recordings_dir, "test")
    assert filename.exists(), "应该创建音频文件"

    logger.info("文件操作测试通过")


async def main():
    """运行所有测试"""
    print("\n=== 运行基本翻译测试 ===")
    await test_basic_translation()

    print("\n=== 运行音频处理测试 ===")
    await test_audio_processing()

    print("\n=== 运行设备配置测试 ===")
    await test_device_configuration()

    print("\n=== 运行错误处理测试 ===")
    await test_error_handling()

    print("\n=== 运行文件操作测试 ===")
    await test_file_operations()

if __name__ == "__main__":
    asyncio.run(main())
