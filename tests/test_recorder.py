from scipy.io import wavfile
from pathlib import Path
import numpy as np
import logging
import asyncio

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if True:
    from device import AudioRecorder, list_devices


async def test_basic_recording():
    """测试基本录音功能"""
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Test.Recorder')

    # 创建保存目录
    save_dir = Path("test_recordings")
    save_dir.mkdir(exist_ok=True)

    # 创建录音器
    recorder = AudioRecorder(
        channels=1,
        samplerate=16000,
        blocksize=1600
    )

    try:
        logger.info("开始录音测试（5秒）...")
        audio_chunks = []

        # 开始录音
        async for audio_data in recorder.start():
            # 收集5秒的音频数据
            audio_chunks.append(np.frombuffer(audio_data, dtype=np.int16))

            # 计算已录制的时长
            total_samples = sum(len(chunk) for chunk in audio_chunks)
            duration = total_samples / recorder.samplerate

            if duration >= 5.0:  # 录制5秒
                break

        # 合并音频数据
        audio_data = np.concatenate(audio_chunks)

        # 保存为WAV文件
        output_file = save_dir / "test_recording.wav"
        wavfile.write(output_file, recorder.samplerate, audio_data)
        logger.info(f"录音已保存到: {output_file}")

        # 打印音频信息
        logger.info(f"录音信息:")
        logger.info(f"- 采样率: {recorder.samplerate} Hz")
        logger.info(f"- 时长: {duration:.2f} 秒")
        logger.info(f"- 样本数: {len(audio_data)}")
        logger.info(f"- 通道数: {recorder.channels}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise
    finally:
        await recorder.stop()
        logger.info("测试完成")


async def test_recorder_properties():
    """测试录音器的属性和状态"""
    logger = logging.getLogger('Test.Recorder')

    recorder = AudioRecorder()

    # 测试初始状态
    assert not recorder.is_active, "录音器初始状态应该是非活动的"

    # 启动录音
    recording = recorder.start()
    async for _ in recording:
        # 测试活动状态
        assert recorder.is_active, "录音过程中应该是活动状态"
        break  # 只测试一个数据块

    # 停止录音
    await recorder.stop()
    assert not recorder.is_active, "停止后应该是非活动状态"
    logger.info("状态测试通过")


async def test_error_handling():
    """测试错误处理"""
    logger = logging.getLogger('Test.Recorder')

    # 测试无效设备
    try:
        recorder = AudioRecorder(device="不存在的设备")
        async for _ in recorder.start():
            pass
    except Exception as e:
        logger.info(f"预期的错误被捕获: {e}")

    # 测试无效参数
    try:
        recorder = AudioRecorder(samplerate=-1)
        async for _ in recorder.start():
            pass
    except Exception as e:
        logger.info(f"预期的错误被捕获: {e}")


async def main():
    """运行所有测试"""
    # 列出可用设备
    list_devices()

    # 运行测试
    print("\n=== 运行基本录音测试 ===")
    await test_basic_recording()

    print("\n=== 运行属性测试 ===")
    await test_recorder_properties()

    print("\n=== 运行错误处理测试 ===")
    await test_error_handling()

if __name__ == "__main__":
    asyncio.run(main())
