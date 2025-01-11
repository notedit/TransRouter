from scipy.io import wavfile
from pathlib import Path
import numpy as np
import logging
import asyncio
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if True:
    from device import AudioPlayer, list_devices


async def generate_test_audio():
    """生成测试音频数据"""
    # 生成一个3秒的测试音频（440Hz的正弦波）
    duration = 3
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # 每100ms发送一块数据
    chunk_size = 2400
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i + chunk_size]
        yield chunk.tobytes()
        await asyncio.sleep(0.1)


async def test_basic_playback():
    """测试基本播放功能"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Test.Player')

    # 创建播放器
    player = AudioPlayer(
        channels=1,
        samplerate=24000,
        blocksize=2400
    )

    try:
        logger.info("开始播放测试音频（3秒）...")

        # 播放测试音频
        await player.play(generate_test_audio())

        logger.info("播放完成")

        # 打印播放器信息
        logger.info(f"播放器信息:")
        logger.info(f"- 设备: {player.device}")
        logger.info(f"- 采样率: {player.samplerate} Hz")
        logger.info(f"- 通道数: {player.channels}")
        logger.info(f"- 块大小: {player.blocksize}")

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise
    finally:
        await player.stop()
        logger.info("测试完成")


async def test_file_playback():
    """测试播放WAV文件"""
    logger = logging.getLogger('Test.Player')

    # 创建测试音频文件
    test_file = Path("test_recordings") / "test_playback.wav"
    test_file.parent.mkdir(exist_ok=True)

    # 生成测试音频
    duration = 2
    sample_rate = 24000
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    wavfile.write(test_file, sample_rate, samples)

    async def file_audio_generator():
        """从文件读取音频数据的生成器"""
        chunk_size = 2400
        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            yield chunk.tobytes()
            await asyncio.sleep(0.1)

    # 创建播放器并播放文件
    player = AudioPlayer()
    try:
        logger.info(f"播放文件: {test_file}")
        await player.play(file_audio_generator())
    finally:
        await player.stop()


async def test_player_properties():
    """测试播放器的属性和状态"""
    logger = logging.getLogger('Test.Player')

    player = AudioPlayer()

    # 测试初始状态
    assert not player.is_active, "播放器初始状态应该是非活动的"

    # 开始播放
    try:
        play_task = asyncio.create_task(player.play(generate_test_audio()))
        # 等待播放器启动
        await asyncio.sleep(0.2)
        assert player.is_active, "播放过程中应该是活动状态"

        # 停止播放
        await player.stop()
        assert not player.is_active, "停止后应该是非活动状态"

        await play_task
    except Exception as e:
        logger.error(f"状态测试错误: {e}")
        raise

    logger.info("状态测试通过")


async def test_error_handling():
    """测试错误处理"""
    logger = logging.getLogger('Test.Player')

    # 测试无效设备
    try:
        player = AudioPlayer(device="不存在的设备")
        await player.play(generate_test_audio())
    except Exception as e:
        logger.info(f"预期的错误被捕获: {e}")

    # 测试无效参数
    try:
        player = AudioPlayer(samplerate=-1)
        await player.play(generate_test_audio())
    except Exception as e:
        logger.info(f"预期的错误被捕获: {e}")


async def test_queue_management():
    """测试队列管理"""
    logger = logging.getLogger('Test.Player')

    player = AudioPlayer()

    # 生成大量数据测试队列限制
    async def large_audio_generator():
        for _ in range(200):  # 生成大量数据
            yield bytes([0] * 2400)  # 生成静音数据
            await asyncio.sleep(0.01)

    try:
        await player.play(large_audio_generator())
    except Exception as e:
        logger.error(f"队列管理测试错误: {e}")
    finally:
        await player.stop()

    logger.info("队列管理测试完成")


async def test_direct_playback():
    """测试直接播放功能"""
    logger = logging.getLogger('Test.Player')

    player = AudioPlayer()

    async def test_audio_generator():
        # 生成1秒的440Hz正弦波
        duration = 1
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        samples = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        # 一次性发送所有数据
        yield samples.tobytes()
        yield None  # 发送结束标记

    try:
        await player.play(test_audio_generator())
        logger.info("直接播放测试通过")
    except Exception as e:
        logger.error(f"播放测试错误: {e}")
        raise


async def main():
    """运行所有测试"""
    # 列出可用设备
    list_devices()

    # 运行测试
    print("\n=== 运行基本播放测试 ===")
    await test_basic_playback()

    print("\n=== 运行文件播放测试 ===")
    await test_file_playback()

    print("\n=== 运行属性测试 ===")
    await test_player_properties()

    print("\n=== 运行错误处理测试 ===")
    await test_error_handling()

    print("\n=== 运行队列管理测试 ===")
    await test_queue_management()

    print("\n=== 运行直接播放测试 ===")
    await test_direct_playback()

if __name__ == "__main__":
    asyncio.run(main())
