import sounddevice as sd
import numpy as np
import asyncio
import logging
from typing import AsyncIterator


logger = logging.getLogger('TransRouter.Device')


def list_devices():
    # 列出设备列表，并返回
    devices = sd.query_devices()

    default_input = sd.query_devices(kind='input')
    print(f"\n默认输入设备: {default_input['name']}")
    print(f"支持的采样率: {default_input['default_samplerate']}")

    # 打印可用设备信息，方便调试
    print("\n可用音频设备:")
    print(sd.query_devices())

    return devices


async def record(device=None, channels=1, samplerate=16000, blocksize=1600, dtype=np.int16) -> AsyncIterator[bytes]:
    """
    创建异步音频采集迭代器，返回原始字节数据

    Args:
        device: 输入设备名称或ID，默认为None（使用系统默认设备）
        channels: 通道数，默认为1（单声道）
        samplerate: 采样率，默认为16000Hz
        blocksize: 块大小，默认为1600（100ms的数据量）
        dtype: 数据类型，默认为np.int16

    Yields:
        bytes: 原始音频字节数据
    """
    queue = asyncio.Queue()

    def callback(indata, frames, time, status):
        """音频回调函数"""
        if status:
            logger.warning(f'状态: {status}')
        # 使用 call_soon_threadsafe 确保线程安全
        asyncio.get_event_loop().call_soon_threadsafe(
            queue.put_nowait, bytes(indata))

    try:
        # 创建原始输入流
        stream = sd.RawInputStream(
            device=device,
            channels=channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype=dtype,
            callback=callback
        )

        # 使用上下文管理器确保流的正确关闭
        with stream:
            logger.info(f'开始录音: 设备={device}, 采样率={
                        samplerate}, 块大小={blocksize}')
            while True:
                try:
                    # 异步等待音频数据
                    audio_data = await queue.get()
                    yield audio_data
                except asyncio.CancelledError:
                    # 异步取消时退出
                    break
                except Exception as e:
                    logger.error(f'录音错误: {e}')
                    break

    except Exception as e:
        logger.error(f'创建录音流错误: {e}')
        raise

    finally:
        logger.info('录音已停止')


async def playout(
    audio_iterator: AsyncIterator[bytes],
    device: str = "BlackHole 2ch",
    channels: int = 1,
    samplerate: int = 24000,
    blocksize: int = 2400,
    dtype: np.dtype = np.int16
) -> None:
    """
    异步播放音频数据
    """
    queue = asyncio.Queue(maxsize=100)  # 限制队列大小防止内存占用过大
    done = asyncio.Event()
    buffer = bytearray()

    def callback(outdata, frames, time, status):
        nonlocal buffer
        if status:
            logger.warning(f'状态: {status}')

        required_bytes = frames * channels * np.dtype(dtype).itemsize

        try:
            loop = asyncio.get_event_loop()
            if len(buffer) < required_bytes:
                try:
                    # 获取新数据
                    future = asyncio.run_coroutine_threadsafe(
                        queue.get(), loop)
                    buffer.extend(future.result(timeout=0.1))
                except Exception:
                    # 统一处理所有异常情况
                    if done.is_set() and len(buffer) == 0:
                        raise sd.CallbackStop()
                    # 不够的部分用静音填充
                    buffer.extend(bytes(required_bytes - len(buffer)))

            # 写入数据并更新buffer
            outdata.write(bytes(buffer[:required_bytes]))
            del buffer[:required_bytes]

        except Exception as e:
            logger.error(f'播放错误: {e}')
            raise sd.CallbackStop()

    try:
        # 创建音频流
        with sd.RawOutputStream(
            device=device,
            channels=channels,
            samplerate=samplerate,
            blocksize=blocksize,
            dtype=dtype,
            callback=callback
        ) as stream:
            logger.info(f'开始播放: 设备={device}')
            stream.start()

            # 主循环：读取数据到队列
            try:
                async for data in audio_iterator:
                    await queue.put(data)
            finally:
                done.set()

            # 等待播放完成
            while not queue.empty() or len(buffer) > 0:
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f'播放错误: {e}')
        raise

    finally:
        logger.info('播放已停止')


# 使用示例
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 列出设备
    list_devices()

    # 测试录音和播放
    async def main():
        try:
            # 创建一个简单的音频生成器用于测试
            async def audio_generator():
                # 生成一个简单的正弦波
                duration = 3  # 3秒
                t = np.linspace(0, duration, int(24000 * duration), False)
                samples = (np.sin(2 * np.pi * 440 * t)
                           * 32767).astype(np.int16)

                # 每100ms发送一块数据
                chunk_size = 2400
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i:i + chunk_size]
                    yield chunk.tobytes()
                    await asyncio.sleep(0.1)

            # 播放测试音频
            await playout(audio_generator())

        except KeyboardInterrupt:
            print("\n播放已停止")

    # 运行异步主函数
    asyncio.run(main())
