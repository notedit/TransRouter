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


class AudioRecorder:
    """音频录制器"""

    def __init__(self, device=None, channels=1, samplerate=16000, blocksize=1600, dtype=np.int16):
        """
        初始化音频录制器

        Args:
            device: 输入设备名称或ID，默认为None（使用系统默认设备）
            channels: 通道数，默认为1（单声道）
            samplerate: 采样率，默认为16000Hz
            blocksize: 块大小，默认为1600（100ms的数据量）
            dtype: 数据类型，默认为np.int16
        """
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype

        self.queue = asyncio.Queue()
        self.stream = None
        self.is_running = False
        self.logger = logging.getLogger('TransRouter.Device.Recorder')

    def callback(self, indata, frames, time, status):
        """音频回调函数"""
        if status:
            self.logger.warning(f'状态: {status}')
        # 使用 call_soon_threadsafe 确保线程安全
        self.loop.call_soon_threadsafe(
            self.queue.put_nowait, bytes(indata))

    async def start(self) -> AsyncIterator[bytes]:
        """
        启动录音并返回音频数据迭代器

        Yields:
            bytes: 原始音频字节数据
        """
        try:
            # 创建原始输入流
            self.stream = sd.RawInputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                callback=self.callback
            )

            # 获取当前事件循环
            self.loop = asyncio.get_running_loop()

            # 使用上下文管理器确保流的正确关闭
            with self.stream:
                self.is_running = True
                self.logger.info(f'开始录音: 设备={self.device}, 采样率={
                                 self.samplerate}, 块大小={self.blocksize}')

                while self.is_running:
                    try:
                        # 异步等待音频数据
                        audio_data = await self.queue.get()
                        yield audio_data
                    except asyncio.CancelledError:
                        # 异步取消时退出
                        break
                    except Exception as e:
                        self.logger.error(f'录音错误: {e}')
                        break

        except Exception as e:
            self.logger.error(f'创建录音流错误: {e}')
            raise

        finally:
            await self.stop()
            self.logger.info('录音已停止')

    async def stop(self):
        """停止录音"""
        self.is_running = False
        if self.stream:
            self.stream.close()
            self.stream = None

        # 清空队列
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    @property
    def is_active(self):
        """检查录音器是否处于活动状态"""
        return self.is_running and self.stream is not None


class AudioPlayer:
    """音频播放器"""

    def __init__(self, device=None, channels=1, samplerate=24000, blocksize=2400, dtype=np.int16):
        """
        初始化音频播放器

        Args:
            device (str): 输出设备名称或ID
            channels (int): 通道数，默认为1（单声道）
            samplerate (int): 采样率，默认为24000Hz
            blocksize (int): 块大小，默认为2400（100ms的数据量）
            dtype (np.dtype): 数据类型，默认为np.int16
        """
        self.device = device
        self.channels = channels
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype

        self.queue = asyncio.Queue(maxsize=100)  # 限制队列大小防止内存占用过大
        self.done = asyncio.Event()
        self.stream = None
        self.is_running = False
        self.logger = logging.getLogger('TransRouter.Device.Player')

    async def clear(self):
        """清空播放队列"""
        try:
            while True:
                self.queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        # 重置事件
        self.done.clear()

        self.logger.debug("播放队列已清空")

    async def play(self, audio_iterator: AsyncIterator[bytes]) -> None:
        """
        播放音频数据

        Args:
            audio_iterator (AsyncIterator[bytes]): 音频数据迭代器
        """
        try:
            # 创建音频流
            self.stream = sd.RawOutputStream(
                device=self.device,
                channels=self.channels,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype
            )

            with self.stream:
                self.is_running = True
                self.logger.info(f'开始播放: 设备={self.device}')
                self.stream.start()

                # 主循环：读取数据并播放
                while not self.done.is_set():
                    try:
                        async for data in audio_iterator:
                            if not self.is_running:
                                break

                            if data is None:  # 结束标记
                                continue

                            try:
                                # 直接写入数据到流
                                self.stream.write(data)
                            except Exception as e:
                                self.logger.error(f'写入音频数据错误: {e}')
                                break

                    except Exception as e:
                        self.logger.error(f'播放循环错误: {e}')
                    finally:
                        self.done.set()

        except Exception as e:
            self.logger.error(f'创建音频流错误: {e}')
            raise

        finally:
            await self.stop()
            self.logger.info('播放已停止')

    async def stop(self):
        """停止播放"""
        self.is_running = False
        self.done.set()

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # 清空队列
        await self.clear()

    @property
    def is_active(self):
        """检查播放器是否处于活动状态"""
        return self.is_running and self.stream is not None


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

            # 创建播放器实例并播放测试音频
            player = AudioPlayer()
            await player.play(audio_generator())

        except KeyboardInterrupt:
            print("\n播放已停止")

    # 运行异步主函数
    asyncio.run(main())
