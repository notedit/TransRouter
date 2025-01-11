from google import genai
from dotenv import load_dotenv
import os
import asyncio
import logging

from typing import AsyncIterator, TypeVar


T = TypeVar('T')


class QueueIterator(AsyncIterator[T]):
    def __init__(self, queue: asyncio.Queue[T]):
        self.queue = queue

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = await self.queue.get()
            if item is None:  # 使用 None 作为结束标志
                raise StopAsyncIteration
            return item
        finally:
            self.queue.task_done()


class GeminiTranscriber:
    def __init__(self, system_instruction: str = None, model_id: str = "models/gemini-2.0-flash-exp", modalities: list[str] = ["AUDIO"]):
        super().__init__()

        # 获取日志记录器
        self.logger = logging.getLogger('TransRouter.Gemini')

        # 加载环境变量
        load_dotenv(override=True)

        self.model_id = "models/gemini-2.0-flash-exp" if model_id is None else model_id

        if system_instruction is None:
            self.system_instruction = "As a professional interpreter, translate the audio input to English, providing only the translation with no additional text."
        else:
            self.system_instruction = system_instruction

        self.modalities = modalities if modalities is not None else ["AUDIO"]

        # 音频配置
        self.config = {
            "generation_config": {"response_modalities": self.modalities},
            "system_instruction": self.system_instruction
        }

        # 创建音频输入输出队列
        self.audio_in = asyncio.Queue()   # 接收从session获得的音频
        self.audio_out = asyncio.Queue(maxsize=50)  # 限制队列最大长度为50
        # 创建异步结果队列
        self.result_queue = asyncio.Queue()

        self.audio_in_iterator = QueueIterator(self.audio_in)

        # 创建会话
        self.client = genai.Client(api_key=os.getenv(
            'GOOGLE_API_KEY'), http_options={'api_version': 'v1alpha'})
        self.session = None
        self.send_task = None
        self.session_task = None

    async def _send_audio(self):
        """后台发送音频任务"""
        try:
            while True:
                # 从输出队列获取音频数据
                audio_data = await self.audio_out.get()
                if self.session:
                    await self.session.send(
                        {"data": audio_data, "mime_type": "audio/pcm"}
                    )
        except asyncio.CancelledError:
            self.logger.info("发送任务已取消")
        except Exception as e:
            import traceback
            self.logger.error(f"发送音频错误: {e}")
            self.logger.debug(traceback.format_exc())

    async def start_session(self):
        """启动会话并开始接收响应"""
        if self.session is None:
            async with self.client.aio.live.connect(
                model=self.model_id,
                config=self.config
            ) as session:
                self.session = session

                # 启动发送任务
                self.send_task = asyncio.create_task(self._send_audio())

                # 开始接收响应
                try:
                    while True:
                        full_text = ''
                        async for response in self.session.receive():
                            if response.server_content is not None:
                                model_turn = response.server_content.model_turn
                                if model_turn is not None and model_turn.parts:
                                    for part in model_turn.parts:
                                        if part.text is not None:
                                            full_text += part.text
                                        elif part.inline_data is not None:
                                            # 将接收到的音频放入输入队列
                                            await self.audio_in.put(part.inline_data.data)

                                if response.server_content.turn_complete:
                                    await self.result_queue.put(full_text)
                                    self.logger.info(f'turn_complete: ===')
                                    if full_text:
                                        self.logger.info(f'text: {full_text}')
                                        full_text = ''
                                    # 发送结束信号
                                    await self.audio_in.put(None)

                except Exception as e:
                    self.logger.error(f"接收响应错误: {e}")
                    if self.session:
                        await self.session.close()
                        self.session = None

    async def stop_session(self):
        """停止会话"""
        if self.send_task:
            self.send_task.cancel()
            try:
                await self.send_task
            except asyncio.CancelledError:
                pass
            self.send_task = None

        if self.session_task:
            self.session_task.cancel()
            try:
                await self.session_task
            except asyncio.CancelledError:
                pass
            self.session_task = None

        if self.session:
            await self.session.close()
            self.session = None

    async def transcribe_audio(self, audio_data):
        """转录音频数据"""
        try:
            # 确保会话已启动
            if self.session_task is None:
                # 启动会话任务
                self.session_task = asyncio.create_task(self.start_session())

            # 检查队列长度，如果已满则丢弃
            if self.audio_out.qsize() >= 50:
                self.logger.warning("音频队列已满，丢弃当前数据")
                return None

            # 将音频数据放入输出队列
            try:
                await asyncio.wait_for(
                    self.audio_out.put(audio_data),
                    timeout=0.1  # 100ms超时
                )
            except asyncio.TimeoutError:
                self.logger.warning("放入音频队列超时")
                return None

            return None

        except Exception as e:
            self.logger.error(f"Gemini 转录错误: {e}")
            return None
