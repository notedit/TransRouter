import typer
import asyncio
import logging
from typing import Optional
import device
from translator import AudioTranslator
import sounddevice as sd

# 创建 Typer 应用
app = typer.Typer(help="TransRouter - 实时语音翻译工具")


def get_default_output_device():
    """获取系统默认输出设备名称"""
    try:
        device_info = sd.query_devices(kind='output')
        return device_info['name']
    except Exception as e:
        logging.warning(f"获取默认输出设备失败: {e}")
        return None


@app.command()
def run(
    input_device: Optional[str] = typer.Option(
        None, "--input", "-i", help="输入设备名称，默认使用系统默认设备"),
    output_device: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="输出设备名称，默认使用系统默认设备。可选 'BlackHole 2ch' 用于 Zoom 集成"),
    input_sample_rate: int = typer.Option(
        16000, "--input-rate", "-ir", help="输入采样率"),
    output_sample_rate: int = typer.Option(
        24000, "--output-rate", "-or", help="输出采样率"),
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l", help="日志级别: DEBUG, INFO, WARNING, ERROR"),
    list_devices: bool = typer.Option(
        False, "--list-devices", "-ld", help="列出所有可用的音频设备"),
    use_blackhole: bool = typer.Option(
        False, "--blackhole", "-b", help="使用 BlackHole 2ch 作为输出设备")
):
    """
    启动 TransRouter 语音翻译服务
    """
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'无效的日志级别: {log_level}')
    logging.basicConfig(level=numeric_level)

    logger = logging.getLogger('TransRouter.Main')

    # 如果只是列出设备，则列出后退出
    if list_devices:
        device.list_devices()
        return

    # 创建翻译器实例
    translator = AudioTranslator()

    # 配置设备参数
    if input_device:
        translator.input_device = input_device

    # 确定输出设备
    if use_blackhole:
        translator.output_device = "BlackHole 2ch"
    elif output_device:
        translator.output_device = output_device
    else:
        default_output = get_default_output_device()
        if default_output:
            translator.output_device = default_output
            logger.info(f"使用系统默认输出设备: {default_output}")
        else:
            logger.error("无法获取默认输出设备")
            return

    translator.input_sample_rate = input_sample_rate
    translator.output_sample_rate = output_sample_rate

    # 运行翻译器
    async def main():
        try:
            await translator.run()
        except KeyboardInterrupt:
            await translator.stop()
            typer.echo("\n程序已停止")
        except Exception as e:
            typer.echo(f"发生错误: {e}", err=True)
            await translator.stop()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        typer.echo("\n程序已停止")


if __name__ == "__main__":
    app()
