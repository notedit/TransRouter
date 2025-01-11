import pyaudio
import wave
import time


def record_with_core_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    # 查找 Core Audio 设备
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if 'coreaudio' in device_info['name'].lower():
            device_index = i
            break

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK
    )

    frames = []
    try:
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            print(f"Received {len(data)} bytes")
    except KeyboardInterrupt:
        pass

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames
