import asyncio
import websockets
import json
import soundfile
import os

# 每一个chunk有9600帧，在16000Hz的采样率下，chunk的时长为600ms
chunk_frames = 9600 
sampling_rate = 16000
chunk_ms = int(chunk_frames/sampling_rate*1000)

# 读取一个文件，将文件切分为多个chunk
wav_file = '/data1/wanghuan/audio_services/cache/asr_example.wav'
speech, _ = soundfile.read(wav_file)
total_chunk_num = int(len((speech)-1)/chunk_frames+1)
    
async def connect_to_websocket():
    uri = "ws://localhost:7788/asr/streaming"  # Replace with your server's address and port
    async with websockets.connect(uri) as websocket:
        
        # 逐个发送每个chunk(音频流)的数据
        for i in range(total_chunk_num):
            speech_chunk = speech[i*chunk_frames:(i+1)*chunk_frames]
            data = {
                "chunk": speech_chunk.tolist(),
                "status": "processing"
            }
            # 发送数据
            await websocket.send(json.dumps(data))
            print(f"Sent: success")

            # 接收数据
            response = await websocket.recv()
            print(f"Received: {response}")

        # 发送会话结束信号
        end_message = {"status": "end", "chunk": []}
        await websocket.send(json.dumps(end_message))
        print(f"Sent: {end_message}")

# Run the WebSocket client
res = asyncio.run(connect_to_websocket())
print('Got result', res)
