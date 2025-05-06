import asyncio
import nest_asyncio
import websockets
import json
import soundfile
import time

nest_asyncio.apply()

URI = "ws://localhost:7890/asr/streaming/pipeline"

async def send_audio_chunks(websocket, speech, chunk_frames: int = 9600):
    total_chunk_num = int(len(speech) - 1) // chunk_frames + 1
    for i in range(total_chunk_num):
        await asyncio.sleep(0.6)
        speech_chunk = speech[i * chunk_frames:(i + 1) * chunk_frames]
        data = {
            "chunk": speech_chunk.tolist(),
            "status": "processing",
            "is_final": i == total_chunk_num - 1
        }
        await websocket.send(json.dumps(data))
        print(f"Sent: success {str(speech_chunk[:5])}...")  # Print first 5 samples for brevity
        

    # Send end message
    end_message = {"status": "end", "chunk": []}
    await websocket.send(json.dumps(end_message))
    print(f"Sent: {end_message}")

async def receive_responses(websocket):
    results = []
    while True:
        try:
            response = await websocket.recv()
            response = json.loads(response)
            results.append({
                "action": "receive",
                "timestamp": time.time(),
                "data": response,
                "text": response.get("text", ""),
                "is_partial": response.get("is_partial", False)
            })
            print(f"Received: {response}")
        except websockets.ConnectionClosed:
            print("WebSocket connection closed.")
            break
    return results

async def stream_asr(speech, chunk_frames: int = 9600):
    async with websockets.connect(URI) as websocket:
        # Run send_audio_chunks and receive_responses concurrently
        send_task = send_audio_chunks(websocket, speech, chunk_frames)
        receive_task = receive_responses(websocket)
        results = await asyncio.gather(send_task, receive_task)
    return results[1]  # Return the results from receive_responses

if __name__ == "__main__":

    wav_file = '/Users/wanghuan/Projects/audio_services/cache/recording.wav'
    speech, _ = soundfile.read(wav_file)
    res = asyncio.run(stream_asr(speech))
    print('Got result', res)
    import json
    with open('result.json', 'w') as f:
        json.dump(res, f, indent=4)

