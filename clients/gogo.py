import logging
logging.basicConfig(level=logging.INFO)
import asyncio
import nest_asyncio
import websockets
import json
import soundfile
import time

nest_asyncio.apply()

async def send_audio_chunks(websocket, speech, log:list):
    sampling_rate = 16000
    chunk_frames = 3200
    total_chunk_num = int(len((speech)-1)/chunk_frames+1)
    
    # Send each audio chunk
    sent_chunks = []  # Track chunks we've sent
    log.append({"type":"begin",'chunk_id':0, 'timestamp':time.time()})
    
    for i in range(total_chunk_num):
        await asyncio.sleep(0.2)  # Using sleep to simulate real-time processing
        speech_chunk = speech[i*chunk_frames:(i+1)*chunk_frames]
        # chunk_hash = hash(str(speech_chunk[:5]))  # Simple hash for tracking
        chunk_hash = ""
        sent_chunks.append(chunk_hash)
        
        is_final = i == total_chunk_num - 1
        data = {
            "chunk": speech_chunk.tolist(),
            "status": "processing",
            "is_final": is_final
        }
        
        # Send the data
        await websocket.send(json.dumps(data))
        log.append({"type":"send",'chunk_id':i + 1, 'timestamp':time.time()})
        print(f"Sent chunk {i+1}/{total_chunk_num} (hash: {chunk_hash})")
    
    # Send end session signal
    end_message = {"status": "end", "chunk": []}
    await websocket.send(json.dumps(end_message))
    print(f"Sent end message: {end_message}")
    

async def receive_results(websocket, expected_responses, log:list):
    results = []
    response_count = 0
    
    try:
        while True:
            response = await websocket.recv()
            print(response)
            response = json.loads(response)['data']
            log.append({
                "type":"receive",
                'chunk_id':response_count + 1, 
                'timestamp':time.time(),
                "start_time":response.get("start_time", None),
                "end_time":response.get("end_time", None),
                "text": response.get("text", None),
                "is_partial": response.get("is_partial", None),
                })
            results.append(response)
            response_count += 1
            print(f"Received result {response_count}: {response}")
            
            # Check if we've received all expected responses or if this is the final response
            if response_count >= expected_responses or response.get("is_final", False):
                print(f"Received all {response_count} expected responses or final response marker")
                break
    except Exception as e:
        print(f"Error receiving results: {e}")
    
    return results

async def stream_asr(speech):
    uri = "ws://localhost:7891/asr/streaming/pipeline"
    chunk_frames = 3200
    total_chunk_num = int(len((speech)-1)/chunk_frames+1)
    
    logs = []
    async with websockets.connect(uri) as websocket:
        send_task = asyncio.create_task(send_audio_chunks(websocket, speech, log = logs))
        receive_task = asyncio.create_task(receive_results(websocket, total_chunk_num, log = logs))
        results = await asyncio.gather(send_task, receive_task)
    return logs

# Load audio file (needs to be 16000Hz WAV format)
wav_file = '/Users/mac/Desktop/audio-services/cache/recording.wav'
speech, _ = soundfile.read(wav_file)

# Run the WebSocket client with concurrent tasks
res = asyncio.run(stream_asr(speech))
with open('log.json', 'w') as f:
    json.dump(res, f, indent=4, ensure_ascii=False)