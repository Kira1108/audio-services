from fastapi import BackgroundTasks
from fastapi.routing import APIRouter

from app.asr import ParaformerOffline, ParaformerStreaming
from app.pipeline.audio_input_pipe import AudioInputPipeline
from typing import Optional
from fastapi import WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
from pathlib import Path
from uuid import uuid4

router = APIRouter(prefix = "/asr",tags = ['asr'])
asr = ParaformerStreaming()
_ = AudioInputPipeline()

@router.websocket("/streaming")
async def websocket_endpoint(websocket: WebSocket, sessionId: Optional[str] = None):
    await websocket.accept()
    
    chunk_ms = asr.chunk_ms
    chunk_id = 0
    session_key = str(uuid4())
    
    while True:
        data = await websocket.receive_json() 
        if data.get("status") == "end":
            await websocket.close()
            break
            
        is_final = data.get("is_final", False)
        chunk = data['chunk']
        asr_info = asr.run(chunk, is_final)[0]
        
        await websocket.send_json(
            {   
                'session_key':session_key,
                'text':asr_info['text'], 
                "start_time": chunk_id * chunk_ms,
                "end_time": (chunk_id + 1) * chunk_ms,
                "chunk_id": chunk_id,
                "message_key": asr_info['key']
            }
        )
        chunk_id += 1
        

@router.websocket("/streaming/pipeline")
async def websocket_pipeline_endpoint(websocket: WebSocket, sessionId: Optional[str] = None):
    
    pipe = AudioInputPipeline()
    await websocket.accept()
    
    while True:
        data = await websocket.receive_json() 
        if data.get("status") == "end":
            await websocket.close()
            break
            
        is_final = data.get("is_final", False)
        chunk = data['chunk']
        
        asr_info = pipe.parse(chunk, is_final=is_final)
        
        if asr_info is None:
            continue
        
        else:
            
            await websocket.send_json(   
            {
                "code":0,
                "msg":"success",
                "data": asr_info.model_dump()
            }
            )


@router.post("/offline")
def paraformer_offline_endpoint(file: UploadFile = File(...)):
    paraformer_offline = ParaformerOffline()
    # find the folder of the current script
    current_fp = Path(__file__).parent.parent.parent
    # make a cache folder
    cache_folder = current_fp / "cache"
    cache_folder.mkdir(exist_ok=True)
    fp = cache_folder / file.filename
    with fp.open("wb") as f:
        f.write(file.file.read())
    res = paraformer_offline.run(str(fp.resolve()))
    return JSONResponse(content={"result": res})
