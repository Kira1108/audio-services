from dataclasses import dataclass
from app.asr import ParaformerStreaming
from app.asr import ParaformerOffline
from app.vad import FMSNVad
from app.punctuations import PunctuationModel
from app.schemas.core import ASRResult
from uuid import uuid4
import numpy as np

@dataclass
class AudioInputPipeline:
    
    chunk_ms: int = 600
    
    def __post_init__(self):
        self.vad = FMSNVad()
        self.asr = ParaformerStreaming(chunk_ms=self.chunk_ms)
        self.offline = ParaformerOffline()
        self.punc_model = PunctuationModel()
        self.stream_cache = []
        self.temp_cache = []
        self.speech_cache = np.array([])
        self.chunk_id = 0
        self.conversation_id = str(uuid4())
        
    def parse(
        self, 
        speech_chunk, 
        sampling_rate = 16000, 
        is_final = False):
        
        start_time = self.chunk_id *self.chunk_ms
        end_time = (self.chunk_id + 1) * self.chunk_ms
        
        # 语音流缓存
        speech_chunk = np.array(speech_chunk)
        self.speech_cache = np.concatenate([self.speech_cache, speech_chunk], axis = 0)

        vad_result = self.vad.vad(
            speech_chunk, 
            sample_rate=sampling_rate, 
            is_final=is_final)
        
        complete_vad = vad_result or is_final
        
        if not complete_vad:
            asr_result = self.asr.run(
                speech_chunk, 
                sampling_rate=sampling_rate, 
                is_final = is_final)
        else:
            asr_result = self.offline.run(self.speech_cache)
            
            
        if len(asr_result) == 0:
            asr_result = " "
            
        else:
            asr_result = asr_result[0].get('text', " ")
        
        # 整体ASR流缓存
        self.stream_cache.append(asr_result)
        
        if complete_vad:
            # t = [v[0]['text'] for v in self.temp_cache]
            # output = "".join(t)
            # 重置vad缓存
            self.speech_cache = np.array([])
            self.temp_cache = []
            output = self.punc_model.run(asr_result)
            is_partial = False
        else:
            self.temp_cache.append(asr_result)
            output = asr_result
            is_partial = True
            
        res = ASRResult(
            conversation_id=self.conversation_id,
            chunk_id=self.chunk_id + 1,
            start_time=start_time,
            end_time=end_time,
            text=output,
            is_partial=is_partial
        )
        
        self.chunk_id += 1
        
        return res