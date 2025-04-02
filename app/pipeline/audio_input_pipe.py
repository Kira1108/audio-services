from dataclasses import dataclass
from app.asr import ParaformerStreaming
from app.vad import FMSNVad
from app.punctuations import PunctuationModel
from app.schemas.core import ASRResult
from uuid import uuid4

@dataclass
class AudioInputPipeline:
    
    chunk_ms: int = 600
    
    def __post_init__(self):
        self.vad = FMSNVad()
        self.asr = ParaformerStreaming(chunk_ms=self.chunk_ms)
        self.punc_model = PunctuationModel()
        self.stream_cache = []
        self.temp_cache = []
        self.chunk_id = 0
        self.conversation_id = str(uuid4())
        
    def parse(
        self, 
        speech_chunk, 
        sampling_rate = 16000, 
        is_final = False):
        
        start_time = self.chunk_id *self.chunk_ms
        end_time = (self.chunk_id + 1) * self.chunk_ms

        asr_result = self.asr.run(
            speech_chunk, 
            sampling_rate=sampling_rate, 
            is_final = is_final)
        
        # vad缓存
        self.temp_cache.append(asr_result)
        
        # 整体ASR流缓存
        self.stream_cache.append(asr_result)
        
        if self.vad.vad(
            speech_chunk, 
            sample_rate = sampling_rate, 
            is_final = is_final):
            
            t = [v[0]['text'] for v in self.temp_cache]
            output = "".join(t)
            # 重置vad缓存
            self.temp_cache = []
            output = self.punc_model.run(output)
            is_partial = False
        else:
            output = asr_result[0]['text']
            is_partial = True
            
        self.chunk_id += 1
        
        return ASRResult(
            conversation_id=self.conversation_id,
            chunk_id=self.chunk_id + 1,
            start_time=start_time,
            end_time=end_time,
            text=output,
            is_partial=is_partial
        )