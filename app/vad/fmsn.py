from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from funasr import AutoModel
from app.audio_utils import resample_audio_librosa
from app.log import timer
from app.config import get_config

config = get_config()

@lru_cache(maxsize=None)
def load_fmsn():
    return AutoModel(
        model=config.online_vad_model_path,
        model_revision="v2.0.4",
        disable_update = True)

@dataclass
class FMSNVad:
    
    def __post_init__(self):
        self.model = load_fmsn()
        self.cache = {}
        self.timestamp = 0
        
    def run(self, speech_chunk, sample_rate:int = 16000, is_final:bool = False):
        if not sample_rate == 16000:
            speech_chunk = resample_audio_librosa(speech_chunk, sample_rate, 16000)
            
        chunk_seconds = int(len(speech_chunk) * 1000 // 16000)  # in milliseconds
        speech_chunk = np.array(speech_chunk).astype("float32") # convert to float32 type
        self.timestamp += chunk_seconds
        if len(speech_chunk) == 0:
            return [{'value':[-1,self.timestamp]}]
        
        res = self.model.generate(
            input=speech_chunk, 
            cache=self.cache, 
            is_final=is_final, 
            chunk_size=chunk_seconds
        )
        return res
    
    @timer("FMSNVad")
    def vad(self, speech_chunk, sample_rate:int = 16000, is_final:bool = False):
        res = self.run(speech_chunk, sample_rate, is_final)
        # 最后一个vad段，是一个结束的vad段
        
        try:
            if res[0]['value'][-1][-1] >=0:
                return True
        except:
            return False
        
        return False