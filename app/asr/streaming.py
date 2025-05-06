from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
import numpy as np
from app.audio_utils import resample_audio_librosa
from app.log import timer

@lru_cache(maxsize=None)
def load_model():
    return AutoModel(model="paraformer-zh-streaming", device = "cuda:0",disable_update = True)

@dataclass
class ParaformerStreaming:
    chunk_ms:int = 600 
    encoder_chunk_look_back:int = 4
    decoder_chunk_look_back:int = 1
    
    def __post_init__(self):
        self.model = load_model()
        self.cache = {}
        self.chunk_size = [0, int(self.chunk_ms / 60), int(self.chunk_ms / 120)]
    
    @timer("ParaformerStreaming")
    def run(self, speech_chunk, sampling_rate:int =16000, is_final = False):
        
        if not sampling_rate != 16000:
            speech_chunk = resample_audio_librosa(speech_chunk, sampling_rate, 16000)
            
        speech_chunk = np.array(speech_chunk).astype("float32")
        
        res = self.model.generate(
            input=speech_chunk, 
            cache=self.cache, is_final=is_final, 
            chunk_size=self.chunk_size, 
            encoder_chunk_look_back=self.encoder_chunk_look_back, 
            decoder_chunk_look_back=self.decoder_chunk_look_back
        )
        return res
    
if __name__ == "__main__":
    import soundfile
    
    chunk_frames = 9600 
    sampling_rate = 16000
    chunk_ms = int(chunk_frames/sampling_rate*1000)
    
    p = ParaformerStreaming(
        chunk_ms = chunk_ms, 
        sampling_rate = sampling_rate)
    
    wav_file = '/home/yixin/.cache/modelscope/hub/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/example/asr_example.wav'
    speech, _ = soundfile.read(wav_file)
    
    total_chunk_num = int(len((speech)-1)/chunk_frames+1)
    for i in range(total_chunk_num):
        speech_chunk = speech[i*chunk_frames:(i+1)*chunk_frames]
        is_final = i == total_chunk_num - 1
        res = p.run(speech_chunk, is_final)
        print(res)
