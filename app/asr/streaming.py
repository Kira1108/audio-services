from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
import numpy as np
# from app.audio_utils import resample_audio_librosa
from app.log import timer
from app.config import get_config
from funasr_onnx.paraformer_online_bin import Paraformer


config = get_config()

# @lru_cache(maxsize=None)
# def load_model():
#     return AutoModel(
#         model=config.online_asr_model_path, 
#         device = config.online_model_device,
#         disable_update = True)

# @dataclass
# class ParaformerStreaming:
#     chunk_ms:int = 600 
#     encoder_chunk_look_back:int = 4
#     decoder_chunk_look_back:int = 1
    
#     def __post_init__(self):
#         self.model = load_model()
#         self.cache = {}
#         self.chunk_size = [0, int(self.chunk_ms / 60), int(self.chunk_ms / 120)]
        

#     def new(self):
#         return ParaformerStreaming(
#             chunk_ms=self.chunk_ms,
#             encoder_chunk_look_back=self.encoder_chunk_look_back, 
#             decoder_chunk_look_back=self.decoder_chunk_look_back)
    
#     @timer("ParaformerStreaming")
#     def run(self, speech_chunk, sampling_rate:int =16000, is_final = False):
        
#         # if not sampling_rate != 16000:
#         #     speech_chunk = resample_audio_librosa(speech_chunk, sampling_rate, 16000)
            
#         speech_chunk = np.array(speech_chunk).astype("float32")
        
#         res = self.model.generate(
#             input=speech_chunk, 
#             cache=self.cache, is_final=is_final, 
#             chunk_size=self.chunk_size, 
#             encoder_chunk_look_back=self.encoder_chunk_look_back, 
#             decoder_chunk_look_back=self.decoder_chunk_look_back
#         )
#         return res

from dataclasses import dataclass
from functools import lru_cache


@lru_cache(maxsize=None)
def load_model():
    return Paraformer(
        model_dir=config.online_asr_model_path, 
        device = config.online_model_device,
        quantize = True,
        chunk_size = [0,10,5],
        disable_update = True)


@dataclass
class ParaformerStreaming:
    chunk_ms:int = 600 
    encoder_chunk_look_back:int = 4
    decoder_chunk_look_back:int = 1
    
    def __post_init__(self):
        self.model = load_model()
        self.cache = {}
        self.chunk_size = [0, int(self.chunk_ms / 60), int(self.chunk_ms / 120)]
        

    def new(self):
        return ParaformerStreaming(
            chunk_ms=self.chunk_ms,
            encoder_chunk_look_back=self.encoder_chunk_look_back, 
            decoder_chunk_look_back=self.decoder_chunk_look_back)

    def run(self, speech_chunk, sampling_rate:int =16000, is_final = False):
            
        speech_chunk = np.array(speech_chunk).astype("float32")
        
        res = self.model(
            audio_in=speech_chunk, 
            param_dict = dict(
                cache = self.cache,
                is_final = is_final,
                encoder_chunk_look_back = self.encoder_chunk_look_back,
                decoder_chunk_look_back = self.decoder_chunk_look_back
            ))
        if len(res) == 0:
            text = " "
        elif "preds" not in res[0]:
            text = " "
        else:
            try:
                text = res[0]['preds'][0]
            except:
                text = " "
        return [{"text":text}]
    
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
