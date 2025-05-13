from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
import logging
from app.log import timer
from app.config import get_config
from funasr_onnx import Paraformer
from funasr.utils.postprocess_utils import  rich_transcription_postprocess
from app.punctuations.ct_punc import PunctuationModel
import numpy as np


config = get_config()

@lru_cache(maxsize=None)
def load_model():
    return AutoModel(
        model=config.offline_asr_model_path,  
        vad_model=config.offline_vad_model_path,  
        punc_model=config.offline_punc_model_path,
        device = config.offline_model_device,
        disable_update = True)

# @dataclass
# class ParaformerOffline:
#     def __post_init__(self):
#         self.model = load_model()
    
#     @timer("ParaformerOffline")
#     def run(self, fp:str, batch_size_s=300, hotword=None):
#         try:
#             res = self.model.generate(input=fp, batch_size_s=batch_size_s, hotword=hotword)
#         except Exception as e:
#             logging.info(f"Error processing offline ASR: {str(fp)}")
#             raise e
#         return res


@lru_cache(maxsize = None)
def load_onnx_model():
    print(config.offline_asr_model_path)
    print(config.offline_model_device)
    return Paraformer(
        model_dir = config.offline_asr_model_path,
        device = config.offline_model_device,
        quantize = True    
    )

@dataclass
class ParaformerOffline:
    def __post_init__(self):
        self.model = load_onnx_model()
        self.punc_model = PunctuationModel()
    
    @timer("ParaformerOffline")
    def run(self, fp:str, batch_size_s=300, hotword=None):
        try:
            # res = self.model.generate(input=fp, batch_size_s=batch_size_s, hotword=hotword)
            res = self.model(np.array(fp))
            res = res[0]['preds'][0]
            
            if len(res) ==0:
                return ""
            
            res = self.punc_model.run(res)
            res = rich_transcription_postprocess(res)
        except Exception as e:
            logging.info(f"Error processing offline ASR: {str(fp)}")
            raise e
        return [{"text":res}]