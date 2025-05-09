from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
import logging
from app.log import timer
from app.config import get_config


config = get_config()

@lru_cache(maxsize=None)
def load_model():
    return AutoModel(
        model=config.offline_asr_model_path,  
        vad_model=config.offline_vad_model_path,  
        punc_model=config.offline_punc_model_path,
        device = config.offline_model_device,
        disable_update = True)

@dataclass
class ParaformerOffline:
    def __post_init__(self):
        self.model = load_model()
    
    @timer("ParaformerOffline")
    def run(self, fp:str, batch_size_s=300, hotword=None):
        try:
            res = self.model.generate(input=fp, batch_size_s=batch_size_s, hotword=hotword)
        except Exception as e:
            logging.info(f"Error processing offline ASR: {str(fp)}")
            raise e
        return res