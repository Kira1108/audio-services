from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
import logging
from app.log import timer

@lru_cache(maxsize=None)
def load_model():
    return AutoModel(
        model="paraformer-zh",  
        vad_model="fsmn-vad",  
        punc_model="ct-punc",
        device = 'cuda:0',
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