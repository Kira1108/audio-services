from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass

@lru_cache(maxsize=None)
def load_model():
    return AutoModel(
        model="paraformer-zh",  
        vad_model="fsmn-vad",  
        punc_model="ct-punc",
        device = 'cuda:1')

@dataclass
class ParaformerOffline:
    def __post_init__(self):
        self.model = load_model()
        
    def run(self, fp:str, batch_size_s=300, hotword=None):
        res = self.model.generate(input=fp, batch_size_s=batch_size_s, hotword=hotword)
        return res