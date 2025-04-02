from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
from app.config import config

@lru_cache(maxsize=None)
def load_punc_model():
    return AutoModel(model="ct-punc", model_revision="v2.0.4", device = config.device)

@dataclass
class PunctuationModel:

    def __post_init__(self):
        self.model = load_punc_model()
        
    def run(self, text:str) -> str:
        return self.model.generate(input=text)[0]['text']