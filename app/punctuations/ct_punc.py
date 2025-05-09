from funasr import AutoModel
from functools import lru_cache
from dataclasses import dataclass
from app.log import timer
from app.config import get_config

config = get_config()

@lru_cache(maxsize=None)
def load_punc_model():
    return AutoModel(
        model= config.offline_punc_model_path, 
        model_revision="v2.0.4",
        disable_update = True)

@dataclass
class PunctuationModel:

    def __post_init__(self):
        self.model = load_punc_model()
        
    @timer("CT-PuncModel")
    def run(self, text:str) -> str:
        if len(text) == 0:
            return text
        return self.model.generate(input=text)[0]['text']