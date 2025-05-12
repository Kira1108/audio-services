from dataclasses import dataclass
from pathlib import Path
import os
from functools import lru_cache


@dataclass
class Config:
    
    # 模型文件的存储文件夹
    resource_path: str = Path(__file__).parents[1] / "resources" / "modelscope" / "hub" / "iic"
    
    # 模型文件的存储路径
    offline_asr_model_name: str = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    offline_vad_model_name: str = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    offline_punc_model_name: str = "punc_ct-transformer_cn-en-common-vocab471067-large"
    online_asr_model_name: str = "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
    online_vad_model_name:str = "speech_fsmn_vad_zh-cn-16k-common-pytorch"
    online_punc_model_name:str = "punc_ct-transformer_cn-en-common-vocab471067-large"
    offline_model_device:str = "cuda:1"
    online_model_device:str = "cuda:0"
    
    def __post_init__(self):
        self.offline_asr_model_path = Path(self.resource_path) / self.offline_asr_model_name
        self.offline_vad_model_path = Path(self.resource_path) / self.offline_vad_model_name
        self.offline_punc_model_path = Path(self.resource_path) / self.offline_punc_model_name
        self.online_asr_model_path = Path(self.resource_path) / self.online_asr_model_name
        self.online_vad_model_path = Path(self.resource_path) / self.online_vad_model_name
        self.online_punc_model_path = Path(self.resource_path) / self.online_punc_model_name
        
        if not os.path.exists(self.offline_asr_model_path):
            raise FileNotFoundError(f"Offline ASR model not found at {self.offline_asr_model_path}")
        
        if not os.path.exists(self.offline_vad_model_path):
            raise FileNotFoundError(f"Offline VAD model not found at {self.offline_vad_model_path}")
        
        if not os.path.exists(self.offline_punc_model_path):
            raise FileNotFoundError(f"Offline Punc model not found at {self.offline_punc_model_path}") 
        
        if not os.path.exists(self.online_asr_model_path):
            raise FileNotFoundError(f"Online ASR model not found at {self.online_asr_model_path}")
        
        if not os.path.exists(self.online_vad_model_path):
            raise FileNotFoundError(f"Online VAD model not found at {self.online_vad_model_path}")
        
        if not os.path.exists(self.offline_punc_model_path):
            raise FileNotFoundError(f"Online Punc model not found at {self.online_punc_model_path}")
        
        
@lru_cache(maxsize=None)
def get_config() -> Config:
    return Config()

if __name__ == "__main__":
    config = get_config()
    print(f"Offline ASR model path: {config.offline_asr_model_path}")
    print(f"Offline VAD model path: {config.offline_vad_model_path}")
    print(f"Offline Punc model path: {config.offline_punc_model_path}")
    print(f"Online ASR model path: {config.online_asr_model_path}")
    print(f"Online VAD model path: {config.online_vad_model_path}")
    print(f"Online Punc model path: {config.online_punc_model_path}")