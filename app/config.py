from dataclasses import dataclass
import torch



@dataclass
class Config:
    
    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
        
config = Config()
