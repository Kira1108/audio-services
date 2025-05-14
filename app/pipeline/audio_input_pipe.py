from dataclasses import dataclass
from app.asr import ParaformerStreaming
from app.asr import ParaformerOffline
from app.vad import FMSNVad
from app.punctuations import PunctuationModel
from app.schemas.core import ASRResult
from uuid import uuid4
import numpy as np
import logging

# @dataclass
# class AudioInputPipeline:
    
#     chunk_ms: int = 200
    
#     def __post_init__(self):
#         self.vad = FMSNVad()
#         self.asr = ParaformerStreaming(chunk_ms=600)
#         self.offline = ParaformerOffline()
#         self.punc_model = PunctuationModel()
#         self.speech_cache = np.array([])
#         self.chunk_id = 0
#         self.conversation_id = str(uuid4())
#         self.partial_cache = np.array([])
#         self.pending_start:int = None
        
#     def parse(
#         self, 
#         speech_chunk, 
#         sampling_rate = 16000, 
#         is_final = False):
        
#         start_time = self.chunk_id *self.chunk_ms
#         end_time = (self.chunk_id + 1) * self.chunk_ms
        
#         # 语音流缓存
#         speech_chunk = np.array(speech_chunk)
#         self.speech_cache = np.concatenate([self.speech_cache, speech_chunk], axis = 0)
#         self.partial_cache = np.concatenate([self.partial_cache, speech_chunk], axis = 0)
        

#         vad_result = self.vad.vad(
#             speech_chunk, 
#             sample_rate=sampling_rate, 
#             is_final=is_final)
        
#         complete_vad = vad_result or is_final
        
#         is_partial = True
        
#         if complete_vad:
#             complete_asr_result = self.offline.run(self.speech_cache)
#             logging.info(f"Complete ASR result: {complete_asr_result}, Chunk: {self.chunk_id}, {len(self.speech_cache)}")
#             self.speech_cache = np.array([])
#             is_partial = False
            
#             if len(complete_asr_result) == 0:
#                 complete_asr_result = " "
#             else:
#                 complete_asr_result = complete_asr_result[0].get('text', " ")
            
#         # 如果缓存的语音流小于9600帧
#         if len(self.partial_cache) < 9600:
#             self.chunk_id += 1
#             partial_asr_result = ""
#             logging.info(f"Skip short vad")
#             if not complete_vad:
#                 return None
        
#         else:
#             partial_asr_result = self.asr.run(
#                 self.partial_cache, 
#                 sampling_rate=sampling_rate, 
#                 is_final = is_final)
#             logging.info(f"Partial ASR result: {partial_asr_result}, Chunk: {self.chunk_id}, {len(self.partial_cache)}")
#             self.partial_cache = np.array([])

#         if len(partial_asr_result) == 0:
#             partial_asr_result = " "
#         else:
#             partial_asr_result = partial_asr_result[0].get('text', " ")

        
#         logging.info(f"Is complete vad: {complete_vad} ")
#         if complete_vad:    
#             res = ASRResult(
#                 conversation_id=self.conversation_id,
#                 chunk_id=self.chunk_id + 1,
#                 start_time=start_time,
#                 end_time=end_time,
#                 text=complete_asr_result,
#                 is_partial=is_partial
#             )
#             logging.info(str(res))
#         else:
#             res = ASRResult(
#                 conversation_id=self.conversation_id,
#                 chunk_id=self.chunk_id + 1,
#                 start_time=start_time,
#                 end_time=end_time,
#                 text=partial_asr_result,
#                 is_partial=is_partial
#             )
#             logging.info(str(res))
        
#         self.chunk_id += 1
#         return res
    
@dataclass
class AudioInputPipeline:
    
    chunk_ms: int = 200
    
    def __post_init__(self):
        self.vad = FMSNVad()
        self.asr = ParaformerStreaming(chunk_ms=600)
        self.offline = ParaformerOffline()
        self.punc_model = PunctuationModel()
        self.speech_cache = np.array([])
        self.chunk_id = 0
        self.conversation_id = str(uuid4())
        self.partial_cache = np.array([])
        self.pending_start:int = None
        
    def parse(
        self, 
        speech_chunk, 
        sampling_rate = 16000, 
        is_final = False):
        
        start_time = self.chunk_id *self.chunk_ms
        end_time = (self.chunk_id + 1) * self.chunk_ms
        
        # 语音流缓存
        speech_chunk = np.array(speech_chunk)
        self.speech_cache = np.concatenate([self.speech_cache, speech_chunk], axis = 0)
        self.partial_cache = np.concatenate([self.partial_cache, speech_chunk], axis = 0)
        

        vad_result = self.vad.vad(
            speech_chunk, 
            sample_rate=sampling_rate, 
            is_final=is_final)
        
        complete_vad = vad_result or is_final
        
        is_partial = True
        
        if complete_vad:
            complete_asr_result = self.offline.run(self.speech_cache)
            logging.info(f"Complete ASR result: {complete_asr_result}, Chunk: {self.chunk_id}, {len(self.speech_cache)}")
            self.speech_cache = np.array([])
            is_partial = False
            
            if len(complete_asr_result) == 0:
                complete_asr_result = " "
            else:
                complete_asr_result = complete_asr_result[0].get('text', " ")
                
            res = ASRResult(
                conversation_id=self.conversation_id,
                chunk_id=self.chunk_id + 1,
                start_time=start_time,
                end_time=end_time,
                text=complete_asr_result.strip(),
                is_partial=is_partial
            )
            
            self.chunk_id += 1
            self.asr = self.asr.new()
            return res
            
        # 如果缓存的语音流小于9600帧
        if len(self.partial_cache) < 9600:
            self.chunk_id += 1
            partial_asr_result = ""
            logging.info(f"Skip short vad")
            if not complete_vad:
                return None
        
        else:
            partial_asr_result = self.asr.run(
                self.partial_cache, 
                sampling_rate=sampling_rate, 
                is_final = is_final)
            logging.info(f"Partial ASR result: {partial_asr_result}, Chunk: {self.chunk_id}, {len(self.partial_cache)}")
            self.partial_cache = np.array([])

        if len(partial_asr_result) == 0:
            partial_asr_result = " "
        else:
            partial_asr_result = partial_asr_result[0].get('text', " ")
            
        self.chunk_id += 1
        
        return ASRResult(
                conversation_id=self.conversation_id,
                chunk_id=self.chunk_id + 1,
                start_time=start_time,
                end_time=end_time,
                text=partial_asr_result.strip(),
                is_partial=is_partial
            )
        
        # logging.info(f"Is complete vad: {complete_vad} ")
        # if complete_vad:    
        #     res = ASRResult(
        #         conversation_id=self.conversation_id,
        #         chunk_id=self.chunk_id + 1,
        #         start_time=start_time,
        #         end_time=end_time,
        #         text=complete_asr_result,
        #         is_partial=is_partial
        #     )
        #     logging.info(str(res))
        # else:
        #     res = ASRResult(
        #         conversation_id=self.conversation_id,
        #         chunk_id=self.chunk_id + 1,
        #         start_time=start_time,
        #         end_time=end_time,
        #         text=partial_asr_result,
        #         is_partial=is_partial
        #     )
        #     logging.info(str(res))
        
        # self.chunk_id += 1
        # return res