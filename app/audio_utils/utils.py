import numpy as np
# import librosa

# def resample_audio_librosa(
#     audio_data:np.ndarray, 
#     original_rate:int = 8000, 
#     target_rate:int = 16000):
#     audio_data = np.array(audio_data)
#     resampled_audio = librosa.resample(audio_data, orig_sr=original_rate, target_sr=target_rate)
#     return resampled_audio