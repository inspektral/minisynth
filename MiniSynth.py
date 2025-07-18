from abc import ABC, abstractmethod
import numpy as np
from scipy.signal import butter, filtfilt

class MiniSynth(ABC):
    def __init__(self, sr: int, duration: float):
        self.sr = sr
        self.duration = duration
        self.samples = int(sr * duration)
        self.wavetable = None

    @abstractmethod
    def render(self):
        """Render the audio output of the synthesizer."""
        pass

    @abstractmethod
    def get_setter_methods(self):
        """Return a dictionary of setter methods for the synthesizer parameters."""
        pass

    def _stretch_array(self, arr:np.ndarray, target_length:int):
        old_indices = np.arange(len(arr))
        new_indices = np.linspace(0, len(arr) - 1, target_length)
        
        return np.interp(new_indices, old_indices, arr)
    
    def _apply_antialiasing(self, audio):
        
        cutoff = self.sr / 2.2
        nyquist = self.sr / 2
        normalized_cutoff = cutoff / nyquist
        
        b, a = butter(4, normalized_cutoff, btype='low')
        audio = filtfilt(b, a, audio)
        
        return audio