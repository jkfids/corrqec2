import numpy as np
import stim

# from ..noisemodels.base_noise_model import NoiseModel
# from ..sample.sampler import Sampler

class Experiment:
    def __init__(self):
        pass
    
    def gen_stim_circuit(self):
        self.circuit = self._gen_stim_circuit()
        self.qubit_coords = self._get_qubit_coords()
        
        return self.circuit
        
    def _gen_stim_circuit(self):
        pass
    
    def _get_qubit_coords(self):
        pass
    
    # def compile_sampler(self, noise_model: 'NoiseModel | None') -> 'Sampler':
    #     return None
