import numpy as np
import stim

from .base_noise_model import NoiseModel

class StandardCircuitLevel(NoiseModel):
    def __init__(self, p: float, noisy_qubit_types: str | list[str] = 'all'):
        gate_noise = {
            'after_identity_depolarization': p,
            'after_clifford_depolarization': p,
            'before_measure_flip_probability': p,
            'after_reset_flip_probability': p,
        }
        super().__init__(gate_noise=gate_noise, noisy_qubit_types=noisy_qubit_types)
        self._no_error_matrix = True
