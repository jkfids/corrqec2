from typing import Type
import numpy as np
from ..experiments import Experiment
from ..noisemodels import NoiseModel
from ..decoding import Decoder
from ..simulation.simulator import CircuitSimulator
from ..simulation.error_masks import ErrorMasks, ErrorMaskConverter


class Sampler:
    """Orchestrates custom error generation, circuit simulation, and decoding for QEC experiments."""

    def __init__(
        self,
        experiment: Experiment,
        noise_model: NoiseModel,
        decoder: Decoder,
    ):
        self.experiment = experiment
        self.noise_model = noise_model
        self.simulator = CircuitSimulator(experiment, noise_model)

        # Configure detector error model
        detector_error_model = noise_model.gen_detector_error_model(experiment)
        decoder.configure_from_detector_error_model(detector_error_model)
        self.decoder = decoder

    def gen_error_masks(self, batch_size: int) -> ErrorMasks:
        """_summary_

        Args:
            batch_size (int): _description_

        Returns:
            ErrorMasks: _description_
        """
        error_matrix = self.noise_model.gen_error_matrix(self.experiment, batch_size)
        error_masks = ErrorMaskConverter.error_matrix_to_error_masks(
            error_matrix, self.experiment, self.noise_model.noisy_qubit_types
        )
        return error_masks

    def simulate_with_errors(
        self, error_masks: ErrorMasks
    ) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            error_masks (ErrorMasks): _description_

        Returns:
            tuple[np.ndarray, np.ndarray]: _description_
        """

        return self.simulator.simulate_batch(error_masks)

    def decode_errors(self, detection_events: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            detection_events (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        return self.decoder.decode_batch(detection_events)
