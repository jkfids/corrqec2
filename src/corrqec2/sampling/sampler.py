from typing import Type
import numpy as np
import sinter
from ..experiments import Experiment
from ..noisemodels import NoiseModel
from ..decoding import Decoder
from ..simulation.simulator import CircuitSimulator
from ..simulation.error_masks import ErrorMasks, ErrorMaskConverter

import time


class Sampler:
    """Orchestrates custom error generation, circuit simulation, and decoding for QEC experiments."""

    def __init__(
        self,
        experiment: Experiment,
        noise_model: NoiseModel,
        decoder: Decoder,
        marginalized_detector_error_model: bool = False,
    ):
        self.experiment = experiment
        self.noise_model = noise_model
        self.simulator = CircuitSimulator(experiment, noise_model)

        # Configure detector error model
        if marginalized_detector_error_model:
            detector_error_model = noise_model.gen_marginalized_detector_error_model(
                experiment
            )
        else:
            detector_error_model = noise_model.gen_detector_error_model(experiment)
        decoder.configure_from_detector_error_model(detector_error_model)
        self.decoder = decoder

    def sample_for_sinter(
        self,
        suggested_shots: int,
    ) -> sinter.AnonTaskStats:

        # t0 = time.perf_counter()
        error_masks = self.gen_error_masks(batch_size=suggested_shots)
        # t1 = time.perf_counter()
        detection_events, observable_flips = self.simulate_with_errors(error_masks)
        # t2 = time.perf_counter()
        predictions = self.decode_batch(detection_events)
        # t3 = time.perf_counter()
        if predictions.ndim == 1:
            predictions = predictions[:, np.newaxis]
        if observable_flips.ndim == 1:
            observable_flips = observable_flips[:, np.newaxis]
        n_shots = predictions.shape[0]
        n_errors = int(np.any(predictions != observable_flips, axis=1).sum())
        # print(
        #     f"Error sampling: {t1 - t0:.2f} s | Simulation: {t2 - t1:.2f} s | Decoding: {t3 - t2:.2f} s"
        # )

        return n_errors, n_shots

    def gen_error_masks(self, batch_size: int) -> ErrorMasks:
        """Generate error masks for a simulation batch based on the noise model.

        Args:
            batch_size: Number of parallel simulations

        Returns:
            Masks specifying where to apply Pauli errors in the circuit.
        """
        error_matrix = self.noise_model.gen_error_matrix(self.experiment, batch_size)
        error_masks = ErrorMaskConverter.error_matrix_to_error_masks(
            error_matrix, self.experiment, self.noise_model.noisy_qubit_types
        )
        return error_masks

    def simulate_with_errors(
        self, error_masks: ErrorMasks
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate a batch of shots with the given Pauli errors injected.

        Args:
            error_masks: Masks specifying where to apply Pauli errors.

        Returns:
            Detection events and observable flips for the batch.
        """

        return self.simulator.simulate_batch(error_masks)

    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode a batch of detection events with the configured decoder.

        Args:
            detection_events: Detection event bits, shape (n_shots, n_detectors).

        Returns:
            Predicted logical observable flips, shape (n_shots, n_observables).
        """
        return self.decoder.decode_batch(detection_events)
