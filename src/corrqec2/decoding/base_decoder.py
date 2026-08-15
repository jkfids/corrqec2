from abc import ABC, abstractmethod
import numpy as np
import stim


class Decoder(ABC):
    """Abstract base class for QEC decoders."""

    @abstractmethod
    def configure_from_detector_error_model(self, dem: stim.DetectorErrorModel):
        """Configure the decoder instance from the detector error model."""
        pass

    @abstractmethod
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode detection events to predict logical observable flips.

        Args:
            detection_events: Detection event bits, shape (n_shots, n_detectors).

        Returns:
            Predicted logical observable flips, shape (n_shots, n_observables).
        """
        pass
