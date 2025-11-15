from abc import ABC, abstractmethod
import numpy as np
import stim


class Decoder(ABC):
    """Abstract base class for QEC decoders."""

    @abstractmethod
    def configure_from_detector_error_model(dem: stim.DetectorErrorModel):
        """Configure the decoder instance from the detector error model."""
        pass

    @abstractmethod
    def decode_batch(self, detection_events: np.ndarray) -> np.ndarray:
        """Decode detection events to predict logical observable flips.

        Args:
            detection_events (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        pass
