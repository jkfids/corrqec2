import pymatching

from .base_decoder import Decoder


class Pymatching(Decoder):
    def __init__(self, enable_correlations=False):
        self.enable_correlations = enable_correlations
        self.matcher = None

    def configure_from_detector_error_model(self, detector_error_model):
        matcher = pymatching.Matching.from_detector_error_model(
            detector_error_model, enable_correlations=self.enable_correlations
        )
        self.matcher = matcher

    def decode_batch(self, detection_events):
        predictions = self.matcher.decode_batch(detection_events).flatten()
        return predictions
