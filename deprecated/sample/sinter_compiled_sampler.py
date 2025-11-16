import sinter

from .sampler import Sampler


class SinterCompiledSampler(sinter.CompiledSampler):
    def __init__(self, sampler):
        self._sampler = sampler

    def sample(
        self,
        suggested_shots: int,
    ) -> sinter.AnonTaskStats:

        error_masks = self._sampler.sample_error_masks(batch_size=suggested_shots)
        detection_events, observable_flips = self._sampler.simulate_circuit(error_masks)
        n_errors = self._sampler.decode_count_errors(detection_events, observable_flips)
        n_shots = len(observable_flips)

        return sinter.AnonTaskStats(shots=n_shots, errors=n_errors)
