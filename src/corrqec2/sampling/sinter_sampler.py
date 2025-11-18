import time
import sinter
from .sampler import Sampler
from ..experiments import SurfaceCodeMemory, SurfaceCodeStability
from ..noisemodels import StandardCircuitLevel, StormModel
from ..decoding import Pymatching

EXPERIMENTS = {
    "SurfaceCodeMemory": SurfaceCodeMemory,
    "SurfaceCodeStability": SurfaceCodeStability,
}

NOISE_MODELS = {
    "StandardCircuitLevel": StandardCircuitLevel,
    "StormModel": StormModel,
}

DECODERS = {
    "Pymatching": Pymatching,
}


class SinterSampler(sinter.Sampler):

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:

        metadata = task.json_metadata

        experiment = metadata["experiment"]
        experiment_args = metadata["experiment_args"]
        noise_model = metadata["noise_model"]
        noise_model_args = metadata["noise_model_args"]
        decoder = metadata["decoder"]
        decoder_args = metadata.get("decoder_args", {})

        marginalized_dem = metadata.get("marginalized_detector_error_model", None)
        min_batch_size = metadata.get("min_batch_size", 1000)

        experiment = EXPERIMENTS[experiment](**experiment_args)
        noise_model = NOISE_MODELS[noise_model](**noise_model_args)
        decoder = DECODERS[decoder](**decoder_args)

        sampler = Sampler(experiment, noise_model, decoder, marginalized_dem)

        return SinterCompiledSampler(sampler, min_batch_size)


class SinterCompiledSampler(sinter.CompiledSampler):
    def __init__(self, sampler: Sampler, min_batch_size: int):
        self.sampler = sampler
        self.min_batch_size = min_batch_size

    def sample(
        self,
        suggested_shots: int,
    ) -> sinter.AnonTaskStats:

        suggested_shots = max(suggested_shots, self.min_batch_size)

        start_time = time.perf_counter()
        n_errors, n_shots = self.sampler.sample_for_sinter(suggested_shots)
        elapsed_time = time.perf_counter() - start_time

        return sinter.AnonTaskStats(
            shots=n_shots, errors=n_errors, seconds=elapsed_time
        )
