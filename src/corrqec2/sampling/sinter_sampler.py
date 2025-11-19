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

    def __init__(self, print_progress: bool = False):
        self.print_progress = print_progress
        super().__init__()

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

        return SinterCompiledSampler(sampler, min_batch_size, self.print_progress)


class SinterCompiledSampler(sinter.CompiledSampler):
    def __init__(
        self, sampler: Sampler, min_batch_size: int, print_progress: bool = False
    ):
        self.sampler = sampler
        self.min_batch_size = min_batch_size
        self.print_progress = print_progress

    def sample(
        self,
        suggested_shots: int,
    ) -> sinter.AnonTaskStats:

        # suggested_shots = max(suggested_shots, self.min_batch_size)
        suggested_shots = self.min_batch_size

        start_time = time.perf_counter()
        n_errors, n_shots = self.sampler.sample_for_sinter(suggested_shots)
        elapsed_time = time.perf_counter() - start_time

        if self.print_progress:
            experiment_name = self.sampler.experiment.__class__.__name__
            distance = self.sampler.experiment.distance
            print(
                f"{experiment_name} (distance {distance}): Sampled {n_shots} shots with {n_errors} errors in {elapsed_time:.2f} seconds",
                flush=True,
            )

        return sinter.AnonTaskStats(
            shots=n_shots, errors=n_errors, seconds=elapsed_time
        )
