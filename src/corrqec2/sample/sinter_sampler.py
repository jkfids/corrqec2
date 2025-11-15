import sinter

from .sampler import Sampler
from .sinter_compiled_sampler import SinterCompiledSampler


class SinterSampler(sinter.Sampler):
    def __init__(self, experiment_class, noise_model_class):
        self.experiment_class = experiment_class
        self.noise_model_class = noise_model_class

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:

        metadata = task.json_metadata

        # Experiment = metadata["experiment_class"]
        experiment_args = metadata["experiment_args"]

        # NoiseModel = metadata["noise_model_class"]
        noise_model_args = metadata["noise_model_args"]

        min_batch_size = metadata.get("min_batch_size", 1000)

        Experiment = self.experiment_class
        NoiseModel = self.noise_model_class

        experiment = Experiment(**experiment_args)
        noise_model = NoiseModel(**noise_model_args)

        sampler = Sampler(
            experiment=experiment,
            noise_model=noise_model,
            min_batch_size=min_batch_size,
        )

        return SinterCompiledSampler(sampler)
