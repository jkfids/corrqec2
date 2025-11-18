import stim
import sinter
from sinter import Task
from ..experiments import Experiment
from ..noisemodels import NoiseModel
from ..decoding import Decoder


def _gen_task_metadata(
    experiment: str | type[Experiment],
    experiment_args: dict,
    noise_model: str | type[NoiseModel],
    noise_model_args: dict,
    decoder: str | type[Decoder],
    decoder_args: dict | None = None,
    min_batch_size: int = 1000,
    marginalized_detector_error_model: bool = False,
) -> dict:
    """Generate metadata dictionary for a Sinter task."""
    if isinstance(experiment, type) and issubclass(experiment, Experiment):
        experiment = experiment.__name__
    if isinstance(noise_model, type) and issubclass(noise_model, NoiseModel):
        noise_model = noise_model.__name__
    if isinstance(decoder, type) and issubclass(decoder, Decoder):
        decoder = decoder.__name__
    return {
        "experiment": experiment,
        "experiment_args": experiment_args,
        "noise_model": noise_model,
        "noise_model_args": noise_model_args,
        "decoder": decoder,
        "decoder_args": decoder_args or {},
        "min_batch_size": min_batch_size,
        "marginalized_detector_error_model": marginalized_detector_error_model,
    }


def _task_from_metadata(
    metadata: dict,
) -> sinter.Task:
    """Create a Sinter task from metadata dictionary."""
    return Task(circuit=stim.Circuit(), json_metadata=metadata)


def create_task(
    experiment: str | type[Experiment],
    experiment_args: dict,
    noise_model: str | type[NoiseModel],
    noise_model_args: dict,
    decoder: str | type[Decoder],
    decoder_args: dict | None = None,
    min_batch_size: int = 1000,
    marginalized_detector_error_model: bool = False,
) -> sinter.Task:
    """Create a Sinter task with the given experiment, noise model, and decoder.

    Args:
        experiment: Experiment class or name.
        experiment_args: Arguments to initialize the experiment.
        noise_model: NoiseModel class or name.
        noise_model_args: Arguments to initialize the noise model.
        decoder: Decoder class or name.
        decoder_args: Arguments to initialize the decoder.
        min_batch_size: Minimum batch size for sampling.
        marginalized_detector_error_model: Whether to use marginalized detector error model.

    Returns:
        sinter.Task: The created Sinter task.
    """
    metadata = _gen_task_metadata(
        experiment=experiment,
        experiment_args=experiment_args,
        noise_model=noise_model,
        noise_model_args=noise_model_args,
        decoder=decoder,
        decoder_args=decoder_args,
        min_batch_size=min_batch_size,
        marginalized_detector_error_model=marginalized_detector_error_model,
    )
    return _task_from_metadata(metadata)


def create_tasks_sweep_experiment_args(
    experiment_args_sweep: list[dict],
    experiment: str | type[Experiment],
    noise_model: str | type[NoiseModel],
    noise_model_args: dict,
    decoder: str | type[Decoder],
    decoder_args: dict | None = None,
    min_batch_size: int = 1000,
    marginalized_detector_error_model: bool = False,
) -> list[sinter.Task]:
    """Create a list of Sinter tasks by sweeping over experiment arguments.

    Args:
        experiment_args_sweep (list[dict]): _description_
    """
    tasks = []
    for experiment_args in experiment_args_sweep:
        task = create_task(
            experiment=experiment,
            experiment_args=experiment_args,
            noise_model=noise_model,
            noise_model_args=noise_model_args,
            decoder=decoder,
            decoder_args=decoder_args,
            min_batch_size=min_batch_size,
            marginalized_detector_error_model=marginalized_detector_error_model,
        )
        tasks.append(task)
    return tasks


def create_tasks_sweep_noise_model_args(
    noise_model_args_sweep: list[dict],
    experiment: str | type[Experiment],
    experiment_args: dict,
    noise_model: str | type[NoiseModel],
    decoder: str | type[Decoder],
    decoder_args: dict | None = None,
    min_batch_size: int = 1000,
    marginalized_detector_error_model: bool = False,
) -> list[sinter.Task]:
    """Create a list of Sinter tasks by sweeping over noise model arguments.

    Args:
        noise_model_args_sweep (list[dict]): _description_
    """
    tasks = []
    for noise_model_args in noise_model_args_sweep:
        task = create_task(
            experiment=experiment,
            experiment_args=experiment_args,
            noise_model=noise_model,
            noise_model_args=noise_model_args,
            decoder=decoder,
            decoder_args=decoder_args,
            min_batch_size=min_batch_size,
            marginalized_detector_error_model=marginalized_detector_error_model,
        )
        tasks.append(task)
    return tasks
