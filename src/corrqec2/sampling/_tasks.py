import argparse
from pathlib import Path
import stim
import sinter
from sinter import Task
from ..experiments import Experiment
from ..noisemodels import NoiseModel
from ..decoding import Decoder
from ..sampling import SinterSampler, save_stats_to_csv


def _gen_task_metadata(
    experiment: str | type[Experiment],
    experiment_args: dict,
    noise_model: str | type[NoiseModel],
    noise_model_args: dict,
    decoder: str | type[Decoder],
    decoder_args: dict | None = None,
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
        marginalized_detector_error_model=marginalized_detector_error_model,
    )
    return _task_from_metadata(metadata)


def run_tasks_to_csv(
    tasks: list[sinter.Task],
    n_workers: int,
    max_shots: int,
    min_batch_size: int,
    output_dir: Path,
    filename: str,
    print_progress: bool = True,
):
    """Run Sinter tasks and save results to a CSV file.

    Args:
        tasks (list[sinter.Task]): _description_
        n_workers (int): _description_
        max_shots (int): _description_
        output_dir (Path): _description_
        filename (str): _description_
        print_progress (bool, optional): _description_. Defaults to True.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if filename.endswith(".csv"):
        filename = filename[:-4]
    resume_path = output_dir / f"{filename}.csv.tmp"
    output_path = output_dir / f"{filename}.csv"

    if resume_path.exists() and print_progress:
        print(f"Resuming from existing file: {resume_path}")

    stats = run_tasks(
        tasks=tasks,
        n_workers=n_workers,
        max_shots=max_shots,
        min_batch_size=min_batch_size,
        save_resume_filepath=resume_path,
        print_progress=print_progress,
    )

    # Save results to CSV
    save_stats_to_csv(stats, output_path)
    if resume_path.exists():
        resume_path.unlink()

    if print_progress:
        print("Completed all tasks!")
        print(f"Results saved to {output_path}")


def run_tasks(
    tasks: list[sinter.Task],
    n_workers: int,
    max_shots: int,
    min_batch_size: int = 1000,
    save_resume_filepath: Path | None = None,
    print_progress: bool = False,
):
    """Run Sinter tasks using sinter.collect().

    Args:
        tasks (list[sinter.Task]): _description_
        n_workers (int): _description_
        max_shots (int): _description_
        save_resume_filepath (Path | None, optional): _description_. Defaults to None.
        print_progress (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    sampler = SinterSampler(
        min_batch_size=min_batch_size, print_progress=print_progress
    )
    stats = sinter.collect(
        tasks=tasks,
        num_workers=n_workers,
        decoders="custom_sampler",
        custom_decoders={"custom_sampler": sampler},
        max_shots=max_shots,
        save_resume_filepath=save_resume_filepath,
        print_progress=False,  # Disable sinter's own progress printing
    )

    return stats


def get_default_parser() -> argparse.ArgumentParser:
    """Get argument parser for experiments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=100,
        help="Maximum shots per task",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Minimum batch size for each worker",
    )

    return parser
