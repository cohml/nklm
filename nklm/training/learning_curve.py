import click
from itertools import chain
import json
import matplotlib.pyplot as plt
from typing import Dict, List


def parse_training_metrics_json(
        training_metrics_json_path: str
) -> Dict[str, List[float]]:
    with open(training_metrics_json_path) as f:
        return json.load(f)


def parse_data_coordinates(training_metrics: Dict[str, List[float]]):
    per_epoch_metrics = training_metrics['per_epoch']
    per_step_metrics = list(chain(*training_metrics['per_step']))

    n_epochs = len(per_epoch_metrics)
    n_steps_per_epoch = len(training_metrics['per_step'][0])
    n_steps = n_steps_per_epoch * n_epochs

    per_epoch_metrics_x = [n_steps_per_epoch * (i + 1) for i in range(n_epochs)]
    per_step_metrics_x = range(n_steps)

    return (
        per_epoch_metrics_x,
        per_epoch_metrics,
        per_step_metrics_x,
        per_step_metrics,
    )


@click.command()
@click.option('--training_metrics_json_path', type=str)
def main(training_metrics_json_path: str) -> None:
    training_metrics = parse_training_metrics_json(training_metrics_json_path)
    (
        per_epoch_metrics_x,
        per_epoch_metrics,
        per_step_metrics_x,
        per_step_metrics,
    ) = parse_data_coordinates(training_metrics)
    plt.plot(per_epoch_metrics_x, per_epoch_metrics, label='per epoch mean')
    plt.plot(per_step_metrics_x, per_step_metrics, label='per batch', alpha=1/3)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
