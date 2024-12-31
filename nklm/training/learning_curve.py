import click
from itertools import chain
import json
import matplotlib.pyplot as plt
from typing import Dict, List


def parse_training_losses_json(
        training_losses_json_path: str
) -> Dict[str, List[float]]:
    with open(training_losses_json_path) as f:
        return json.load(f)


def parse_data_coordinates(training_losses: Dict[str, List[float]]):
    per_epoch_train_losses = training_losses['per_epoch_train_loss']
    per_epoch_eval_losses = training_losses['per_epoch_eval_loss']
    per_step_losses = list(chain(*training_losses['per_step_train_loss']))

    n_epochs = len(per_epoch_train_losses)
    n_steps_per_epoch = len(training_losses['per_step_train_loss'][0])
    n_steps = n_steps_per_epoch * n_epochs

    per_epoch_train_losses_x = [
        n_steps_per_epoch * (i + 1) for i in range(n_epochs)
    ]
    per_step_losses_x = range(n_steps)

    return (
        per_epoch_train_losses_x,
        per_epoch_train_losses,
        per_epoch_eval_losses,
        per_step_losses_x,
        per_step_losses,
        n_epochs,
    )


@click.command()
@click.option('--training-losses-json-path', type=str)
def main(training_losses_json_path: str) -> None:

    # parse and reshape loss metrics for plottibg
    training_losses = parse_training_losses_json(training_losses_json_path)
    (
        per_epoch_train_losses_x,
        per_epoch_train_losses,
        per_epoch_eval_losses,
        per_step_losses_x,
        per_step_losses,
        n_epochs,
    ) = parse_data_coordinates(training_losses)

    # plot loss metrics
    ax = plt.subplot()
    ax.plot(per_epoch_train_losses_x, per_epoch_train_losses, label='train')
    ax.plot(per_epoch_train_losses_x, per_epoch_eval_losses, label='eval')
    ax.plot(per_step_losses_x, per_step_losses, label='batch', alpha=1/3)
    ax.set_xlim(per_step_losses_x[0], per_step_losses_x[-1])
    ax.set_xticks(per_epoch_train_losses_x, labels=range(1, n_epochs + 1))
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend()
    for x in per_epoch_train_losses_x:
        ax.axvline(x, ls='--', c='k', alpha=1/3)
    plt.show()


if __name__ == '__main__':
    main()
