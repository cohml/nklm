import click
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from config import TrainingConfig, TrainingConfigDefaults
from dataset import RodongSinmunDataset
from model import get_model, get_optimizer


DEFAULTS = TrainingConfigDefaults()


@click.command()
@click.option(
    '--config-json-path', type=str, help='Default: None'
)
@click.option(
    '--output-directory', type=str, help='Default: None'
)
@click.option(
    '--data-csv-path', type=str, help=f'Default: {DEFAULTS.data_csv_path}'
)
@click.option(
    '--overwrite-existing', is_flag=True, help=f'Default: {DEFAULTS.overwrite_existing}'
)
@click.option(
    '--model-name-or-path', type=str, help=f'Default: {DEFAULTS.model_name_or_path}'
)
@click.option(
    '--sentence-tokenize', is_flag=True, help=f'Default: {DEFAULTS.sentence_tokenize}'
)
@click.option(
    '--checkpoint-epochs', is_flag=True, help=f'Default: {DEFAULTS.checkpoint_epochs}'
)
@click.option(
    '--num-epochs', type=int, help=f'Default: {DEFAULTS.num_epochs}'
)
# @click.option(
# '--dev-proportion', type=float, help=f'Default: {DEFAULTS.dev_proportion}'
# )
@click.option(
    '--batch-size', type=int, help=f'Default: {DEFAULTS.batch_size}'
)
@click.option(
    '--max-length', type=int, help=f'Default: {DEFAULTS.max_length}'
)
@click.option(
    '--mlm-probability', type=float, help=f'Default: {DEFAULTS.mlm_probability}'
)
@click.option(
    '--learning-rate', type=float, help=f'Default: {DEFAULTS.learning_rate}'
)
@click.option(
    '--weight-decay', type=float, help=f'Default: {DEFAULTS.weight_decay}'
)
@click.option(
    '--no-decay-bias-and-layer-norm', is_flag=True, help=f'Default: {DEFAULTS.no_decay_bias_and_layer_norm}'
)
@click.option(
    '--no-gpu', is_flag=True, help=f'Default: {DEFAULTS.no_gpu}'
)
@click.option(
    '--seed', type=int, help=f'Default: {DEFAULTS.seed}'
)
def main(
    config_json_path: str | None,
    output_directory: str | None,
    data_csv_path: str | None,
    overwrite_existing: bool,
    model_name_or_path: str | None,
    sentence_tokenize: bool,
    checkpoint_epochs: bool,
    num_epochs: int | None,
    batch_size: float | None,
    max_length: int | None,
    mlm_probability: float | None,
    learning_rate: float | None,
    weight_decay: float | None,
    no_decay_bias_and_layer_norm: bool,
    no_gpu: bool,
    seed: int,
) -> None:

    # configs
    config = TrainingConfig(
        config_json_path,
        output_directory=output_directory,
        data_csv_path=data_csv_path,
        overwrite_existing=overwrite_existing,
        model_name_or_path=model_name_or_path,
        sentence_tokenize=sentence_tokenize,
        checkpoint_epochs=checkpoint_epochs,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        mlm_probability=mlm_probability,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        no_decay_bias_and_layer_norm=no_decay_bias_and_layer_norm,
        no_gpu=no_gpu,
        seed=seed,
    )
    torch.manual_seed(config.seed)

    # data
    dataset = RodongSinmunDataset(config)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=dataset.tokenizer, mlm_probability=config.mlm_probability
    )
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=config.batch_size,
        collate_fn=data_collator,
    )
    config.write_config_json()

    # model
    model = get_model(config)
    optimizer = get_optimizer(config, model)

    # set device
    if not config.no_gpu and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # train
    print('***** Running training *****')
    model = model.to(device)
    n_batches = len(dataloader)
    per_epoch_mean_losses = []
    per_epoch_batch_losses = []
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_total_loss = 0

        per_step_losses = []
        for step, batch in tqdm(
            enumerate(dataloader),
            desc=f'Epoch {epoch}',
            total=n_batches,
            unit=' batches',
        ):

            # predict
            batch = batch.to(device)
            output = model(**batch)

            # compute and backpropogate loss
            loss = output.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss = loss.detach().float()
            per_step_losses.append(loss.item())
            epoch_total_loss += loss

        epoch_mean_loss = (epoch_total_loss / n_batches).item()
        per_epoch_batch_losses.append(per_step_losses)
        per_epoch_mean_losses.append(epoch_mean_loss)
        print(f'Epoch {epoch} Mean train loss: {epoch_mean_loss}')

        # save model checkpoint
        if checkpoint_epochs or epoch == config.num_epochs:
            checkpoint_directory = config.output_directory / f'epoch-{epoch}'
            model.save_pretrained(checkpoint_directory)
            print(f'Model state saved to {checkpoint_directory}')

    # save metrics
    with (config.output_directory / 'training_metrics.json').open('w') as f:
        metrics = {
            'per_epoch': per_epoch_mean_losses,
            'per_step': per_epoch_batch_losses,
        }
        json.dump(metrics, f, indent=4)

    print('***** Training complete *****')


if __name__ == '__main__':
    main()
