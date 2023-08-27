import click
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

from config import TrainingConfig, TrainingConfigDefaults
from dataset import RodongSinmunDataset
from model import get_model, get_optimizer
from utils import validate_proportion


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
@click.option(
    '--do-eval', is_flag=True, help=f'Default: {DEFAULTS.do_eval}'
)
@click.option(
    '--test-proportion', type=float, help=f'Default: {DEFAULTS.test_proportion}', callback=validate_proportion
)
@click.option(
    '--train-batch-size', type=int, help=f'Default: {DEFAULTS.train_batch_size}'
)
@click.option(
    '--test-batch-size', type=int, help=f'Default: {DEFAULTS.test_batch_size}'
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
    do_eval: bool,
    test_proportion: float | None,
    train_batch_size: float | None,
    test_batch_size: float | None,
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
        do_eval=do_eval,
        test_proportion=test_proportion,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
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
    train_dataloader = DataLoader(
        dataset.examples['train'],
        shuffle=True,
        batch_size=config.train_batch_size,
        collate_fn=data_collator,
    )
    n_train_batches = len(train_dataloader)
    if config.do_eval:
        eval_dataloader = DataLoader(
            dataset.examples['test'],
            shuffle=True,
            batch_size=config.test_batch_size,
            collate_fn=data_collator,
        )
        n_eval_batches = len(eval_dataloader)
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
    per_epoch_mean_train_losses = []
    per_epoch_batch_train_losses = []
    per_epoch_mean_eval_losses = []
    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_total_train_loss = 0

        per_step_losses = []
        for train_step, train_batch in tqdm(
            enumerate(train_dataloader),
            desc=f'Epoch {epoch} (train)',
            total=n_train_batches,
            unit=' batches',
        ):

            # predict
            train_batch = train_batch.to(device)
            output = model(**train_batch)

            # compute and backpropogate loss
            train_loss = output.loss
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss = train_loss.detach().float()
            per_step_losses.append(train_loss.item())
            epoch_total_train_loss += train_loss

        # evaluate
        if config.do_eval:
            model.eval()
            epoch_total_eval_loss = 0

            for eval_step, eval_batch in tqdm(
                enumerate(eval_dataloader),
                desc=f'Epoch {epoch} (eval)',
                total=n_eval_batches,
                unit=' batches',
            ):

                # predict
                eval_batch = eval_batch.to(device)
                with torch.no_grad():
                    output = model(**eval_batch)

                # compute loss
                eval_loss = output.loss
                eval_loss = eval_loss.detach().float()
                per_step_losses.append(eval_loss.item())
                epoch_total_eval_loss += eval_loss

        # show epoch results
        epoch_mean_train_loss = (epoch_total_train_loss / n_train_batches).item()
        per_epoch_batch_train_losses.append(per_step_losses)
        per_epoch_mean_train_losses.append(epoch_mean_train_loss)
        print(f'Epoch {epoch} mean train loss: {epoch_mean_train_loss}')
        if config.do_eval:
            epoch_mean_eval_loss = (epoch_total_eval_loss / n_eval_batches).item()
            per_epoch_mean_eval_losses.append(epoch_mean_eval_loss)
            print(f'Epoch {epoch} mean eval loss: {epoch_mean_eval_loss}')

        # save model checkpoint
        if checkpoint_epochs or epoch == config.num_epochs:
            checkpoint_directory = config.output_directory / f'epoch-{epoch}'
            model.save_pretrained(checkpoint_directory)
            print(f'Model state saved to {checkpoint_directory}')

    # save metrics
    metrics = {
        'per_epoch_train_loss': per_epoch_mean_train_losses,
        'per_step_train_loss': per_epoch_batch_train_losses,
    }
    if config.do_eval:
        metrics['per_epoch_eval_loss'] = per_epoch_mean_eval_losses
    with (config.output_directory / 'training_metrics.json').open('w') as f:
        json.dump(metrics, f, indent=4)

    print('***** Training complete *****')


if __name__ == '__main__':
    main()
