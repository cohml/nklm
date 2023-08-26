import click

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from config import TrainingConfig
from dataset import RodongSinmunDataset
from model import get_model, get_optimizer


@click.command()
@click.option('--config-json-path', type=str)
@click.option('--data-csv-path', type=str)
@click.option('--model-name-or-path', type=str)
@click.option('--sentence-tokenize', is_flag=True)
@click.option('--num-epochs', type=int)
# @click.option('--dev-proportion', type=float)
@click.option('--batch-size', type=float)
@click.option('--max-length', type=int)
@click.option('--mlm-probability', type=float)
@click.option('--learning-rate', type=float)
@click.option('--weight-decay', type=float)
@click.option('--no-decay-bias-and-layer-norm', is_flag=True)
def main(
    config_json_path: str | None,
    data_csv_path: str | None,
    model_name_or_path: str | None,
    sentence_tokenize: bool,
    num_epochs: int | None,
    batch_size: float | None,
    max_length: int | None,
    mlm_probability: float | None,
    learning_rate: float | None,
    weight_decay: float | None,
    no_decay_bias_and_layer_norm: bool,
) -> None:

    # configs
    config = TrainingConfig(
        config_json_path,
        data_csv_path=data_csv_path,
        model_name_or_path=model_name_or_path,
        sentence_tokenize=sentence_tokenize,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_length=max_length,
        mlm_probability=mlm_probability,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        no_decay_bias_and_layer_norm=no_decay_bias_and_layer_norm,
    )
    torch.manual_seed(42)

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

    # model
    model = get_model(config)
    optimizer = get_optimizer(config, model)

if __name__ == '__main__':
    main()
