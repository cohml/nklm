import click

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from config import TrainingConfig
from dataset import RodongSinmunDataset


@click.command()
@click.option('--config_json_path', default=None)
def main(config_json_path: str | None) -> None:
    torch.manual_seed(42)
    config = TrainingConfig(config_json_path)

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

    breakpoint()


if __name__ == '__main__':
    main()
