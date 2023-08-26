import json
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, Field


class TrainingConfigDefaults(BaseModel):

    data_csv_path: str = Field(
        default=str(
            Path(__file__).resolve().parents[1]
                / 'data'
                / 'raw'
                / 'articles_2018-01-02_2022-06-03.csv'
        )
    )
    output_directory: str = Field(default=None, required=True) # TODO: how to require non-None?
    overwrite_existing: bool = Field(default=False)
    model_name_or_path: str = Field(default='distilbert-base-cased')
    sentence_tokenize: bool = Field(default=False)
    checkpoint_epochs: bool = Field(default=False)
    num_epochs: int = Field(default=3)
    # dev_proportion: float = Field(default=0.025)
    batch_size: int = Field(default=8)
    max_length: int = Field(default=None)
    mlm_probability: float = Field(default=0.15)
    learning_rate: float = Field(default=5e-5)
    weight_decay: float = Field(default=0.0)
    no_decay_bias_and_layer_norm: bool = Field(default=False)
    no_gpu: bool = Field(default=False)


class TrainingConfig(TrainingConfigDefaults):

    def __init__(self, config_json_path: str | None = None, **kwargs):
        """
        Initialize training configurations. Can be done in three ways:

        First, all configuration values are set to their defaults as defined
        via the `TrainingConfigDefaults` fields.

        Then and optionally, these values can be overridden by values specified
        in either a config JSON file or else passed programmatically as kwargs.
        """
        super().__init__()
        if config_json_path:
            with open(config_json_path) as f:
                config_json = json.load(f)
            self._override_defaults(config_json)
        elif kwargs:
            self._override_defaults(kwargs)
        self.output_directory = Path(self.output_directory).resolve()
        if not self.overwrite_existing:
            assert not self.output_directory.exists() # TODO: add error message

    def _override_defaults(self, custom_configs: Dict[str, str]) -> None:
        for field, default_value in BaseModel._iter(TrainingConfigDefaults()):
            custom_value = custom_configs.get(field)
            if custom_value is not None and custom_value != default_value:
                setattr(self, field, custom_value)
                if field != 'output_directory':
                    print(
                        'Training configuration -- Overriding default '
                        f'({field}={default_value!r}): {custom_value!r}'
                    )

    # def _split()

    def write_config_json(self):
        self.output_directory.mkdir(exist_ok=True)
        with (self.output_directory / 'training_config.json').open('w') as f:
            configs = vars(self).copy()
            configs['output_directory'] = str(configs['output_directory'])
            json.dump(configs, f, indent=4, sort_keys=True)
