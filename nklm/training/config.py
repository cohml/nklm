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
    model_name_or_path: str = Field(default='distilbert-base-cased')
    sentence_tokenize: bool = Field(default=False)
    # dev_proportion: float = Field(default=0.025)
    batch_size: float = Field(default=8)
    max_length: int = Field(default=None)
    mlm_probability: float = Field(default=0.15)


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

    def _override_defaults(self, custom_configs: Dict[str, str]) -> None:
        for field, default_value in BaseModel._iter(TrainingConfigDefaults()):
            custom_value = custom_configs.get(field)
            if custom_value is not None and custom_value != default_value:
                print(
                    'Training configuration -- Overriding default '
                    f'({field}={default_value!r}): {custom_value!r}'
                )
                setattr(self, field, custom_value)

    # def _split()
