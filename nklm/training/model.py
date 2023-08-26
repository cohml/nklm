from torch.optim import AdamW
from transformers import AutoModelForMaskedLM

from config import TrainingConfig


def get_model(config: TrainingConfig) -> AutoModelForMaskedLM:
    return AutoModelForMaskedLM.from_pretrained(config.model_name_or_path)


def get_optimizer(config: TrainingConfig, model: AutoModelForMaskedLM):
    if config.no_decay_bias_and_layer_norm:
        return AdamW(model.parameters(), lr=config.learning_rate)
    else:
        no_decay_parameter_names = ('bias', 'LayerNorm.weight')
        no_decay = []
        yes_decay = []
        for name, parameter in model.named_parameters():
            if name.endswith(no_decay_parameter_names):
                no_decay.append(parameter)
            else:
                yes_decay.append(parameter)
        optimizer_grouped_parameters = [
            {'params': no_decay, 'weight_decay': 0.0},
            {'params': yes_decay, 'weight_decay': config.weight_decay},
        ]
        return AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
