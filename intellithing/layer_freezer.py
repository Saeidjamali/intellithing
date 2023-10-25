import torch.nn as nn

class LayerFreezer:
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type

    def _get_layer_name(self, idx, part='encoder'):
        if self.model_type in ["t5", "bart"]:
            return f"{part}.block.{idx}"
        elif self.model_type in ["bert", "roberta", "distilbert", "electra"]:
            return f"{part}.layer.{idx}"  # Note: these models don't have a decoder
        elif self.model_type == "xlnet":
            return f"transformer.layer.{idx}"
        elif self.model_type in ["gpt2", "transformerxl"]:
            return f"h.{idx}"  # These models don't differentiate between encoder/decoder
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def freeze_layers(self, layer_indices, part='encoder'):
        if part == 'all':
            parts = ['encoder', 'decoder'] if self.model_type in ["t5", "bart"] else ['encoder']
        else:
            parts = [part]

        for part in parts:
            for idx in layer_indices:
                layer_name = self._get_layer_name(idx, part)
                for name, param in self.model.named_parameters():
                    if layer_name in name:
                        param.requires_grad = False

    def unfreeze_layers(self, layer_indices, part='encoder'):
        if part == 'all':
            parts = ['encoder', 'decoder'] if self.model_type in ["t5", "bart"] else ['encoder']
        else:
            parts = [part]

        for part in parts:
            for idx in layer_indices:
                layer_name = self._get_layer_name(idx, part)
                for name, param in self.model.named_parameters():
                    if layer_name in name:
                        param.requires_grad = True

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True
