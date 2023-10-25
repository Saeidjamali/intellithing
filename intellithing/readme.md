# LayerFreezer Documentation

`LayerFreezer` is a utility class designed to make freezing and unfreezing layers in various Hugging Face models more intuitive and streamlined.

## Table of Contents
1. [Supported Model Types](#supported-model-types)
2. [Usage](#usage)
   - [Initialization](#initialization)
   - [Freezing and Unfreezeing](#Freeze-and-Unfreeze All Layers)
   - [Examples](#Examples)
   
3. [Notes](#notes)

## Supported Model Types

- **T5 and BART**
  - Structure: Both encoder and decoder.
  - Naming convention: `encoder.block.{index}` and `decoder.block.{index}`
- **BERT, RoBERTa, DistilBERT, and Electra**
  - Structure: Encoder only.
  - Naming convention: `encoder.layer.{index}`
- **XLNet**
  - Structure: Encoder only.
  - Naming convention: `transformer.layer.{index}`
- **GPT-2 and TransformerXL**
  - Structure: No differentiation between encoder/decoder.
  - Naming convention: `h.{index}`



## Usage

### Initialization

To initialize the `LayerFreezer` class:

```python
from layer_freezer import LayerFreezer
freezer = LayerFreezer(model_instance, model_type='model_name_here')
```

- `model_instance`: Your HuggingFace model instance.
- `model_type`: The type of the model you're using (e.g., 't5', 'bert', 'xlnet', etc.).

### Methods

#### Freeze/Unfreeze Specific Layers

To freeze or unfreeze specific layers in the model:

```python
# Freeze specific layers
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')

# Unfreeze specific layers
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
```

Parameters:
- `layer_indices`: A list of indices for the layers you want to freeze or unfreeze.
- `part`: The part of the model you're referring to - this can be 'encoder', 'decoder', or 'all'. By default, it's set to 'encoder'.

#### Freeze and Unfreeze All Layers

To freeze or unfreeze all layers in the model:

```python
# Freeze all layers
freezer.freeze_all()

# Unfreeze all layers
freezer.unfreeze_all()
```

## Examples

Here are some practical examples:

1. **Freeze all layers**:
```python
freezer.freeze_all()
```

2. **Unfreeze all layers**:
```python
freezer.unfreeze_all()
```

3. **Freeze specific layers in the encoder**:
```python
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')
```

4. **Unfreeze specific layers in the decoder**:
```python
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
```

## Note:
Please ensure the right model names are included. For example instead of "t5-base" write it as "t5"
