# LayerFreezer Documentation

`LayerFreezer` is a utility class designed to make freezing and unfreezing layers in various Hugging Face models more intuitive and streamlined.

## Table of Contents
1. [Supported Model Types](#supported-model-types)
2. [Usage](#usage)
   - [Initialization](#initialization)
   - [Freezing Layers](#freezing-layers)
   - [Unfreezing Layers](#unfreezing-layers)
   - [Freezing and Unfreezing All Layers](#freezing-and-unfreezing-all-layers)
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

To start, initialize the LayerFreezer with the model and its type:

```python
freezer = LayerFreezer(model, model_type='bert')
