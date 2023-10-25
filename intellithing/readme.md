LayerFreezer Documentation
LayerFreezer is a utility class designed to make freezing and unfreezing layers in various Hugging Face models more intuitive and streamlined.

Table of Contents
Supported Model Types
Usage
Initialization
Freezing Layers
Unfreezing Layers
Freezing and Unfreezing All Layers
Notes
Supported Model Types
T5 and BART

Structure: Both encoder and decoder.
Naming convention: encoder.block.{index} and decoder.block.{index}
BERT, RoBERTa, DistilBERT, and Electra

Structure: Encoder only.
Naming convention: encoder.layer.{index}
XLNet

Structure: Encoder only.
Naming convention: transformer.layer.{index}
GPT-2 and TransformerXL

Structure: No differentiation between encoder/decoder.
Naming convention: h.{index}
Usage
Initialization
To start, initialize the LayerFreezer with the model and its type:

python
Copy code
freezer = LayerFreezer(model, model_type='bert')
Freezing Layers
For models with both encoder and decoder, use:

python
Copy code
freezer.freeze_layers([0, 1, 2], part='encoder')
freezer.freeze_layers([10, 11], part='decoder')
For models with just an encoder:

python
Copy code
freezer.freeze_layers([0, 1, 2], part='encoder')
Unfreezing Layers
For models with both encoder and decoder:

python
Copy code
freezer.unfreeze_layers([3, 4], part='encoder')
freezer.unfreeze_layers([9], part='decoder')
For encoder-only models:

python
Copy code
freezer.unfreeze_layers([3, 4], part='encoder')
Freezing and Unfreezing All Layers
To freeze all layers:

python
Copy code
freezer.freeze_all()
To unfreeze all layers:

python
Copy code
freezer.unfreeze_all()
Notes
Ensure you specify the correct model_type during initialization. This helps the class correctly identify the layers.
For models with both encoder and decoder, like T5 and BART, ensure you specify the part parameter as encoder or decoder when needed.
