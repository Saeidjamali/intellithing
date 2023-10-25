LayerFreezer Documentation
LayerFreezer is a utility class designed to assist in freezing and unfreezing layers of popular models from the Hugging Face library.

Supported Model Types:
t5: T5 model which contains both encoder and decoder blocks.
bart: BART model which contains both encoder and decoder blocks.
bert: BERT model which only contains an encoder.
roberta: RoBERTa model which only contains an encoder.
distilbert: DistilBERT model which only contains an encoder.
electra: ELECTRA model which only contains an encoder.
xlnet: XLNet model which can be considered as an encoder.
gpt2: GPT-2 model which doesn't differentiate between encoder and decoder.
transformerxl: TransformerXL model which doesn't differentiate between encoder and decoder.
Initialization:
python
Copy code
freezer = LayerFreezer(model_instance, model_type='model_name_here')
model_instance: The model instance you want to manipulate.
model_type: The type of model you're using (e.g., 't5', 'bert'). This must be specified during initialization.
Methods:
freeze_layers(layer_indices, part='encoder'):

Freezes the specified layers.
layer_indices: A list of layer indices you want to freeze.
part: Can be 'encoder', 'decoder', or 'all'. It specifies which part of the model the layers belong to. Defaults to 'encoder'.
unfreeze_layers(layer_indices, part='encoder'):

Unfreezes the specified layers.
layer_indices: A list of layer indices you want to unfreeze.
part: Can be 'encoder', 'decoder', or 'all'. It specifies which part of the model the layers belong to. Defaults to 'encoder'.
freeze_all(): Freezes all layers of the model.

unfreeze_all(): Unfreezes all layers of the model.

Examples:
Freezing all layers of a model:
python
Copy code
freezer.freeze_all()
Unfreezing all layers of a model:
python
Copy code
freezer.unfreeze_all()
Freezing specific layers of a model's encoder:
python
Copy code
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')
Unfreezing specific layers of a model's decoder:
python
Copy code
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
Freezing all encoder layers and unfreezing specific decoder layers:
python
Copy code
freezer.freeze_layers(list(range(12)), part='encoder')  # Assuming 12 layers in encoder
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
You can expand upon this basic documentation as needed. For more advanced documentation techniques, consider using tools like Sphinx which can generate documentation based on Python docstrings.




User
can you write the doc  in a readme.md format
ChatGPT
Of course! Here's a README.md format for the LayerFreezer class:

LayerFreezer
LayerFreezer is a utility class designed to facilitate freezing and unfreezing layers of popular models from the Hugging Face library.

Supported Model Types
T5 and BART (t5, bart): Models that contain both encoder and decoder blocks.
BERT, RoBERTa, DistilBERT, and ELECTRA (bert, roberta, distilbert, electra): Models that only contain an encoder.
XLNet (xlnet): Model which can be considered as an encoder.
GPT-2 and TransformerXL (gpt2, transformerxl): Models that don't differentiate between encoder and decoder.
Usage
Initialization
python
Copy code
from layer_freezer import LayerFreezer
freezer = LayerFreezer(model_instance, model_type='model_name_here')
model_instance: Your model instance.
model_type: Specify the type of model (e.g., 't5', 'bert').
Methods
Freeze/Unfreeze Specific Layers
python
Copy code
# Freeze layers
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')

# Unfreeze layers
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
layer_indices: List of layer indices to freeze/unfreeze.
part: Specify the part of the model - 'encoder', 'decoder', or 'all'. Defaults to 'encoder'.
Freeze/Unfreeze All Layers
python
Copy code
# Freeze all layers
freezer.freeze_all()

# Unfreeze all layers
freezer.unfreeze_all()
Examples
Freeze all layers
python
Copy code
freezer.freeze_all()
Unfreeze all layers
python
Copy code
freezer.unfreeze_all()
Freeze specific encoder layers
python
Copy code
layers_to_freeze = [0, 1, 2]
freezer.freeze_layers(layers_to_freeze, part='encoder')
Unfreeze specific decoder layers
python
Copy code
layers_to_unfreeze = [10, 11]
freezer.unfreeze_layers(layers_to_unfreeze, part='decoder')
