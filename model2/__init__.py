 
# This file makes the 'model2' directory a Python package.
# You can also use it to make imports cleaner when using the package.

# Import key classes to be directly available when importing 'model2'
# e.g., from model2 import InnerSpeech_LBLM_MSTP

from .lblm_mstp_model import InnerSpeech_LBLM_MSTP
from .lblm_input_processor import LBLMInputProcessor
from .lgtransformer import (
    LBLMConformerBackbone,
    LayerGatedConformerBlock,
    ConvolutionModule,
    FeedForwardModule,
    MultiHeadSelfAttentionModule,
    ZeroConv1d,
    Swish
)
from .lblm_prediction_heads import SpectroTemporalPredictionHeads
from .lblm_loss import HuberLoss
from .st_classifier import STClassifier

# You can define an __all__ variable to specify what gets imported with 'from model2 import *'
# This is generally good practice if you use 'import *'.
__all__ = [
    "InnerSpeech_LBLM_MSTP",
    "LBLMInputProcessor",
    "LBLMConformerBackbone",
    "LayerGatedConformerBlock",
    "ConvolutionModule",
    "FeedForwardModule",
    "MultiHeadSelfAttentionModule",
    "ZeroConv1d",
    "Swish",
    "SpectroTemporalPredictionHeads",
    "HuberLoss",
    "STClassifier"
]

print("Package 'model2' initialized.") # Optional: for confirmation