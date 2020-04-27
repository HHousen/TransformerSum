import torch
from torch import nn

try:
    from transformers.activations import get_activation
except ImportError:
    logger.warn(
        "Could not import `get_activation` from `transformers.activations`. Only GELU will be available for use in the classifier."
    )

class LinearClassifier(nn.Module):
    def __init__(
        self,
        web_hidden_size,
        linear_hidden=1536,
        first_dropout=0.1,
        last_dropout=0.1,
        activation_string="gelu",
    ):
        """nn.Module to classify sentences by reducing the hidden dimension to 1
        
        Arguments:
            web_hidden_size {int} -- The output hidden size from the word embedding model. Used as
                                     the input to the first linear layer in this nn.Module.
        
        Keyword Arguments:
            linear_hidden {int} -- The number of hidden parameters for this Classifier. (default: {1536})
            first_dropout {float} -- The value for dropout applied before any other layers. (default: {0.1})
            last_dropout {float} -- The dropout after the last linear layer. (default: {0.1})
            activation_string {str} -- A string representing an activation function in `get_activation()` (default: {"gelu"})
        """
        super(LinearClassifier, self).__init__()
        self.dropout1 = nn.Dropout(first_dropout) if first_dropout else nn.Identity()
        self.dropout2 = nn.Dropout(last_dropout) if last_dropout else nn.Identity()
        self.linear1 = nn.Linear(web_hidden_size, linear_hidden)
        self.linear2 = nn.Linear(linear_hidden, 1)
        self.sigmoid = nn.Sigmoid()

        # support older versions of huggingface/transformers
        if activation_string == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = (
                get_activation(activation_string)
                if activation_string
                else nn.Identity()
            )

    def forward(self, x, mask):
        x = self.dropout1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.sigmoid(x)
        sent_scores = x.squeeze(-1) * mask.float()
        return sent_scores

class TransformerEncoderClassifier(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, num_layers=2, reduction=None):
        """nn.Module to classify sentences by running the sentence vectors through some
        nn.TransformerEncoder layers and then reducing the hidden dimension to 1 with a
        linear layer.

        Arguments:
            d_model {int} -- the number of expected features in the input

        Keyword Arguments:
            nhead {int} -- the number of heads in the multiheadattention models (default: {8})
            dim_feedforward {int} -- the dimension of the feedforward network model (default: {2048})
            dropout {float} -- the dropout value (default: {0.1})
            num_layers {int} -- the dropout value (default: {2})
            reduction {nn.Module} -- a nn.Module that maps `d_model` inputs to 1 value; if not specified
                                     then a `nn.Sequential()` module consisting of a linear layer and a
                                     sigmoid will automatically be created. (default: nn.Sequential(linear, sigmoid))
        """        
        super(TransformerEncoderClassifier, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=layer_norm)

        if reduction:
            self.reduction = reduction
        else:
            linear = nn.Linear(d_model, 1)
            sigmoid = nn.Sigmoid()
            self.reduction = nn.Sequential(linear, sigmoid)
    
    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.reduction(x)
        sent_scores = x.squeeze(-1) * mask.float()
        return sent_scores
