import logging
import sys

import torch
from packaging import version
from torch import nn

logger = logging.getLogger(__name__)

try:
    from transformers.activations import get_activation
except ImportError:
    logger.warning(
        "Could not import `get_activation` from `transformers.activations`. Only GELU will be "
        + "available for use in the classifier."
    )


class LinearClassifier(nn.Module):
    """``nn.Module`` to classify sentences by reducing the hidden dimension to 1.

    Arguments:
        web_hidden_size (int): The output hidden size from the word embedding model. Used as
            the input to the first linear layer in this nn.Module.
        linear_hidden (int, optional): The number of hidden parameters for this Classifier.
            Default is 1536.
        dropout (float, optional): The value for dropout applied before the 2nd linear layer.
            Default is 0.1.
        activation_string (str, optional): A string representing an activation function
            in ``get_activation()`` Default is "gelu".
    """

    def __init__(
        self,
        web_hidden_size,
        linear_hidden=1536,
        dropout=0.1,
        activation_string="gelu",
    ):
        super(LinearClassifier, self).__init__()
        self.dropout1 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.linear1 = nn.Linear(web_hidden_size, linear_hidden)
        self.linear2 = nn.Linear(linear_hidden, 1)
        # self.sigmoid = nn.Sigmoid()

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
        """
        Forward function. ``x`` is the input ``sent_vector`` tensor and ``mask`` avoids computations
        on padded values. Returns ``sent_scores``.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        # x = self.sigmoid(x)
        sent_scores = x.squeeze(-1) * mask.float()
        sent_scores[sent_scores == 0] = -9e3
        return sent_scores


class SimpleLinearClassifier(nn.Module):
    """``nn.Module`` to classify sentences by reducing the hidden dimension to 1. This module
    contains a single linear layer and a sigmoid.

    Arguments:
        web_hidden_size (int): The output hidden size from the word embedding model. Used as
            the input to the first linear layer in this nn.Module.
    """

    def __init__(self, web_hidden_size):
        super(SimpleLinearClassifier, self).__init__()
        self.linear = nn.Linear(web_hidden_size, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        """
        Forward function. ``x`` is the input ``sent_vector`` tensor and ``mask`` avoids computations
        on padded values. Returns ``sent_scores``.
        """
        x = self.linear(x).squeeze(-1)
        # x = self.sigmoid(x)
        sent_scores = x * mask.float()
        sent_scores[sent_scores == 0] = -9e3
        return sent_scores


class TransformerEncoderClassifier(nn.Module):
    r"""
    ``nn.Module`` to classify sentences by running the sentence vectors through some
    ``nn.TransformerEncoder`` layers and then reducing the hidden dimension to 1 with a
    linear layer.

    Arguments:
        d_model (int): The number of expected features in the input
        nhead (int, optional): The number of heads in the multiheadattention models. Default is 8.
        dim_feedforward (int, optional): The dimension of the feedforward network model.
            Default is 2048.
        dropout (float, optional): The dropout value. Default is 0.1.
        num_layers (int, optional): The number of ``TransformerEncoderLayer``\ s. Default is 2.
        reduction (nn.Module, optional): a nn.Module that maps `d_model` inputs to 1 value; if not
            specified then a ``nn.Sequential()`` module consisting of a linear layer and a
            sigmoid will automatically be created. Default is ``nn.Sequential(linear, sigmoid)``.
    """

    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        num_layers=2,
        custom_reduction=None,
    ):
        super(TransformerEncoderClassifier, self).__init__()

        if version.parse(torch.__version__) < version.parse("1.5.0"):
            logger.error(
                "You have PyTorch version %s installed, but `TransformerEncoderClassifier` "
                + "requires at least version 1.5.0.",
                torch.__version__,
            )
            sys.exit(1)

        self.nhead = nhead
        self.custom_reduction = custom_reduction

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=layer_norm)

        if custom_reduction:
            self.reduction = custom_reduction
        else:
            linear = nn.Linear(d_model, 1)
            # sigmoid = nn.Sigmoid()
            self.reduction = linear  # nn.Sequential(linear, sigmoid)

    def forward(self, x, mask):
        """
        Forward function. ``x`` is the input ``sent_vector`` tensor and ``mask`` avoids computations
        on padded values. Returns ``sent_scores``.
        """
        # add dimension in the middle
        attn_mask = mask.unsqueeze(1)
        # expand the middle dimension to the same size as the last dimension (the number of
        # sentences/source length)
        # Example with batch size 2: There are two masks since there are two sequences in the
        # batch. Each mask is a list of booleans for each sentence vector. The below line expands
        # each of these lists by duplicating them until they are each as long as the number of
        # sentences. Now instead of a list of booleans, each mask is a matrix where each row is
        # identical. This effectively masks tokens where the entire column is False.
        # Slight Explanation (for 2D not 3D): https://discuss.pytorch.org/t/how-to-add-padding-mask-to-nn-transformerencoder-module/63390/3  # noqa: E501
        # Detailed Explanation for Beginners: https://github.com/bentrevett/pytorch-seq2seq/blob/master/4%20-%20Packed%20Padded%20Sequences%2C%20Masking%2C%20Inference%20and%20BLEU.ipynb # noqa: E501
        # PyTorch MultiheadAttention Docs: https://pytorch.org/docs/stable/nn.html#torch.nn.MultiheadAttention.forward # noqa: E501
        attn_mask = attn_mask.expand(-1, attn_mask.size(2), -1)
        # repeat the mask for each attention head
        attn_mask = attn_mask.repeat(self.nhead, 1, 1)
        # attn_mask is shape (batch size*num_heads, target sequence length, source sequence length)
        # set all the 0's (False) to negative infinity and the 1's (True) to 0.0 because the
        # attn_mask is additive
        attn_mask = (
            attn_mask.float()
            .masked_fill(attn_mask == 0, float("-inf"))
            .masked_fill(attn_mask == 1, float(0.0))
        )

        x = x.transpose(0, 1)
        # x is shape (source sequence length, batch size, feature number)

        x = self.encoder(x, mask=attn_mask)
        # x is still shape (source sequence length, batch size, feature number)
        x = x.transpose(0, 1).squeeze()
        # x is shape (batch size, source sequence length, feature number)
        if self.custom_reduction:
            x = self.reduction(x, mask)
        else:
            x = self.reduction(x)
        # x is shape (batch size, source sequence length, 1)
        # mask is shape (batch size, source sequence length)
        sent_scores = x.squeeze(-1) * mask.float()
        sent_scores[sent_scores == 0] = -9e3
        return sent_scores
