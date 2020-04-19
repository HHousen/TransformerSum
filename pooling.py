import torch
from torch import nn


class Pooling(nn.Module):
    def __init__(self, sent_rep_tokens=True, mean_tokens=False):
        """Methods to obtains sentence embeddings from word vectors. Multiple methods can be specificed
        and their results will be concatenated together. 
        
        Keyword Arguments:
            sent_rep_tokens {bool} -- Use the sentence representation token vectors as sentence
                                      embeddings. (default: {True})
            mean_tokens {bool} -- Take the mean of all the token vectors in each sentence. (default: {False})
        """
        super(Pooling, self).__init__()

        self.sent_rep_tokens = sent_rep_tokens
        self.mean_tokens = mean_tokens

        # pooling_mode_multiplier = sum([sent_rep_tokens, mean_tokens])
        # self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(
        self,
        word_vectors=None,
        sent_rep_token_ids=None,
        sent_rep_mask=None,
        sent_lengths=None,
        sent_lengths_mask=None,
    ):
        output_vectors = []
        output_masks = []

        if self.sent_rep_tokens:
            sents_vec = word_vectors[
                torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
            ].squeeze()
            sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_rep_mask)

        if self.mean_tokens:
            batch_sequences = [
                torch.split(word_vectors[idx], seg)
                for idx, seg in enumerate(sent_lengths)
            ]
            sents_list = [
                torch.stack(
                    [
                        # the mean with padding ignored
                        (sequence.sum(dim=0) / (sequence != 0).sum(dim=0))
                        # if the sequence contains values that are not zero
                        if ((sequence != 0).sum() != 0)
                        # any tensor with 2 dimensions (one being the hidden size) that has already been created (will be set to zero from padding)
                        else word_vectors[0, 0].float()
                        # for each sentence
                        for sequence in sequences
                    ],
                    dim=0,
                )
                for sequences in batch_sequences  # for all the sentences in each batch
            ]
            sents_vec = torch.stack(sents_list, dim=0)
            sents_vec = sents_vec * sent_lengths_mask[:, :, None].float()
            output_vectors.append(sents_vec)
            output_masks.append(sent_lengths_mask)

        output_vector = torch.cat(output_vectors, 1)
        output_mask = torch.cat(output_masks, 1)

        return output_vector, output_mask
