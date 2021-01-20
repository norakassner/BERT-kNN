import re
import torch

MASK = "[MASK]"
BERT_UNK = "[UNK]"
BERT_CLS = "[CLS]"
BERT_SEP = "[SEP]"
BERT_PAD = "[PAD]"

SPECIAL_SYMBOLS = [
    MASK,
    BERT_UNK,
    BERT_CLS,
    BERT_SEP,
    BERT_PAD,
    ]

SPACE_NORMALIZER = re.compile(r"\s+")


def default_tokenizer(line):
    """Default tokenizer for models that don't have one

    Args:
        line: a string representing a sentence

    Returns:
        A list of tokens
    """

    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    line = line.replace(MASK, " "+str(MASK)+" ") #make sure MASK is correctly splitted

    # fix tokenization for parentheses
    line = line.replace('(', " ( ")
    line = line.replace(')', " ) ")

    # fix tokenization for comma
    line = line.replace(',', " , ")

    # fix tokenization for -- (e.g., 1954--1988)
    line = line.replace('--', " -- ")

    result = line.split()
    return result


class Base_Connector():

    def __init__(self):

        # these variables should be initialized
        self.vocab = None

        # This defines where the device where the model is. Changed by try_cuda.
        self._model_device = 'cpu'

    def optimize_top_layer(self, vocab_subset):
        """
        optimization for some LM
        """
        pass

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                print('Moving model to CUDA')
                self._cuda()
                self._model_device = 'cuda'
        else:
            print('No CUDA found')

    def _cuda(self):
        """Move model to GPU."""
        raise NotImplementedError

    def get_id(self, string):
        raise NotImplementedError()

    def get_generation(self, sentences, logger=None):
        [log_probs], [token_ids], [masked_indices] = self.get_batch_generation(
            [sentences], logger=logger, try_cuda=False)
        return log_probs, token_ids, masked_indices

    def get_batch_generation(self, sentences_list, logger= None, try_cuda=True):
        raise NotImplementedError()

    def get_contextual_embeddings(self, sentences):
        """Compute the contextual embeddings of a list of sentences

        Parameters:
        sentences (list[list[string]]): list of elements. Each element is a list
                                        that contains either a single sentence
                                        or two sentences

        Returns:
        encoder_layers (list(Tensor)): a list of the full sequences of encoded-hidden-states
                            at the end of each attention block (e.g., 12 full
                            sequences for BERT-base,), each encoded-hidden-state
                            is a torch.FloatTensor of size [batch_size,
                            sequence_length, hidden_size]
        sentence_lengths (list[int]): list of lenghts for the sentences in the
                                      batch
        tokenized_text_list: (list[list[string]]): tokenized text for the sentences
                                                   in the batch
        """
        raise NotImplementedError()
