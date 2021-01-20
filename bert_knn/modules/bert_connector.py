import torch
import transformers
from transformers import BertTokenizer, BertForMaskedLM, BasicTokenizer, BertModel, BertConfig
import numpy as np
from bert_knn.modules.base_connector import *
import torch.nn.functional as F

class Bert(Base_Connector):

    def __init__(self, args):
        super().__init__()

        bert_model_name = args.bert_model_name
        # When using a cased model, make sure to pass do_lower_case=False directly to BaseTokenizer
        do_lower_case = False
        if 'uncased' in bert_model_name:
            do_lower_case=True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Load pre-trained model (weights)
        # ... to get prediction/generation

        self.masked_bert_model = BertForMaskedLM.from_pretrained(bert_model_name)

        self.masked_bert_model.eval()

        # ... to get pooled output
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_model.eval()

        # ... to get hidden states
        config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)
        self.bert_model_hidden = BertModel.from_pretrained(bert_model_name, config=config)
        self.bert_model_hidden.eval()

        self.pad_id = self.inverse_vocab[BERT_PAD]

        self.unk_index = self.inverse_vocab[BERT_UNK]

    def get_id(self, string):
        tokenized_text = self.tokenizer.tokenize(string)
        indexed_string = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)

        return indexed_string

    def __get_input_tensors_batch(self, sentences_list):
        tokens_tensors_list = []
        segments_tensors_list = []
        masked_indices_list = []
        tokenized_text_list = []
        max_tokens = 0
        for sentences in sentences_list:
            tokens_tensor, segments_tensor, masked_indices, tokenized_text = self.__get_input_tensors(sentences)
            tokens_tensors_list.append(tokens_tensor)
            segments_tensors_list.append(segments_tensor)
            masked_indices_list.append(masked_indices)
            tokenized_text_list.append(tokenized_text)
            # assert(tokens_tensor.shape[1] == segments_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
        # print("MAX_TOKENS: {}".format(max_tokens))
        # apply padding and concatenate tensors
        # use [PAD] for tokens and 0 for segments
        final_tokens_tensor = None
        final_segments_tensor = None
        final_attention_mask = None
        for tokens_tensor, segments_tensor in zip(tokens_tensors_list, segments_tensors_list):
            dim_tensor = tokens_tensor.shape[1]
            pad_lenght = max_tokens - dim_tensor
            attention_tensor = torch.full([1,dim_tensor], 1, dtype= torch.long)
            if pad_lenght>0:
                pad_1 = torch.full([1,pad_lenght], self.pad_id, dtype= torch.long)
                pad_2 = torch.full([1,pad_lenght], 0, dtype= torch.long)
                attention_pad = torch.full([1,pad_lenght], 0, dtype= torch.long)
                tokens_tensor = torch.cat((tokens_tensor,pad_1), dim=1)
                segments_tensor = torch.cat((segments_tensor,pad_2), dim=1)
                attention_tensor = torch.cat((attention_tensor,attention_pad), dim=1)
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_segments_tensor = segments_tensor
                final_attention_mask = attention_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor,tokens_tensor), dim=0)
                final_segments_tensor = torch.cat((final_segments_tensor,segments_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask,attention_tensor), dim=0)

        return final_tokens_tensor, final_segments_tensor, final_attention_mask, masked_indices_list, tokenized_text_list

    def __get_input_tensors(self, sentences):

        if len(sentences) > 2:
            print(sentences)
            raise ValueError("BERT accepts maximum two sentences in input for each data point")

        first_tokenized_sentence = self.tokenizer.tokenize(sentences[0])
        first_segment_id = np.zeros(len(first_tokenized_sentence), dtype=int).tolist()

        # add [SEP] token at the end
        first_tokenized_sentence.append(BERT_SEP)
        first_segment_id.append(0)

        if len(sentences)>1 :
            second_tokenized_sentece = self.tokenizer.tokenize(sentences[1])
            second_segment_id = np.full(len(second_tokenized_sentece),1, dtype=int).tolist()

            # add [SEP] token at the end
            second_tokenized_sentece.append(BERT_SEP)
            second_segment_id.append(1)

            tokenized_text = first_tokenized_sentence + second_tokenized_sentece
            segments_ids = first_segment_id + second_segment_id
        else:
            tokenized_text = first_tokenized_sentence
            segments_ids = first_segment_id

        # add [CLS] token at the beginning
        tokenized_text.insert(0, BERT_CLS)
        segments_ids.insert(0,0)

        # look for masked indices
        masked_indices = []
        for i in range(len(tokenized_text)):
            token = tokenized_text[i]
            if token == MASK:
                masked_indices.append(i)

        max_tokens = 512
        if len(tokenized_text) > max_tokens:
            shift = int(max_tokens/2)
            if masked_indices[0] > shift:
                start =  masked_indices[0]-shift
                end = masked_indices[0]+shift
                masked_indices[0] = shift
            else:
                start = 0
                end = max_tokens
            segments_ids = segments_ids[start:end]
            tokenized_text = tokenized_text[start:end]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokens_tensor, segments_tensors, masked_indices, tokenized_text

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.masked_bert_model.cuda()
        self.bert_model.cuda()
        self.bert_model_hidden.cuda()

    def get_batch_generation(self, sentences_list, logger= None,
                             try_cuda=True):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, tokenized_text_list = self.__get_input_tensors_batch(sentences_list)

        if logger is not None:
            logger.debug("\n{}\n".format(tokenized_text_list))

        with torch.no_grad():
            logits = self.masked_bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device),
            )

            #log_probs = F.log_softmax(logits[0], dim=-1).cpu()
            all_output = logits[0]

        masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        masked_output = torch.softmax(masked_output, dim=-1).cpu()

        #log_probs = predictions[0, masked_indices_list]
        """token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))"""

        return masked_output, masked_indices_list

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, _, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, pooled_output = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        pooled_output = pooled_output.cpu()
        # attention_mask_tensor = attention_mask_tensor.type(torch.bool)
        return _, pooled_output

    def get_contextual_embeddings_mean(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, _, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_embeddings, pooled_output = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        attention_mask_tensor = attention_mask_tensor.type(torch.bool)
        output = np.zeros((all_embeddings.shape[0], all_embeddings.shape[2]))
        for idx, (embeddings, attention_mask) in enumerate(zip(all_embeddings, attention_mask_tensor)):
            output[idx] = np.mean(np.array(embeddings[attention_mask].cpu()), axis=0)
        return output


    def get_contextual_embeddings_mask_token(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            all_output, _ = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        masked_output = masked_output.cpu()
        return masked_output

    def get_hidden_state(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, _, hidden = self.bert_model_hidden(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        hidden = hidden[-2]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        #masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        hidden = hidden.cpu()
        return hidden

    def get_hidden_state_3(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, _, hidden = self.bert_model_hidden(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        hidden = hidden[-3]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        #masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        hidden = hidden.cpu()
        return hidden

    def get_hidden_state_4(self, sentences_list, try_cuda=True):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, masked_indices_list, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, _, hidden = self.bert_model_hidden(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        hidden = hidden[-4]
        hidden = hidden[np.arange(hidden.shape[0]), np.array(masked_indices_list).flatten()]
        #masked_output = all_output[np.arange(all_output.shape[0]), np.array(masked_indices_list).flatten()]
        hidden = hidden.cpu()
        return hidden

    def get_NN(self, sentences_list):

        # assume in input 1 or 2 sentences - in general, it considers only the first 2 sentences
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor, segments_tensor, attention_mask_tensor, _, _ = \
            self.__get_input_tensors_batch(sentences_list)

        with torch.no_grad():
            _, pooled_output = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                token_type_ids=segments_tensor.to(self._model_device),
                attention_mask=attention_mask_tensor.to(self._model_device))

        #sentence_lengths = [len(x) for x in tokenized_text_list]
        pooled_output = pooled_output.cpu()
        return pooled_output
