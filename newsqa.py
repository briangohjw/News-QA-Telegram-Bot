import numpy as np
import re
import torch

# For reproducibility
torch.manual_seed(0)

class NewsQaExample(object):
    '''
    Create single NewsQA object with text-question pair and related features
    '''

    def __init__(self, text, question, is_training = True):
        self.doc_text = text
        self.ques_text = question
        self.char_start_idx = None
        self.char_end_idx = None
        self.is_training = is_training

        if self.char_start_idx != -1 or self.char_end_idx != -1:
            self.char_to_word_map = self.get_char_to_word_idx()
            if is_training:
                self.word_start_idx = self.char_to_word_map[self.char_start_idx]
                self.word_end_idx = self.char_to_word_map[self.char_end_idx]
                self.is_impossible = False

        elif is_training:
            self.is_impossible = True

        self.tokens = None
        self.token_to_org_map = None
        self.org_to_token_map = None

        self.token_start_idx = None
        self.token_end_idx = None
        self.offset = 0

        self.input_ids = None
        self.segment_ids = None
        self.attention_mask = None

    def get_char_to_word_idx(self):
        '''
        A functions that returns a list which maps each character index to a word index
        '''
        char_to_word = []
        words = re.split(' ', self.doc_text)

        for idx in range(len(words)):
            # The space next to a word will be considered as part of that word itself
            char_to_word = char_to_word + [idx] * (len(words[idx]) + 1)

        # There is no space after last word, so we need to remove the last element
        char_to_word = char_to_word[:-1]

        # Check for errors
        assert len(char_to_word) == len(self.doc_text)

        return char_to_word

    def encode(self, tokenizer, max_seq_len = 512, max_ques_len = 50, doc_stride = 128):
        '''
        Returns a dictionary with BERT features: input_ids, segment_ids and attn_mask
        '''
        self._calculate_input_features(tokenizer, max_seq_len, max_ques_len,
                                       doc_stride)

        features = []
        for idx in range(len(self.input_ids)):
            feature = {"input_ids": torch.tensor([self.input_ids[idx]]),
                        "token_type_ids": torch.tensor([self.segment_ids[idx]]),
                        "attention_mask": torch.tensor([self.attention_mask[idx]])}
            features.append(feature)
        return features

    def check_errors(self, input_tokens, input_id, segment_id, attn_msk):
        '''
        Check if the count for input_tokens, input_id, segment_id, and attn_msk tallies
        '''
        assert len(input_tokens) == len(input_id)
        assert len(input_id) == len(segment_id)
        assert len(segment_id) == len(attn_msk)

    def update_ftrs(self, input_tokens, input_id, segment_id, attn_msk):
        self.tokens.append(input_tokens)
        self.input_ids.append(input_id)
        self.segment_ids.append(segment_id)
        self.attention_mask.append(attn_msk)

    def reset_ftrs(self):
        self.tokens = []
        self.input_ids = []
        self.segment_ids = []
        self.attention_mask = []

    def get_ftrs(self,ques_tokens, doc_tokens, tokenizer):
        '''
        Get features from ques_tokens and doc_tokens
        '''
        input_tokens = ['[CLS]'] + ques_tokens + ['[SEP]'] + doc_tokens + ['[SEP]']
        input_id = tokenizer.convert_tokens_to_ids(input_tokens)
        segment_id = [0] * (len(ques_tokens) + 2) + [1] * (len(doc_tokens) + 1)
        attn_msk = [1] * len(input_id)
        return input_id, input_tokens, segment_id, attn_msk

    def update_token_mappings(self,max_ques_len,ques_tokens):
        '''
        Update token mappings with newly added question
        '''
        self.org_to_token_map = np.array(self.org_to_token_map)
        self.org_to_token_map = self.org_to_token_map + min(max_ques_len, len(ques_tokens)) + 2

        self.token_to_org_map = [-1] * (min(max_ques_len, len(ques_tokens)) + 2) + self.token_to_org_map + [-1]
        self.token_to_org_map = np.array(self.token_to_org_map)

    def init_token_mapping(self, words, tokenizer):
        '''
        Create mapping between orig word indices and token indices
        '''
        self.token_to_org_map = []
        self.org_to_token_map = []
        all_doc_tokens = []
        for idx, word in enumerate(words):
            self.org_to_token_map.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(word)
            # Check for sub-tokens in space-seperated word
            for token in sub_tokens:
                self.token_to_org_map.append(idx)
                all_doc_tokens.append(token)
        return all_doc_tokens

    def _calculate_input_features(self, tokenizer, max_seq_len, max_ques_len,
                                  doc_stride):
        '''
        1. Tokenization
        2. Creates mapping between token indices and original word indices
        3. Calculates input ids and segment ids
        '''
        ques_tokens = tokenizer.tokenize(self.ques_text)
        # Trim question if exceeds max_ques_len
        if len(ques_tokens) > max_ques_len:
            ques_tokens = ques_tokens[:max_ques_len]

        words = re.split(' ', self.doc_text)
        all_doc_tokens = self.init_token_mapping(words, tokenizer)
        self.update_token_mappings(max_ques_len,ques_tokens)
        # Do not count [CLS], [SEP], [SEP] tokens (i.e. -3)
        max_doc_len = max_seq_len - min(max_ques_len, len(ques_tokens)) - 3

        self.reset_ftrs()

        # Use single question-text pair if document < max_doc_len
        if len(all_doc_tokens) < max_doc_len:
            input_id, input_tokens, segment_id, attn_msk = self.get_ftrs(ques_tokens, all_doc_tokens, tokenizer)
            self.check_errors(input_tokens, input_id, segment_id, attn_msk)
            self.update_ftrs(input_tokens, input_id, segment_id, attn_msk)

        # Else, use multiple question-text pairs
        else:
            for i in range(0, len(all_doc_tokens), doc_stride):
                doc_tokens = all_doc_tokens[i:i+max_doc_len]
                input_id, input_tokens, segment_id, attn_msk = self.get_ftrs(ques_tokens, doc_tokens, tokenizer)
                self.check_errors(input_tokens, input_id, segment_id, attn_msk)
                self.update_ftrs(input_tokens, input_id, segment_id, attn_msk)

    def get_ans_char_range(self, token_start_idx, token_end_idx, offset = None):
        '''
        Calculates answer's start and end index from token indices
        '''
        if offset is None:
            offset = self.offset

        try:
            # Update indices based on offset
            token_start_idx = token_start_idx + offset
            token_end_idx = token_end_idx + offset

            # Getting word indices from token indices
            start_word_idx = self.token_to_org_map[token_start_idx]
            end_word_idx = self.token_to_org_map[token_end_idx]

            # If the indices are a part of the question, it means that there
            # is no answer
            if start_word_idx == -1 or end_word_idx == -1:
                return (0, 0)

            # Getting char indices from word indices
            start_char_idx = self.char_to_word_map.index(start_word_idx)
            end_char_idx = self.char_to_word_map.index(end_word_idx + 1)

        except:
            start_char_idx = 0
            end_char_idx = 0

        return (start_char_idx, end_char_idx)

class NewsQaModel(object):
    def __init__(self, model = None):
        self.model = model

    def load(self, filename):
        self.model = torch.load(filename, map_location=torch.device('cpu'))

    def predict(self, **input_features):
        self.model.to(torch.device("cpu"))
        return self.model(**input_features)

def get_single_prediction(text, question, tokenizer, newsqa_model, doc_stride = 128):
    '''
    Get single prediction given text, question, tokenizer and model
    '''
    # Encoding: Generate features
    ex = NewsQaExample(text, question, is_training=False)
    features = ex.encode(tokenizer)
    outputs = []
    ans_texts = []

    for idx, input_feature in enumerate(features):
        # Get predictions
        model_out = newsqa_model.predict(**input_feature)
        start_scores = model_out[0]
        end_scores = model_out[1]
        start_scores = start_scores.cpu().detach().numpy()
        end_scores = end_scores.cpu().detach().numpy()

        # Build the answer(s) using orig word indices from token start/end indices
        start_idx = int(np.argmax(start_scores))
        end_idx = int(np.argmax(end_scores))
        char_start_idx, char_end_idx = ex.get_ans_char_range(start_idx, end_idx,
                                                             offset = idx * doc_stride)
        outputs.append((char_start_idx, char_end_idx))
        ans_texts.append(text[char_start_idx:char_end_idx])

    return ans_texts, outputs
