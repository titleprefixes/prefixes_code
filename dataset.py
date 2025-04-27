import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
import spacy


def get_data_loader_amazon(data, tokenizer, collate, mode='complete',no_aspect_tokens = True,only_aspect_sentence = False,common_tags=[], **kwargs):
    if mode == 'complete' or mode == 'subsets':
        dataset = ETRDataSetAmazon(data, tokenizer)
    elif mode=='random':
        dataset = ETRDataSetAmazonRandomPrefix(data, tokenizer)
    elif mode=='attributes':
        dataset = ETRDataSetAmazonRandomPrefixWithAspects(data, tokenizer,common_tags,no_aspect_tokens=no_aspect_tokens,only_aspect_sentence=only_aspect_sentence)
    else:
        raise NotImplementedError(f'mode {mode} is not supported!')
    return DataLoader(dataset, collate_fn=collate, **kwargs)


class ETRDataSetAmazonRandomPrefix(Dataset):

    def __init__(self, df, bert_tokenizer, max_length=512):
        """ Load dataset
        """
        self.max_length = max_length
        self.bert_tokenizer = bert_tokenizer
        self.df = df
        self.spacy_tokeizer = spacy.blank("es")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        title = row['title']
        tokens = self.spacy_tokeizer(title)
        length = len(tokens)
        max_index = math.floor(length / 2)
        rand_index = random.randint(1, max_index)
        random_prefix = tokens[:rand_index].text
        input_ids = self.bert_tokenizer.encode(random_prefix, add_special_tokens=True, max_length=self.max_length,
                                               truncation=True)
        label = row['label']
        return input_ids, label


class ETRDataSetAmazon(Dataset):

    def __init__(self, df, bert_tokenizer, max_length=512):
        """ Load dataset
        """
        self.max_length = max_length
        self.bert_tokenizer = bert_tokenizer
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        title = row['title']
        input_ids = self.bert_tokenizer.encode(title, add_special_tokens=True, max_length=self.max_length,
                                               truncation=True)
        label = row['label']
        return input_ids, label


def collate_batch_amazon(batch, pad_token_id=0):
    """ Converts titles (token_ids) and labels into input format for training.
        Including padding, attention masks and token type ids
    """
    # sequences, labels, item_sizes, item_ids, epids, vi_buckets = zip(*batch)
    item_titles, labels = zip(*batch)
    max_length = max(len(sequence) for sequence in item_titles)

    # pad sequences to max length of batch
    input_ids = torch.tensor(
        [sequence + ([pad_token_id] * (max_length - len(sequence))) for sequence in item_titles], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    # only attent to non padding token
    attention_mask = torch.tensor(
        [([1] * len(sequence)) + ([0] * (max_length - len(sequence))) for sequence in item_titles],
        dtype=torch.long)

    # we only have one sequence so we set all position to 0
    token_type_ids = torch.tensor([[0] * max_length for _ in item_titles], dtype=torch.long)

    return (input_ids, attention_mask, token_type_ids), labels





class ETRDataSetAmazonRandomPrefixWithAspects(Dataset):

    def __init__(self, df, bert_tokenizer, common_tags, max_length=512, random=True, no_aspect_tokens=False,
                 only_aspect_sentence=False, index=0):

        self.max_length = max_length
        self.bert_tokenizer = bert_tokenizer
        self.df = df
        self.spacy_tokeizer = spacy.blank("en")
        self.common_tags = common_tags
        self.random_prefix = random
        self.only_aspect_sentence = only_aspect_sentence
        self.no_aspect_tokens = no_aspect_tokens
        self.index = index


    def construct_aspect_sentece(self, response, tokens, index):
        sentence = ""
        prefix_tokens = [t.text for t in tokens[:index]]
        token_tags = response[:index]
        for i,token_data in enumerate(token_tags):
            token = prefix_tokens[i]
            tok_tag = token_data['tag']
            if tok_tag == "UNK" and not self.no_aspect_tokens:
                continue
            if tok_tag in self.common_tags:
                sentence += f' [{tok_tag}] ' + token
            elif self.no_aspect_tokens:
                sentence += f' [UNK] ' + token
        return sentence


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        title = row['title']
        label = row['label']
        response = eval(row['response'])
        if self.random_prefix:
            tokens = self.spacy_tokeizer(title)
            length = len(tokens)
            max_index = math.floor(length / 2)
            rand_index = random.randint(1, max_index)
            prefix = tokens[:rand_index].text
            aspect_sentence = self.construct_aspect_sentece(response,tokens, rand_index)
        else:
            prefix = title[:self.index].text
            aspect_sentence = self.construct_aspect_sentece(response,title, self.index)
        if not self.only_aspect_sentence:
            input_data = self.bert_tokenizer.encode_plus(prefix, aspect_sentence, add_special_tokens=True,
                                                         max_length=self.max_length, truncation='only_second')
        else:
            input_data = self.bert_tokenizer.encode_plus(aspect_sentence, add_special_tokens=True,
                                                         max_length=self.max_length, truncation=True)
        return input_data['input_ids'], input_data['token_type_ids'], label



def collate_batch_random_prefix_with_aspects_amazon(batch, pad_token_id=0, only_aspect_sentence=False):
    """ Converts titles (token_ids) and labels into input format for training.
        Including padding, attention masks and token type ids
    """

    pair_titles_ids, pair_token_type_ids, labels = zip(*batch)
    max_length = max(len(sequence) for sequence in pair_titles_ids)

    input_ids = torch.tensor(
        [sequence + ([pad_token_id] * (max_length - len(sequence))) for sequence in pair_titles_ids], dtype=torch.long)
    attention_mask = torch.tensor(
        [([1] * len(sequence)) + ([0] * (max_length - len(sequence))) for sequence in pair_titles_ids],
        dtype=torch.long)

    if not only_aspect_sentence:
        token_type_ids = torch.tensor(
            [sequence + ([1] * (max_length - len(sequence))) for sequence in pair_token_type_ids],
            dtype=torch.long)
    else:
        token_type_ids = torch.tensor(
            [sequence + ([0] * (max_length - len(sequence))) for sequence in pair_token_type_ids],
            dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return (input_ids, attention_mask, token_type_ids), labels