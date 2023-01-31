import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
import spacy


def get_data_loader_amazon(data, tokenizer, collate, mode='complete', **kwargs):
    if mode == 'complete' or mode == 'subsets':
        dataset = ETRDataSetAmazon(data, tokenizer)
    elif mode=='random':
        dataset = ETRDataSetAmazonRandomPrefix(data, tokenizer)
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
