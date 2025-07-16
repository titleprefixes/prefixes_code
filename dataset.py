import torch
from torch.utils.data import Dataset, DataLoader
import math
import random
import spacy
import re


def get_data_loader_amazon_bert(data, bert_tokenizer, collate, mode='complete', **kwargs):
    if mode == 'complete' or mode == 'subsets':
        dataset = DatasetAmazon(data, bert_tokenizer)
    elif mode == 'random':
        dataset = AmazonRandomPrefix(data, bert_tokenizer)
    else:
        raise NotImplementedError(f'mode {mode} is not supported!')
    return DataLoader(dataset, collate_fn=collate, **kwargs)


class AmazonRandomPrefix(Dataset):

    def __init__(self, df, bert_tokenizer, max_length=512):
        """ Load dataset
        """
        self.max_length = max_length
        self.bert_tokenizer = bert_tokenizer
        self.df = df
        self.spacy_tokenizer = spacy.blank("en")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        title = row['title']
        tokens = self.spacy_tokenizer(title)
        length = len(tokens)
        max_index = math.floor(length / 2)
        rand_index = random.randint(1, max_index)
        random_prefix = tokens[:rand_index].text
        input_ids = self.bert_tokenizer.encode(random_prefix, add_special_tokens=True, max_length=self.max_length,
                                               truncation=True)
        label = row['label']
        return input_ids, label


class DatasetAmazon(Dataset):

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


class LSTMDatasetAmazon(Dataset):

    def __init__(self, df, input_col, label_col, vocab_index, unk_token_id):
        self.df = df
        self.input_col = input_col
        self.label_col = label_col
        self.spacy_tokenizer = spacy.blank("en")
        self.vocab_index = vocab_index
        self.unk_token_id = unk_token_id

    def tokenize_title(self, text):
        text = text.lower()
        # Remove non Ascii letters
        text = "".join(c for c in text if ord(c) < 128)
        # Separate words
        text = " ".join(a.text for a in self.spacy_tokenizer(text))
        # Replace dot without space
        text = text.replace(".", "")
        text = re.sub('[^A-Za-z0-9 ]+', ' ', text.strip())
        tokens = text.split()
        if len(tokens) == 0:
            return [self.unk_token_id]
        return [self.vocab_index[token] if token in self.vocab_index else self.unk_token_id for token in tokens]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        title = row[self.input_col]
        label = row[self.label_col]

        return {'text': self.tokenize_title(title), 'label': label}


class LSTMAmazonRandomPrefix(Dataset):

    def __init__(self, df, input_col, label_col, vocab_index, unk_token_id, random_=True):
        self.df = df
        self.input_col = input_col
        self.label_col = label_col
        self.spacy_tokenizer = spacy.blank("en")
        self.vocab_index = vocab_index
        self.unk_token_id = unk_token_id
        self.random = random_

    def tokenize_title(self, text):
        text = text.lower()

        # Remove non Ascii letters
        text = "".join(c for c in text if ord(c) < 128)
        # Separate words
        text = " ".join(a.text for a in self.spacy_tokenizer(text))
        # Replace dot without space
        text = text.replace(".", "")
        text = re.sub('[^A-Za-z0-9 ]+', ' ', text.strip())
        tokens = text.split()
        if len(tokens) == 0:
            return [self.unk_token_id]
        return [self.vocab_index[token] if token in self.vocab_index else self.unk_token_id for token in tokens]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        title = row[self.input_col]
        if self.random:
            tokens = self.spacy_tokenizer(title)  # Does not preprocess for alignment with BERT
            length = len(tokens)
            max_index = math.floor(length / 2)
            rand_index = random.randint(1, max_index)
            prefix = tokens[:rand_index].text
        else:
            prefix = title

        label = row[self.label_col]

        return {'text': self.tokenize_title(prefix), 'label': label}
