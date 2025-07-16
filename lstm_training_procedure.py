import spacy
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
import gensim
import itertools
from config import parser
from dataset import LSTMAmazonRandomPrefix, LSTMDatasetAmazon
from model import PadSequence, LSTMClassifier
from utils import process_df, create_subsets_df, init_random_seed, train_model_lstm
from transformers import AdamW


def instantiate_train_amazon(configuration, hparams):
    hidden_size, learning_rate, batch_size, dropout, num_layers, wd, num_epochs, window_size = configuration

    init_random_seed(hparams.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    w2v_model_path = os.path.join(hparams.w2v_model_dir,
                                  f"{hparams.w2v_model_path_prefix}_window_{window_size}.kv")
    w2v_model = gensim.models.KeyedVectors.load(w2v_model_path)
    word2vec_vectors = list(w2v_model.vectors)
    vocab_size = len(w2v_model.index_to_key) + 2

    embedding_dim = w2v_model.vector_size

    word2vec_vectors.append(np.random.normal(scale=1.0, size=(embedding_dim,)))
    word2vec_vectors.append(np.zeros(shape=(embedding_dim,)))

    bidirection = True  # set False if you wish to use single direction LSTM

    train = pd.read_csv(hparams.train_dataset_path,
                        keep_default_na=False,
                        na_values=['$$$__$$$'])

    dev = pd.read_csv(hparams.dev_dataset_path,
                      keep_default_na=False,
                      na_values=['$$$__$$$'])
    if hparams.mode in ['subsets', 'random']:
        # If you wish to change the tokenizer you will have to change the \
        # definition inside the dataset classes as well (defined this way due to spacy's instability in multithreading)
        spacy_tokenizer = spacy.blank("en")
        train['title'] = train['title'].apply(lambda x: " ".join(x.split()))
        dev['title'] = dev['title'].apply(lambda x: " ".join(x.split()))

        # removes class 102 and reassings label for alignment between classes in amazon dataset
        train = process_df(train, spacy_tokenizer)
        dev = process_df(dev, spacy_tokenizer)

        if hparams.mode == 'subsets':
            # extracting all prefix subsets from original data
            train = create_subsets_df(train, spacy_tokenizer)
            dev = create_subsets_df(dev, spacy_tokenizer)

    number_of_classes = train['label'].nunique()

    if hparams.mode == 'random':
        train_dl = DataLoader(LSTMAmazonRandomPrefix(train, "title", "label", w2v_model.key_to_index,
                                                     unk_token_id=len(word2vec_vectors) - 2),
                              batch_size=batch_size, num_workers=4, shuffle=True,
                              collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
        dev_dl = DataLoader(
            LSTMAmazonRandomPrefix(dev, "title", "label", w2v_model.key_to_index,
                                   unk_token_id=len(word2vec_vectors) - 2),
            batch_size=batch_size, num_workers=4, collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))

    elif hparams.mode in ['complete', 'subsets']:
        train_dl = DataLoader(LSTMDatasetAmazon(train, "title", "label", w2v_model.key_to_index,
                                                unk_token_id=len(word2vec_vectors) - 2),
                              batch_size=batch_size, num_workers=4, shuffle=True,
                              collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
        dev_dl = DataLoader(
            LSTMDatasetAmazon(dev, "title", "label", w2v_model.key_to_index,
                              unk_token_id=len(word2vec_vectors) - 2),
            batch_size=batch_size, num_workers=4, collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
    else:
        raise NotImplementedError(f'mode {hparams.mode} is not supported!')

    lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, num_layers, bidirection, dropout,
                                number_of_classes, pad_token_id=len(word2vec_vectors) - 1)

    word2vec_vectors = torch.FloatTensor(word2vec_vectors)

    lstm_model.embedding.load_state_dict({"weight": word2vec_vectors})
    lstm_model = lstm_model.to(device)

    optimizer = AdamW(lstm_model.parameters(), lr=learning_rate, eps=1e-6, weight_decay=wd)

    checkpoint_name = hparams.checkpoint_prefix + "_".join(
        [str(hparams.seed), str(batch_size), str(wd), str(learning_rate), str(num_epochs), str(dropout),
         str(num_layers),
         str(window_size), str(hidden_size), 'bi-directional' if bidirection else 'uni-directional', hparams.mode])

    train_model_lstm(lstm_model, hparams.models_dir, checkpoint_name, train_dl, dev_dl, optimizer,
                     num_epochs=num_epochs,
                     log_dir=hparams.tensorboard_log_dir, device=device, min_delta=hparams.min_delta,
                     warmup_steps=hparams.warmup_steps, patience=hparams.patience)


def training_wrapper_amazon(hparams):
    # this is the grid used in the paper
    window_sizes = [4, 6]
    hidden_size = [384, ]
    learning_rate = [0.01, 0.001, 0.0001, 1e-5, 3e-5, 5e-5]
    batch_size = [64]
    dropout = [0.1, ]
    num_layers = [1]
    num_epochs = [10]
    weight_decay = [0.01, ]

    configurations = list(
        itertools.product(hidden_size, learning_rate, batch_size, dropout, num_layers, weight_decay, num_epochs,
                          window_sizes))

    for config in configurations:
        instantiate_train_amazon(config, hparams)


if __name__ == "__main__":
    hparams = parser.parse_args()
    training_wrapper_amazon(hparams)
