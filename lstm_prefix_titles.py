import re
import spacy
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import os

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import gensim
import random
import math
import itertools
from config import parser
from utils import process_df, save_ckpt
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup


def tokenize_title(text, tokenizer):
    text = text.lower()
    # Remove non Ascii letters
    text = "".join(c for c in text if ord(c) < 128)
    # Separate words
    text = " ".join(a.text for a in tokenizer(text))
    # Replace dot without space
    text = text.replace(".", "")
    # Clean text from special characters
    text = re.sub('[^A-Za-z0-9 ]+', ' ', text.strip())
    text = " ".join(text.split())
    return text


class LSTMClassifier(nn.Module):

    # define all the layers used in model
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, bidirectional, dropout, number_of_classes,
                 pad_token_id):
        # Constructor
        super().__init__()
        # embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        # lstm layer
        self.hidden_dim = hidden_dim
        self.num_layers = n_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            bidirectional=self.bidirectional,
                            dropout=dropout,
                            batch_first=True)

        # dense layer
        if self.bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, number_of_classes)
        else:
            self.fc = nn.Linear(hidden_dim, number_of_classes)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)

        # packed sequence
        packed_input = pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True)
        with torch.backends.cudnn.flags(enabled=False):
            packed_output, (hidden, cell) = self.lstm(packed_input)

        if self.bidirectional:
            # concat the final forward and backward hidden state
            hidden = torch.cat((hidden[self.num_layers - 1, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        dense_outputs = self.fc(hidden)
        return dense_outputs



class PadSequence:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        sorted_batch = sorted(batch, key=lambda x: len(x['text']), reverse=True)
        sequences = [torch.LongTensor(x['text']) for x in sorted_batch]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.pad_token_id)
        lengths = torch.LongTensor([len(x) for x in sequences])
        labels = torch.LongTensor([x['label'] for x in sorted_batch])
        return sequences_padded, lengths, labels

class LSTMTitlesDataset(Dataset):

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





class LSTMPrefixTitlesDataset(Dataset):

    def __init__(self, df, input_col, label_col, vocab_index, unk_token_id,random_=True):
        self.df = df
        self.input_col = input_col
        self.label_col = label_col
        self.spacy_tokenizer = spacy.blank("en")
        self.vocab_index = vocab_index
        self.unk_token_id = unk_token_id
        self.random=random_

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
            prefix=title

        label = row[self.label_col]

        return {'text': self.tokenize_title(prefix), 'label': label}


def train_model(model,
                checkpoint_name,
                train_loader,
                valid_loader,
                optimizer,
                num_epochs=10,
                device=torch.device('cuda'),
                log_dir="./"
                ):

    LSTM_MODEL_DIR = "lstm_models/"
    if not os.path.exists(LSTM_MODEL_DIR):
        os.makedirs(LSTM_MODEL_DIR)
    MIN_DELTA = 0.05
    PATIENCE = 2

    criterion = nn.CrossEntropyLoss()
    global_step = 0
    termination = False

    writer = SummaryWriter(
        log_dir=os.path.join(log_dir, checkpoint_name))
    total_steps = num_epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=1e4,
                                                num_training_steps=total_steps)
    # training loop
    for epoch_i in range(num_epochs):
        model.train()

        if termination:
            print("No improvement in validation loss, terminating")
            break

        total_train_loss = 0
        total_bs_train = 0
        for batch in train_loader:
            titles, titles_len, labels = batch
            labels = labels.to(device)
            titles = titles.to(device)
            titles_len = titles_len.to(device)
            outputs = model(titles, titles_len)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss.item(), global_step)
            total_train_loss += (loss.item() * len(labels))
            total_bs_train += len(labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # update running values
            global_step += 1

        avg_train_loss = total_train_loss / total_bs_train
        print("")
        print("  Average training loss: {}".format(avg_train_loss), flush=True)

        print("")
        print("Running Validation...")

        total_eval_loss = 0
        total_bs = 0
        all_labels = []
        all_preds = []
        # Evaluate data for one epoch
        model.eval()

        for batch in valid_loader:
            with torch.no_grad():
                titles, titles_len, labels = batch
                titles = titles.to(device)
                titles_len = titles_len.to(device)
                labels = labels.to(device)
                outputs = model(titles, titles_len)
                loss = criterion(outputs, labels)
                total_eval_loss += (loss.item() * len(labels))
                total_bs += len(labels)

                predicted_labels = outputs.argmax(dim=-1)
                all_preds.extend(list(predicted_labels.cpu().numpy()))
                all_labels.extend(list(labels.cpu().numpy()))
        avg_val_loss = total_eval_loss / total_bs
        writer.add_scalar("Loss/dev", avg_val_loss, epoch_i)
        acc = np.mean([1 if i == j else 0 for i, j in zip(all_preds, all_labels)])
        writer.add_scalar("Accuracy/dev", acc, epoch_i)
        print("  Validation Loss: {}".format(avg_val_loss), flush=True)
        print("  Validation classification accuracy: {}".format(acc), flush=True)

        if epoch_i == 0:
            BEST = avg_val_loss
            LOWEST_LOSS = avg_val_loss
            BEST_ACC = acc
            save_ckpt(LSTM_MODEL_DIR, model, checkpoint_name + 'best_acc')
            continue
        else:
            if BEST_ACC < acc:
                save_ckpt(LSTM_MODEL_DIR, model, checkpoint_name + 'best_acc')
                BEST_ACC = acc
            if avg_val_loss < LOWEST_LOSS:
                LOWEST_LOSS = avg_val_loss

            if BEST - MIN_DELTA <= avg_val_loss:
                PATIENCE -= 1
                if PATIENCE == 0:
                    termination = True
            else:
                PATIENCE = 2
                BEST = avg_val_loss
    writer.close()
    print('Finished Training!')
    print("BEST ACC", BEST_ACC)



def instantiate_train_amazon(configuration, seed,hparams):
    hidden_size, learning_rate, batch_size, dropout, num_layers, wd, window_size = configuration


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    nlp = spacy.blank("en")
    device = torch.device('cuda')

    w2v_model = gensim.models.KeyedVectors.load(hparams.w2v_model_path)
    word2vec_vectors = list(w2v_model.vectors)
    vocab_size = len(w2v_model.index_to_key) + 2

    embedding_dim = w2v_model.vector_size

    word2vec_vectors.append(np.random.normal(scale=1.0, size=(embedding_dim,)))
    word2vec_vectors.append(np.zeros(shape=(embedding_dim,)))

    bidirection = True

    train = pd.read_csv(hparams.train_dataset_path,
                        keep_default_na=False,
                        na_values=['$$$__$$$'])

    dev = pd.read_csv(hparams.dev_dataset_path,
                      keep_default_na=False,
                      na_values=['$$$__$$$'])

    train['title'] = train['title'].apply(lambda x: " ".join(x.split()))
    dev['title'] = dev['title'].apply(lambda x: " ".join(x.split()))

    train = process_df(train, nlp)
    dev = process_df(dev, nlp)

    number_of_classes = train['label'].nunique()

    if hparams.mode=='random':
        train_dl = DataLoader(LSTMPrefixTitlesDataset(train, "title", "label", w2v_model.key_to_index,
                                                      unk_token_id=len(word2vec_vectors) - 2),
                              batch_size=batch_size, num_workers=4, shuffle=True,
                              collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
        dev_dl = DataLoader(
            LSTMPrefixTitlesDataset(dev, "title", "label", w2v_model.key_to_index, unk_token_id=len(word2vec_vectors) - 2),
            batch_size=batch_size, num_workers=4, collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
        lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, num_layers, bidirection, dropout,
                                    number_of_classes, pad_token_id=len(word2vec_vectors) - 1)
    else:
        train_dl = DataLoader(LSTMTitlesDataset(train, "title", "label", w2v_model.key_to_index,
                                                      unk_token_id=len(word2vec_vectors) - 2),
                              batch_size=batch_size, num_workers=4, shuffle=True,
                              collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
        dev_dl = DataLoader(
            LSTMTitlesDataset(dev, "title", "label", w2v_model.key_to_index,
                                    unk_token_id=len(word2vec_vectors) - 2),
            batch_size=batch_size, num_workers=4, collate_fn=PadSequence(pad_token_id=len(word2vec_vectors) - 1))
        lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_size, num_layers, bidirection, dropout,
                                    number_of_classes, pad_token_id=len(word2vec_vectors) - 1)

    word2vec_vectors = torch.FloatTensor(word2vec_vectors)

    lstm_model.embedding.load_state_dict({"weight": word2vec_vectors})
    lstm_model = lstm_model.to(device)

    optimizer = AdamW(lstm_model.parameters(), lr=learning_rate, eps=1e-6, weight_decay=wd)

    checkpoint_name = "lstm_prefix_etr_amazon_" + "_".join(
        [str(seed), str(batch_size), str(wd), str(learning_rate), str(dropout), str(num_layers), str(window_size)])

    train_model(lstm_model, checkpoint_name, train_dl, dev_dl, optimizer, num_epochs=hparams.num_epochs,
                log_dir=hparams.tensorboard_log_dir, device=device)







def training_wrapper_amazon(seed,hparams):
    window_sizes = [4,6]
    hidden_size = [384, ]
    learning_rate = [0.01, 0.001, 0.0001, 1e-5, 3e-5, 5e-5]
    batch_size = [64]
    dropout = [0.1, ]
    num_layers = [1]
    weight_decay = [0.01, ]

    configurations = list(
        itertools.product(hidden_size, learning_rate, batch_size, dropout, num_layers, weight_decay, window_sizes))

    for config in configurations:
        instantiate_train_amazon(config, seed, hparams)

if __name__ == "__main__":
    hparams = parser.parse_args()
    training_wrapper_amazon(9001,hparams)
