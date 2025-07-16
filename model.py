from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence


class FineTuneBertMultiClassCLS(nn.Module):
    def __init__(
            self,
            bert,
            num_features_in_last_layer=None,
            num_classes=2,

    ):
        super(FineTuneBertMultiClassCLS, self).__init__()
        self.bert = bert
        if callable(num_features_in_last_layer):
            out_sz = num_features_in_last_layer(bert)
        elif isinstance(num_features_in_last_layer, int):
            out_sz = num_features_in_last_layer
        else:
            raise Exception('Misspecified out_sz')

        self.hidden2tag = nn.Linear(out_sz, num_classes)

    def forward(self, inp):
        input_ids, attention_mask, token_type_ids = inp
        token_out = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)['last_hidden_state']
        cls_repr = token_out[:,0,:]
        return self.hidden2tag(cls_repr)



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
