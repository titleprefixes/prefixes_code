from torch import nn

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
