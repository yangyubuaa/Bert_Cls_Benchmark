import torch

from transformers_model.models.bert.modeling_bert import BertModel

class BertModelForClassification(torch.nn.Module):
    def __init__(self):
        super(BertModelForClassification, self).__init__()
        self.bert_feature_layer = BertModel.from_pretrained("huggingface_pretrained_model/bert-base-chinese")
        self.classification_layer = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        deep_features = self.bert_feature_layer(input_ids, attention_mask, token_type_ids)
        cls_features = deep_features.last_hidden_state[:, 0, :]
        # print(cls_features.shape)  # [batch_size, 768]
        logits = self.classification_layer(cls_features)
        return logits

