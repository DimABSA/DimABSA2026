from transformers import BertModel
import torch.nn as nn


class DimABSA(nn.Module):
    def __init__(self, hidden_size, bert_model_type, num_category):

        super(DimABSA, self).__init__()

        self.bert = BertModel.from_pretrained(bert_model_type)

        self.classifier_a_start = nn.Linear(hidden_size, 2)
        self.classifier_a_end = nn.Linear(hidden_size, 2)
        self.classifier_ao_start = nn.Linear(hidden_size, 2)
        self.classifier_ao_end = nn.Linear(hidden_size, 2)
        self.classifier_o_start = nn.Linear(hidden_size, 2)
        self.classifier_o_end = nn.Linear(hidden_size, 2)
        self.classifier_oa_start = nn.Linear(hidden_size, 2)
        self.classifier_oa_end = nn.Linear(hidden_size, 2)
        self.classifier_category = nn.Linear(hidden_size, num_category)
        self.classifier_valence = nn.Linear(hidden_size, 1)
        self.classifier_arousal = nn.Linear(hidden_size, 1)
        # self.classifier_valence = nn.Linear(hidden_size, 9)
        # self.classifier_arousal = nn.Linear(hidden_size, 9)

    def forward(self, query_tensor, query_mask, query_seg, step):

        hidden_states = self.bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        if step == 'A':
            predict_start = self.classifier_a_start(hidden_states)
            predict_end = self.classifier_a_end(hidden_states)
            return predict_start, predict_end
        elif step == 'O':
            predict_start = self.classifier_o_start(hidden_states)
            predict_end = self.classifier_o_end(hidden_states)
            return predict_start, predict_end
        elif step == 'AO':
            predict_start = self.classifier_ao_start(hidden_states)
            predict_end = self.classifier_ao_end(hidden_states)
            return predict_start, predict_end
        elif step == 'OA':
            predict_start = self.classifier_oa_start(hidden_states)
            predict_end = self.classifier_oa_end(hidden_states)
            return predict_start, predict_end
        elif step == 'C':
            category_hidden_states = hidden_states[:, 0, :]
            category_scores = self.classifier_category(category_hidden_states)
            return category_scores
        elif step == 'Valence':
            valence_hidden_states = hidden_states[:, 0, :]
            valence_scores = self.classifier_valence(valence_hidden_states).squeeze(-1)
            # valence_scores = self.classifier_valence(valence_hidden_states)[:,-1]
            return valence_scores
        elif step == 'Arousal':
            arousal_hidden_states = hidden_states[:, 0, :]
            arousal_scores = self.classifier_arousal(arousal_hidden_states).squeeze(-1)
            # arousal_scores = self.classifier_arousal(arousal_hidden_states)[:,-1]
            return arousal_scores
        else:
            raise KeyError('step error.')
