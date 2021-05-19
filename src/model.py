import torch
import torch.nn as nn
import torch.nn.functional as F


class CharLanguageModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, pad_id, dropout=0.1):
        super(CharLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=pad_id)
        self.rnn_model = nn.GRU(
            embedding_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        self.model_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, data):
        embeddings = self.embedding(data)
        rnn_output = self.rnn_model(embeddings)[0]
        rnn_output = F.relu(rnn_output)
        return self.model_head(rnn_output)
