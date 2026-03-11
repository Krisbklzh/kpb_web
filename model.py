import torch
import torch.nn as nn

# -----------------------------
# Класс модели
# -----------------------------
class MyModel(nn.Module):
    def __init__(self, input_shape, embeddings_shape):
        super(MyModel, self).__init__()
        self.relu = nn.ReLU()
        self.embeddings = nn.ModuleList(nn.Embedding(dim1, dim2) for dim1, dim2 in embeddings_shape)
        len_emb = sum([dim2 for dim1, dim2 in embeddings_shape])
        self.linear_1 = nn.Linear(input_shape + len_emb, 128)
        self.dropout1 = nn.Dropout(0.35)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear_2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.35)
        self.bn2 = nn.BatchNorm1d(64)
        self.linear_3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.35)
        self.bn3 = nn.BatchNorm1d(32)
        self.linear_4 = nn.Linear(32, 16)
        self.dropout4 = nn.Dropout(0.35)
        self.bn4 = nn.BatchNorm1d(16)
        self.linear_5 = nn.Linear(16, 1)

    def forward(self, num_batch, cat_batch):
        x_cat = [embedding_layer(cat_batch[:, i]) for i, embedding_layer in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, 1)
        x = torch.cat([x_cat, num_batch], 1)
        x = self.relu(self.linear_1(x))
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.relu(self.linear_2(x))
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.relu(self.linear_3(x))
        x = self.dropout3(x)
        x = self.bn3(x)
        x = self.relu(self.linear_4(x))
        x = self.dropout4(x)
        x = self.bn4(x)
        out = self.linear_5(x)
        return out