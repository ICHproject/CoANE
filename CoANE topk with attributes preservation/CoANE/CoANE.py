import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CoANE(torch.nn.Module):
    def __init__(self, feat_dim, nb_filter, win_len, t_feat, drop = 0.5):
        super(CoANE, self).__init__()

        self.win_cnnEncoder1 = nn.Conv1d(in_channels = feat_dim, out_channels = int(nb_filter), kernel_size = win_len, stride=win_len)
        self.feat_emb = nn.Embedding.from_pretrained(t_feat)

        self.r1 = nn.Linear (int(nb_filter), int(nb_filter*2))
        self.r2 = nn.Linear (int(nb_filter*2), feat_dim)

        self.MSE = nn.MSELoss()
        self.drop = drop
        self.nb_node = t_feat.shape[0]
        self.feat_dim = feat_dim
        self.nb_filter = nb_filter
        
    def forward(self, x):
        feat_ = torch.LongTensor(x[0].tolist()).to(self.device)

        gather_feat = self.feat_emb(feat_)
        gather_feat = F.dropout(gather_feat, self.drop, training=self.training)*(1-self.drop)

        gather_feat_flat = torch.transpose(gather_feat, 1, 2)

        win_Encoder_feat = self.win_cnnEncoder1(gather_feat_flat).squeeze(-1)
        win_Encoder_feat = F.dropout(win_Encoder_feat, self.drop, training=self.training)
        
        #reorder the node index (total node) based on current index (batch)
        l_map = {i:ix for ix, i in enumerate(x[2])}
        labels = torch.LongTensor(list(map(lambda z: l_map[z],x[1]))).to(self.device)
        _, labels_count = labels.unique(dim=0, return_counts=True)

        labels = labels.view(-1, 1).expand(-1, int(self.nb_filter))
        feat_avg = torch.zeros((len(x[2]), win_Encoder_feat.size(1))).to(self.device).scatter_add_(0, labels, win_Encoder_feat)
        feat_avg  = feat_avg / labels_count.float().unsqueeze(1)

        return win_Encoder_feat, feat_avg

    def forward_f(self, x):
        x = torch.relu(self.r1(x))
        x = F.dropout(x, self.drop, training=self.training)
        x = self.r2(x)
        return x
