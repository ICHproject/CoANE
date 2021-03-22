import torch
import torch.nn as nn
import torch.nn.functional as F

class CoANE(torch.nn.Module):
    def __init__(self, feat_dim, nb_filter, win_len, t_feat, drop = 0.5):
        super(CoANE, self).__init__()

        self.win_cnnEncoder1 = nn.Conv1d(in_channels = feat_dim, out_channels = nb_filter, kernel_size = win_len, stride=win_len)
        self.feat_emb = nn.Embedding.from_pretrained(t_feat)
        
        self.r1 = nn.Linear (nb_filter, int(nb_filter*2))
        self.r2 = nn.Linear (int(nb_filter*2), feat_dim)
        self.MSE = nn.MSELoss()
        self.drop = drop
        
    def forward(self, x):
        gather_feat = self.feat_emb(x[0])
        gather_feat_flat = torch.transpose(gather_feat, 1, 2)
        x_average_no = x[1]
        
        win_Encoder_feat = self.win_cnnEncoder1(gather_feat_flat)
        
        feat_pool = torch.sum(win_Encoder_feat, dim=2)
        feat_avg = torch.transpose(x_average_no*torch.transpose(feat_pool,0,1),0,1)
        
        return win_Encoder_feat, feat_avg

    def forward_f(self, x):
        x = torch.relu(self.r1(x))
        x = F.dropout(x, self.drop, training=self.training)
        x = torch.relu(self.r2(x))
        return x

