import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import transpose as t
from torch import inverse as inv
from torch import mm,solve,matmul

# class AdjustLayer(nn.Module):
#     def __init__(self, init_scale=0.4, num_adjust=None, init_bias=0, base=1):
#         super().__init__()
#         self.scale = nn.Parameter(torch.FloatTensor([init_scale for i in range(num_adjust)]).unsqueeze(1))
#         self.bias = nn.Parameter(torch.FloatTensor([init_bias for i in range(num_adjust)]).unsqueeze(1))

#     def forward(self, x, num_samples):
#         return x * (10**self.scale[num_samples-1]) + self.bias[num_samples-1]

class AdjustLayer(nn.Module):
    def __init__(self, init_scale=6, num_adjust=None, init_bias=0, base=1):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_scale for i in range(num_adjust)]).unsqueeze(1))
        self.bias = nn.Parameter(torch.FloatTensor([init_bias for i in range(num_adjust)]).unsqueeze(1))

    def forward(self, x, num_samples):
        return x * torch.abs(self.scale[num_samples-1]) + self.bias[num_samples-1]

class RESUS_NN(nn.Module):
    def __init__(self, num_fields, COLD_USER_THRESHOLD, encoder, predictor):
        super(RESUS_NN, self).__init__()
        self.num_fields = num_fields
        self.COLD_USER_THRESHOLD = COLD_USER_THRESHOLD
        self.predictor = predictor
        self.encoder = encoder
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(num_adjust=COLD_USER_THRESHOLD)
        self.fc1 = nn.Linear(self.encoder.last_layer_dim+self.encoder.embedding_dim, 1)

    def forward(self, support_set, query_set):
        # support_set: [x_id_support,x_val_support,y_support]
        # x_id_support: 1*sup_size*num_fields
        # x_val_support: 1*sup_size*num_fields
        # y_support: 1*sup_size
        support_set_x = [support_set[0][0],support_set[1][0]]
        support_set_y = support_set[2] # 1*sup_size
        query_set_x = [query_set[0][0],query_set[1][0]]
        num_samples = support_set_y.shape[1]
        _, support_set_encode = self.encoder(support_set_x[0],support_set_x[1], return_hidden=True) # sup_size*encode_dim
        _, query_set_encode = self.encoder(query_set_x[0], query_set_x[1], return_hidden=True) # query_size*encode_dim
        support_set_predict = self.predictor(support_set_x[0], support_set_x[1], return_hidden=False).unsqueeze(0) # 1*sup_size
        query_set_predict = self.predictor(query_set_x[0], query_set_x[1], return_hidden=False) # query_size
        distance = torch.abs(query_set_encode.unsqueeze(1)-support_set_encode.unsqueeze(0))  # query_size*sup_size*encode_dim
        similar_score = self.fc1(distance).squeeze(2) # query_size*sup_size
        similar_score_normalized = nn.Softmax(dim=1)(similar_score*1) # query_size*sup_size
        delta_y = support_set_y-nn.Sigmoid()(support_set_predict) # 1*sup_size
        delta_y_hat = (delta_y*similar_score_normalized).sum(1) # query_size
        prediction = self.adjust(delta_y_hat, num_samples) + query_set_predict  # query_size
        return prediction, torch.sqrt((distance ** 2).sum(-1)).mean(-1)


class LambdaLayer(nn.Module):
    def __init__(self, learn_lambda=True, num_lambda=None, init_lambda=1, base=1):
        super().__init__()
        self.l = torch.FloatTensor([init_lambda]) # COLD
        self.base = base
        self.l = nn.Parameter(self.l, requires_grad=learn_lambda)

    def forward(self, x, n_samples):
        #   x: None*COLD*COLD
        #   n_samples: None
        return x * torch.abs(self.l.unsqueeze(1).unsqueeze(2))

# RR
class RESUS_RR(nn.Module):
    def __init__(self, num_fields, COLD_USER_THRESHOLD, encoder, predictor):
        super(RESUS_RR, self).__init__()
        self.num_fields = num_fields
        self.COLD_USER_THRESHOLD = COLD_USER_THRESHOLD
        self.predictor = predictor
        self.encoder = encoder
        self.lambda_rr = LambdaLayer(learn_lambda=True, num_lambda=COLD_USER_THRESHOLD)
        self.L = nn.CrossEntropyLoss()
        self.adjust = AdjustLayer(num_adjust=COLD_USER_THRESHOLD)

    def rr_standard(self, x, n_samples, yrr_binary, linsys=False):
        #         x /= n_samples
        I = torch.eye(x.shape[1]).to(x)

        if not linsys:
            w = mm(mm(inv(mm(t(x, 0, 1), x) + self.lambda_rr(I)), t(x, 0, 1)), yrr_binary)
        else:
            A = mm(t_(x), x) + self.lambda_rr(I)
            v = mm(t_(x), yrr_binary)
            w, _ = solve(v, A)

        return w

    def rr_woodbury(self, X, n_samples, yrr_binary, linsys=False):
        #   X: None*COLD_USER_THRESHOLD*(hidden_size+1)
        #   n_samples: None
        #         x = X/torch.sqrt(n_samples.float()).unsqueeze(1).unsqueeze(2)    #   x: None*COLD*(hidden+1)
        x = X
        I = torch.eye(x.shape[1]).unsqueeze(0).repeat(x.shape[0],1,1).to(x)    # None*COLD*COLD
        if not linsys:
            w = matmul(matmul(t(x, 1, 2), inv(matmul(x, t(x, 1, 2)) + self.lambda_rr(I, n_samples))), yrr_binary)
        else:
            A = mm(x, t_(x)) + self.lambda_rr(I)
            v = yrr_binary
            w_, _ = solve(v, A)
            w = mm(t_(x), w_)
        return w

    def forward(self, support_set, query_set):
        # support_set: [x_id_support,x_val_support,y_support]
        # x_id_support: 1*sup_size*num_fields
        # x_val_support: 1*sup_size*num_fields
        # y_support: 1*sup_size
        support_set_x = [support_set[0][0],support_set[1][0]]
        support_set_y = support_set[2] # 1*sup_size
        query_set_x = [query_set[0][0],query_set[1][0]]
        num_samples = support_set_y.shape[1]
        _, support_set_encode = self.encoder(support_set_x[0],support_set_x[1], return_hidden=True) # sup_size*encode_dim
        _, query_set_encode = self.encoder(query_set_x[0], query_set_x[1], return_hidden=True) # query_size*encode_dim
        support_set_predict = self.predictor(support_set_x[0], support_set_x[1], return_hidden=False).unsqueeze(0) # 1*sup_size
        query_set_predict = self.predictor(query_set_x[0], query_set_x[1], return_hidden=False) # query_size

        ones = torch.ones((1,support_set_encode.shape[0],1)).to(support_set_encode) # 1*sup_size*1
        support_set_encode_prime = torch.cat((support_set_encode.unsqueeze(0), ones), 2) # 1*sup_size*(encode_dim+1)
        # 1*(encode_dim+1)*1
        delta_W = self.rr_woodbury(support_set_encode_prime, num_samples, support_set_y.unsqueeze(2)-nn.Sigmoid()(support_set_predict.unsqueeze(2))) # None*(hidden_size+1)*1
        delta_w = delta_W[0,:-1] # encode_dim*1
        delta_b = delta_W[0,-1] # 1
        out = matmul(query_set_encode, delta_w).squeeze(1) + delta_b # query_size
        prediction = self.adjust(out, num_samples) + query_set_predict # query_size
        return prediction, out