from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import roc_auc_score

class CTR_Dataset(Dataset):
    def __init__(self, data_df):
        data_x_arr = data_df.drop(columns=['is_click']).values
        self.num_fields = data_x_arr.shape[1]//2
        self.x_id = torch.LongTensor(data_x_arr[:,:self.num_fields])
        self.x_value = torch.Tensor(data_x_arr[:,self.num_fields:])
        self.y = torch.Tensor(data_df['is_click'].values)

    def __getitem__(self, idx):
        return self.x_id[idx], self.x_value[idx], self.y[idx]

    def __len__(self):
        return self.x_id.shape[0]

class QueryWithSupportDataset(Dataset):
    def __init__(self, data_df, train_support_df, COLD_USER_THRESHOLD):
        self.data_x_arr = data_df.drop(columns=['is_click']).values
        self.num_fields = self.data_x_arr.shape[1]//2-1
        self.x_id = torch.LongTensor(self.data_x_arr[:,1:self.num_fields+1])
        self.x_value = torch.Tensor(self.data_x_arr[:,self.num_fields+2:])
        self.y = torch.Tensor(data_df['is_click'].values)
        self.train_support_df = train_support_df
        self.COLD_USER_THRESHOLD = COLD_USER_THRESHOLD

    def __getitem__(self, idx):
        uid=self.data_x_arr[idx][0].item()
        df = self.train_support_df[self.train_support_df['uid']==uid]
        data_x_arr = df.drop(columns=['is_click']).values
        x_id_support_arr = data_x_arr[:,1:self.num_fields+1]
        x_val_support_arr = data_x_arr[:,self.num_fields+2:]
        y_support_arr = df['is_click'].values
        if x_id_support_arr.shape[0]<self.COLD_USER_THRESHOLD:
            x_id_support_arr_paddding = np.array([[0]*self.num_fields]*(
                self.COLD_USER_THRESHOLD-x_id_support_arr[:self.COLD_USER_THRESHOLD].shape[0]))
            x_id_support_arr = np.concatenate([x_id_support_arr,x_id_support_arr_paddding],axis=0)
            x_val_support_arr_paddding = np.array([[0]*self.num_fields]*(
                self.COLD_USER_THRESHOLD-x_val_support_arr[:self.COLD_USER_THRESHOLD].shape[0]))
            x_val_support_arr = np.concatenate([x_val_support_arr,x_val_support_arr_paddding],axis=0)
            y_support_arr_padding =  np.array([-1]*(
                self.COLD_USER_THRESHOLD-y_support_arr[:self.COLD_USER_THRESHOLD].shape[0]))
            y_support_arr = np.concatenate([y_support_arr,y_support_arr_padding],axis=0)
        x_id_support = torch.LongTensor(x_id_support_arr)
        x_val_support = torch.Tensor(x_val_support_arr)
        y_support = torch.Tensor(y_support_arr)
        return self.x_id[idx], self.x_value[idx], self.y[idx], [x_id_support,x_val_support,y_support]

    def __len__(self):
        return self.x_id.shape[0]

# 将数据按照uid进行划分，输入uid，输出所有相关support set和query set样本
class QueryWithSupportDatasetFast(Dataset):
    def __init__(self, query_df, support_df):
        self.query_df = query_df
        self.support_df = support_df
        self.uids = query_df['uid'].unique()
        data_x_arr = query_df.drop(columns=['is_click']).values
        self.num_fields = data_x_arr.shape[1]//2
        self.x_id = torch.LongTensor(data_x_arr[:,:self.num_fields])
        self.x_value = torch.Tensor(data_x_arr[:,self.num_fields:])
        self.y = torch.Tensor(query_df['is_click'].values)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        #         uid=self.x_id[idx][0].item()
        support_df = self.support_df[self.support_df['uid']==uid]
        data_x_arr = support_df.drop(columns=['is_click']).values
        x_id_support_arr = data_x_arr[:,:self.num_fields]
        x_val_support_arr = data_x_arr[:,self.num_fields:]
        y_support_arr = support_df['is_click'].values

        query_df = self.query_df[self.query_df['uid']==uid]
        data_x_arr_query = query_df.drop(columns=['is_click']).values
        x_id_query_arr = data_x_arr_query[:,:self.num_fields]
        x_val_query_arr = data_x_arr_query[:,self.num_fields:]
        y_query_arr = query_df['is_click'].values
        x_id_support = torch.LongTensor(x_id_support_arr)
        x_val_support = torch.Tensor(x_val_support_arr)
        y_support = torch.Tensor(y_support_arr)
        x_id_query = torch.LongTensor(x_id_query_arr)
        x_val_query = torch.Tensor(x_val_query_arr)
        y_query = torch.Tensor(y_query_arr)
        return [x_id_query,x_val_query,y_query], [x_id_support,x_val_support,y_support]

    def __len__(self):
        return self.uids.shape[0]

def cal_gauc(labels, preds, user_id_list):
    """Calculate group auc"""
    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)))
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)
    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag
    impression_total = 0
    total_auc = 0
    true_list=[]
    pred_list=[]
    for user_id in group_flag:
        if group_flag[user_id]:
            true_list +=group_truth[user_id]
            pred_list +=group_score[user_id]
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    #             print(f"auc for {user_id}:{auc}, weight:{len(group_truth[user_id])} ")
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc


def val(model, val_dataloader, gauc_col, device='cuda'):
    model.eval()
    running_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    pred_arr = np.array([])
    label_arr = np.array([])
    group_id_arr = np.array([])
    with torch.no_grad():
        for itr, batch in tqdm(enumerate(val_dataloader)):
            batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
            feature_ids, feature_vals, labels = batch
            group_id_arr = np.hstack(
                [group_id_arr, feature_ids[:, gauc_col].data.detach().cpu()]) if group_id_arr.size else feature_ids[:,
                                                                                                        gauc_col].data.detach().cpu()
            outputs = model(feature_ids, feature_vals).squeeze()
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            pred_arr = np.hstack(
                [pred_arr, outputs.data.detach().cpu()]) if pred_arr.size else outputs.data.detach().cpu()
            label_arr = np.hstack(
                [label_arr, labels.data.detach().cpu()]) if label_arr.size else labels.data.detach().cpu()
        val_loss = running_loss / (itr + 1)
        torch.cuda.empty_cache()
    auc = roc_auc_score(label_arr, pred_arr)
    gauc = cal_gauc(label_arr.tolist(), pred_arr.tolist(), group_id_arr.tolist())
    return val_loss, auc, gauc

def val_query(model, val_dataloader, gauc_col, device='cuda'):
    model.eval()
    running_loss = 0
    #     criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    pred_arr = np.array([])
    label_arr = np.array([])
    dist_arr = np.array([])
    loss_arr = np.array([])
    group_id_arr = np.array([])
    val_size = 0
    for itr, batch in enumerate(tqdm(val_dataloader)):
        batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
        query_data, support_data = batch
        val_size += query_data[0].shape[1]
        query_set_y_pred, distance = model(support_data, query_data)
        group_id_arr = np.hstack(
            [group_id_arr, query_data[0][0][:, gauc_col].data.detach().cpu()]) \
            if group_id_arr.size else query_data[0][0][:, gauc_col].data.detach().cpu()
        loss = criterion(query_set_y_pred, query_data[2][0])
        running_loss += loss.item()
        pred_arr = np.hstack(
            [pred_arr, query_set_y_pred.data.detach().cpu()]) if pred_arr.size else query_set_y_pred.data.detach().cpu()
        label_arr = np.hstack(
            [label_arr, query_data[2][0].data.detach().cpu()]) if label_arr.size else query_data[2][0].data.detach().cpu()
    val_loss = running_loss / (itr + 1)
    torch.cuda.empty_cache()
    auc = roc_auc_score(label_arr, pred_arr)
    gauc = cal_gauc(label_arr.tolist(), pred_arr.tolist(), group_id_arr.tolist())
    return val_loss, auc, gauc

class DeepFM_encoder(nn.Module):
    def __init__(self, num_features, embedding_dim, num_fields, hidden_size=400):
        super(DeepFM_encoder, self).__init__()
        num_fields -= 1
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.num_fields = num_fields
        self.last_layer_dim = 400
        self.feature_embeddings = nn.Embedding(num_features, embedding_dim)
        torch.nn.init.xavier_normal_(self.feature_embeddings.weight)
        self.input_dim = embedding_dim * num_fields
        self.fc1 = nn.Linear(self.input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.last_layer_dim)
        self.fc4 = nn.Linear(self.last_layer_dim+self.embedding_dim, 1)


    def forward(self, feature_ids, feature_vals, return_hidden=False):
        # exclude uid feature field
        feature_ids = feature_ids[:,1:]
        feature_vals = feature_vals[:,1:]
        # None*F*K
        input_embeddings = self.feature_embeddings(feature_ids)
        input_embeddings *= feature_vals.unsqueeze(dim=2)
        # None*K
        square_sum = torch.sum(input_embeddings ** 2, dim=1)
        sum_square = torch.sum(input_embeddings, dim=1) ** 2
        # None*K
        hidden_fm = (sum_square - square_sum) / 2
        # None*(F*K)
        input_embeddings_flatten = input_embeddings.view(-1, self.input_dim)
        hidden = nn.ReLU()(self.fc1(input_embeddings_flatten))
        hidden = nn.ReLU()(self.fc2(hidden))
        hidden_dnn =  nn.ReLU()(self.fc3(hidden))
        hidden_encoder = torch.cat([hidden_fm, hidden_dnn],dim=1)
        prediction = self.fc4(hidden_encoder).squeeze(1)
        if return_hidden:
            return prediction, hidden_encoder
        else:
            return prediction