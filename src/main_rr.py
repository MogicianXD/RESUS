import time
import os
from collections import defaultdict
import gc

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

from utils import CTR_Dataset,QueryWithSupportDatasetFast,val,val_query,DeepFM_encoder
from resus import RESUS_NN, RESUS_RR


dataset = 'ml-1m'
PATH = f'../data/'
COLD_USER_THRESHOLD = 30
batch_size = 1024
embedding_dim = 10
device = torch.device('cuda:1')
lr = 1e-3
num_epochs = 100
overfit_patience = 2
exp_id=0
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

train_df=pd.read_csv(PATH+'train_df.csv')
val_df = pd.read_csv(PATH+f'valid_df.csv')
test_df = pd.read_csv(PATH+f'test_df.csv')

# dataframe->pytorch dataset
train_dataset = CTR_Dataset(train_df)
val_dataset = CTR_Dataset(val_df)
test_dataset = CTR_Dataset(test_df)
num_fields = train_dataset.num_fields
num_features = 1+max([x.x_id.max().item() for x in [train_dataset, val_dataset, test_dataset]])

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

# load encoder
model = DeepFM_encoder(num_features, embedding_dim, num_fields)
model = model.to(device)
checkpoint = torch.load(f"checkpoint/predictor-{dataset}-{exp_id}.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
encoder = DeepFM_encoder(num_features, embedding_dim, num_fields)
encoder = encoder.to(device)

rr = RESUS_RR(num_fields, COLD_USER_THRESHOLD, encoder, model).to(device)
optimizer = torch.optim.Adam(
    [
        #         {"params": rr.predictor.parameters(), "lr": 0.001},
        {"params": rr.encoder.parameters(), "lr": 0.001},
        {"params": rr.adjust.parameters(), "lr": 0.01},
        #         {"params": rr.lambda_rr.parameters(), "lr": 0.1},
    ],
)

best_loss = np.inf
best_epoch = -1
best_auc = 0.5
train_df_gb_uid = train_df.groupby('uid')
num_users = max(train_df_gb_uid.groups.keys())+1

val_df_support = val_df.groupby('uid').apply(lambda x: x[:COLD_USER_THRESHOLD] if len(x)>COLD_USER_THRESHOLD else x[:-1])
val_df_query = val_df.groupby('uid').apply(lambda x: x[COLD_USER_THRESHOLD:] if len(x)>COLD_USER_THRESHOLD else x[-1:])

val_query_dataset = QueryWithSupportDatasetFast(val_df_query,val_df_support)
val_query_dataloader = DataLoader(val_query_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

for epoch in range(num_epochs):
    print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")

    split_point = {}
    def sample_func(x):
        uid = x.iloc[0]['uid']
        sample_num = np.random.randint(1, min(len(x), COLD_USER_THRESHOLD+1))
        split_point[uid] = sample_num
        return x[:sample_num]
    train_support_df = train_df_gb_uid.apply(sample_func).reset_index(level=0, drop=True)

    def query_func(x):
        uid = x.iloc[0]['uid']
        sample_num = split_point[uid]
        return x[sample_num:]
    train_query_df = train_df_gb_uid.apply(query_func).reset_index(level=0, drop=True)
    #     train_query_df = pd.concat([train_df, train_support_df]).drop_duplicates(keep=False)
    train_query_dataset = QueryWithSupportDatasetFast(train_query_df,train_support_df)
    train_query_dataloader = DataLoader(train_query_dataset, 1, shuffle=True, num_workers=8, pin_memory=True)

    # Start training
    rr.train()
    running_loss = 0
    for itr, batch in enumerate(tqdm(train_query_dataloader)):
        batch = [[e.to(device) for e in item] if isinstance(item, list) else item.to(device) for item in batch]
        #         feature_ids, feature_vals, labels, support_data = batch
        query_data, support_data = batch
        outputs, _ = rr(support_data, query_data)
        #         outputs, predictor, delta_W_X, delta_W = knn(feature_ids, feature_vals, support_data)
        labels = query_data[2][0]
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        running_loss += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = running_loss / (itr+1)
    print(f"training loss of epoch {epoch}: {epoch_loss}")
    torch.cuda.empty_cache()

    print(f"Starting epoch: {epoch} | phase: val | ⏰: {time.strftime('%H:%M:%S')}")
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "best_auc": best_auc,
        "model": rr.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    rr.eval()
    val_loss, val_auc, val_gauc = val_query(rr, val_query_dataloader, gauc_col=0, device=device)
    print(f"validation loss of epoch {epoch}: {val_loss}, auc: {val_auc}, gauc: {val_gauc}")
    if val_auc > best_auc:
        print("******** New optimal found, saving state ********")
        patience = overfit_patience
        state["best_loss"] = best_loss = val_loss
        state["best_auc"] = best_auc = val_auc
        best_epoch = epoch
        torch.save(state, f"checkpoint/RESUS_RR-fast-{dataset}-2.tar")
    else:
        patience -= 1
    if optimizer.param_groups[0]['lr'] <= 1e-7:
        print('LR less than 1e-7, stop training...')
        break
    if patience == 0:
        print('patience == 0, stop training...')
        break
    del train_support_df
    del train_query_df
    del train_query_dataset
    del train_query_dataloader
    gc.collect()

# fine-grained test on rr model
print(f"Starting test | ⏰: {time.strftime('%H:%M:%S')}")
model = DeepFM_encoder(num_features, embedding_dim, num_fields)
encoder = DeepFM_encoder(num_features, embedding_dim, num_fields)
rr = RESUS_RR(num_fields, COLD_USER_THRESHOLD, encoder, model).to(device)
checkpoint = torch.load(f"checkpoint/RESUS_RR-fast-{dataset}-2.tar", map_location=torch.device('cpu'))
rr.load_state_dict(checkpoint['model'])

rr_test_losses = []
rr_test_aucs = []

for i in range(1,COLD_USER_THRESHOLD+1,1):
    # omit users with <= i interactions.
    test_support_set = test_df.groupby('uid',as_index=False).apply(
        lambda x: x[:i] if len(x)>i else x[:0])
    test_query_set = test_df.groupby('uid',as_index=False).apply(
        lambda x: x[i:] if len(x)>i else x[:0])
    test_query_dataset = QueryWithSupportDatasetFast(test_query_set,test_support_set)
    test_query_dataloader = DataLoader(test_query_dataset, 1, shuffle=False, num_workers=8, pin_memory=True)

    test_loss, test_auc, test_gauc = val_query(rr, test_query_dataloader, gauc_col=0, device=device)
    print(f"test loss of user group {i}: {test_loss}, auc: {test_auc}, gauc: {test_gauc}")

    rr_test_losses += [test_loss]
    rr_test_aucs += [test_auc]

print('test losses')
for loss in rr_test_losses:
    print(loss)
print('test aucs')
for loss in rr_test_aucs:
    print(loss)

print(f"resus_nn cold start I: Loss: {sum(rr_test_losses[:10])/10}, auc: {sum(rr_test_aucs[:10])/10}")
print(f"resus_nn cold start II: Loss: {sum(rr_test_losses[10:20])/10}, auc: {sum(rr_test_aucs[10:20])/10}")
print(f"resus_nn cold start III: Loss: {sum(rr_test_losses[20:30])/10}, auc: {sum(rr_test_aucs[20:30])/10}")
