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

from utils import CTR_Dataset,QueryWithSupportDataset,val,val_query,DeepFM_encoder


dataset = 'ml-1m'
PATH = f'../data/'
COLD_USER_THRESHOLD = 30
batch_size = 1024
embedding_dim = 10
device = torch.device('cuda:0')
lr = 1e-3
num_epochs = 100
overfit_patience = 4
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

"""
Pretrain the shared predictor \Psi
"""
model = DeepFM_encoder(num_features, embedding_dim, num_fields)
# torch.nn.init.xavier_normal_(model.feature_embeddings.weight)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=overfit_patience)

best_loss = np.inf
best_epoch = -1
best_auc = 0.5
for epoch in range(num_epochs):
    print(f"Starting epoch: {epoch} | phase: train | ⏰: {time.strftime('%H:%M:%S')}")
    model.train()
    running_loss = 0
    for itr, batch in enumerate(tqdm(train_dataloader)):
        batch = [item.to(device) for item in batch]
        feature_ids, feature_vals, labels = batch
        if feature_ids.shape[0]==1:
            break
        outputs = model(feature_ids, feature_vals).squeeze()
        loss = torch.nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        running_loss += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()
    epoch_loss = running_loss / (itr+1)
    print(f"training loss of epoch {epoch}: {epoch_loss}")
    torch.cuda.empty_cache()

    # Validation.
    print(f"Starting epoch: {epoch} | phase: val | ⏰: {time.strftime('%H:%M:%S')}")
    state = {
        "epoch": epoch,
        "best_loss": best_loss,
        "best_auc": best_auc,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    val_loss, val_auc, val_gauc = val(model, val_dataloader, gauc_col=0, device=device)
    scheduler.step(-val_auc)
    print(f"validation loss of epoch {epoch}: {val_loss}, auc: {val_auc}, gauc: {val_gauc}")
    if val_auc > best_auc:
        print("******** New optimal found, saving state ********")
        patience = overfit_patience
        state["best_loss"] = best_loss = val_loss
        state["best_auc"] = best_auc = val_auc
        best_epoch = epoch
        torch.save(state, f"checkpoint/predictor-{dataset}-{exp_id}.tar")
    else:
        patience -= 1
    if optimizer.param_groups[0]['lr'] <= 1e-7:
        print('LR less than 1e-7, stop training...')
        break
    if patience == 0:
        print('patience == 0, stop training...')
        break

# Test, load the best checkpoint on val set
print(f"Starting test | ⏰: {time.strftime('%H:%M:%S')}")
checkpoint = torch.load(f"checkpoint/predictor-{dataset}-{exp_id}.tar", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
val_loss, val_auc, val_gauc = val(model, val_dataloader, gauc_col=0, device=device)
print(f"validation loss of best epoch {best_epoch}: {val_loss}, auc: {val_auc}, gauc: {val_gauc}")
test_loss, test_auc, test_gauc = val(model, test_dataloader, gauc_col=0, device=device)
print(f"test loss: {test_loss}, auc: {test_auc}, gauc: {test_gauc}")