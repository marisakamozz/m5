import random
import logging
import pathlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from torch.distributions.negative_binomial import NegativeBinomial
import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from util import TARGET


def get_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        random.seed(worker_id + seed)
        np.random.seed(worker_id + seed)
    return worker_init_fn


class M5Dataset(torch.utils.data.Dataset):
    def __init__(self, v_sales_dict, data_count, features, weight, te, remove_last4w=0, min_data_4w=3, over_sample=True):
        super().__init__()
        data_dict = {}
        min_data_len = (remove_last4w + min_data_4w) * 28
        for data_id, data in v_sales_dict.items():
            if len(data) >= min_data_len:
                data_dict[data_id] = torch.tensor(data[features + [TARGET]].values, dtype=torch.float)
            else:
                data_count = data_count.drop(data_count[data_count['id'] == data_id].index)
        self.data_dict = data_dict
        data_count['len'] = data_count['count'] // 7 - 4 * (2 + remove_last4w) + 1
        self.id_index = []
        self.data_index = []
        self.weight_index = []
        for _, row in data_count[['id', 'len']].iterrows():
            data_id = row['id']
            data_len = row['len']
            if data_len > 0:
                v_sales = v_sales_dict[data_id]
                if over_sample:
                    monday_list = list(v_sales[v_sales['day_of_week'] == 0].index)[:data_len]
                    counter = 0
                    for year, count in v_sales.iloc[monday_list].groupby('year')['d'].count().iteritems():
                        if year == 2015:
                            sample_ratio = 2
                        elif year >= 2016:
                            sample_ratio = 4
                        else:
                            sample_ratio = 1
                        self.id_index += [data_id] * count * sample_ratio
                        self.data_index += monday_list[counter:counter+count] * sample_ratio
                        self.weight_index += [weight[data_id]] * count * sample_ratio
                        counter += count
                else:
                    self.id_index += [data_id] * data_len
                    self.data_index += list(v_sales[v_sales['day_of_week'] == 0].index)[:data_len]
                    self.weight_index += [weight[data_id]] * data_len
        first_date = v_sales_dict[self.id_index[0]].iloc[self.data_index[0]]['d']
        last_date = v_sales_dict[self.id_index[-1]].iloc[self.data_index[-1]]['d']
        logging.info(f'train from {first_date} to {last_date}')
        self.te = te

    def __len__(self):
        return len(self.id_index)
        
    def __getitem__(self, idx):
        data_id = self.id_index[idx]
        data_idx = self.data_index[idx]
        weight = self.weight_index[idx]
        input_sales = self.data_dict[data_id][data_idx:data_idx+28]
        output_sales = self.data_dict[data_id][data_idx+28:data_idx+56]
        x1 = input_sales[:, :-1].clone()
        y1 = input_sales[:, -1].clone()
        x2 = output_sales[:, :-1].clone()
        y2 = output_sales[:, -1].clone()
        weight = torch.tensor(weight, dtype=torch.float)
        te = torch.tensor(self.te[data_id], dtype=torch.float)
        return (x1, y1, x2, te), y2, weight
    

class M5ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, weight, te, remove_last4w=0, term=1):
        super().__init__()
        self.data_dict = data_dict
        self.id_index = []
        self.weight_index = []
        for key in data_dict.keys():
            self.id_index += [key] * (4*term)
            self.weight_index += [weight[key]] * (4*term)
        index = - 28 * (2 + remove_last4w + term)
        self.data_index = list(range(index-21, index-21+28*term, 7)) * len(data_dict)
        self.te = te
    
    def __len__(self):
        return len(self.data_dict) * 4
    
    def __getitem__(self, idx):
        data_id = self.id_index[idx]
        data_idx = self.data_index[idx]
        weight = self.weight_index[idx]
        input_sales = self.data_dict[data_id][data_idx:data_idx+28]
        if input_sales.size(0) < 28:
            data_idx += 7
            input_sales = self.data_dict[data_id][data_idx:data_idx+28]
        if data_idx + 56 == 0:
            output_sales = self.data_dict[data_id][data_idx+28:]
        else:
            output_sales = self.data_dict[data_id][data_idx+28:data_idx+56]
        x1 = input_sales[:, :-1].clone()
        y1 = input_sales[:, -1].clone()
        x2 = output_sales[:, :-1].clone()
        y2 = output_sales[:, -1].clone()
        weight = torch.tensor(weight, dtype=torch.float)
        te = torch.tensor(self.te[data_id], dtype=torch.float)
        return (x1, y1, x2, te), y2, weight


class M5TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, weight, te, remove_last4w=0, term=1):
        super().__init__()
        self.data_dict = data_dict
        self.id_index = list(data_dict.keys())
        self.data_index = - 28 * (2 + remove_last4w)
        self.weight_index = [weight[data_id] for data_id in self.id_index]
        self.te = te
        self.term = term
    
    def __len__(self):
        return len(self.data_dict) * self.term
    
    def __getitem__(self, idx):
        term_idx = idx // len(self.data_dict)
        idx -= term_idx * len(self.data_dict)
        data_id = self.id_index[idx]
        data_idx = self.data_index - 28 * term_idx
        weight = self.weight_index[idx]
        input_sales = self.data_dict[data_id][data_idx:data_idx+28]
        if data_idx + 56 == 0:
            output_sales = self.data_dict[data_id][data_idx+28:]
        else:
            output_sales = self.data_dict[data_id][data_idx+28:data_idx+56]
        x1 = input_sales[:, :-1].clone()
        y1 = input_sales[:, -1].clone()
        x2 = output_sales[:, :-1].clone()
        y2 = output_sales[:, -1].clone()
        weight = torch.tensor(weight, dtype=torch.float)
        te = torch.tensor(self.te[data_id], dtype=torch.float)
        return data_id, (x1, y1, x2, te), y2, weight


class M5MLPLSTMModel(nn.Module):
    def __init__(self, is_cats, dims, n_hidden=64, dropout=0.5, use_te=True):
        super().__init__()
        self.is_cats = is_cats
        cat_dims = dims['cat_dims']
        emb_dims = dims['emb_dims']
        n_input = len(is_cats) - sum(is_cats) + sum(emb_dims)
        self.emb_layers = nn.ModuleList([
            nn.Embedding(x, y) for x, y in zip(cat_dims, emb_dims)
        ])
        self.dropout1 = nn.Dropout(p=dropout)
        self.lstm1 = nn.LSTM(n_input + 1, n_hidden)
        self.dropout2 = nn.Dropout(p=dropout)
        self.lstm2 = nn.LSTM(n_input, n_hidden)
        self.dropout3 = nn.Dropout(p=dropout)
        n_input = n_hidden + 1 if use_te else n_hidden
        self.mlp_layers = nn.ModuleList([
            nn.Linear(n_input, 2) for _ in range(7)
        ])
        self.use_te = use_te

    def _embed(self, x):
        # x : batch_size * seq_len * n_features
        counter = 0
        embeddings = []
        for i, is_cat in enumerate(self.is_cats):
            if is_cat:
                embeddings.append(self.emb_layers[counter](x[:, :, i].long()))
                counter += 1
            else:
                embeddings.append(x[:, :, i].unsqueeze(-1))
        x = torch.cat(embeddings, dim=2)
        # x : batch_size * seq_len * n_input
        return x

    def forward(self, x):
        x1, y1, x2, te = x
        # x1 : batch_size * seq_len * n_features
        # x2 : batch_size * seq_len * n_features
        x1 = self._embed(x1)
        x2 = self._embed(x2)
        # x1 : batch_size * seq_len * n_input
        # x2 : batch_size * seq_len * n_input
        # y1 : batch_size * seq_len
        x1 = torch.cat([x1, y1.unsqueeze(-1)], dim=2)
        # x1 : batch_size * seq_len * (n_input+1)
        x1 = self.dropout1(x1.transpose(0, 1))
        x2 = self.dropout2(x2.transpose(0, 1))
        # x1 : seq_len * batch_size * (n_input+1)
        # x2 : seq_len * batch_size * (n_input)
        _, (h, c) = self.lstm1(x1)
        # h : 1 * batch_size * n_hidden
        # c : 1 * batch_size * n_hidden
        x2, (_, _) = self.lstm2(x2, (h, c))
        x2 = self.dropout3(x2)
        # x2 : seq_len * batch_size * n_hidden
        counter = 0
        output_list = []
        for _ in range(4):
            for i in range(7):
                # x2 : seq_len * batch_size * n_hidden
                if self.use_te:
                    # te : batch_size * 7
                    x3 = torch.cat([x2[counter, :, :], te[:, i].unsqueeze(-1)], dim=1)
                    # x3 : batch_size * n_hidden+1
                else:
                    x3 = x2[counter, :, :]
                    # x3 : batch_size * n_hidden
                output = self.mlp_layers[i](x3).unsqueeze(dim=1)
                # output : batch_size * 1 * 2
                output_list.append(output)
                counter += 1
        y2 = torch.cat(output_list, dim=1)
        # y2 : batch_size * seq_len * 2
        return y2


class M5Distribution():
    def __init__(self, dist='Normal', use_exp=True, df=1):
        self.dist = dist
        self.use_exp = use_exp
        self.df = df
    
    def _to_dist(self, dist_params):
        if self.dist == 'Normal':
            mean = dist_params[:, :, 0]
            if self.use_exp:
                std = dist_params[:, :, 1].exp()
            else:
                std = F.softplus(dist_params[:, :, 1])
            return Normal(mean, std)
        elif self.dist == 'StudentT':
            mean = dist_params[:, :, 0]
            if self.use_exp:
                std = dist_params[:, :, 1].exp()
            else:
                std = F.softplus(dist_params[:, :, 1])
            return StudentT(self.df, mean, std)
        elif self.dist == 'NegativeBinomial':
            if self.use_exp:
                total_count = dist_params[:, :, 0].exp()
            else:
                total_count = F.softplus(dist_params[:, :, 0])
            logits = dist_params[:, :, 1]
            return NegativeBinomial(total_count, logits=logits)
        else:
            raise NotImplementedError()

    def get_loss(self, dist_params, y):
        dist = self._to_dist(dist_params)
        loss = - dist.log_prob(y).mean()
        return loss

    def get_rmse(self, dist_params, y):
        y_pred = self.predict(dist_params)
        score = torch.sqrt(((y - y_pred) ** 2).mean())
        return score
    
    def get_wloss(self, dist_params, y, weight):
        batch_size = dist_params.size(0)
        dist = self._to_dist(dist_params)
        loss = - torch.dot(dist.log_prob(y).mean(dim=1), weight) / batch_size
        return loss

    def get_wrmse(self, dist_params, y, weight):
        batch_size = dist_params.size(0)
        y_pred = self.predict(dist_params)
        rmse = torch.sqrt(((y - y_pred) ** 2).mean(dim=1))
        score = torch.dot(rmse, weight) / batch_size
        return score
    
    def sample(self, dist_params, n_samples):
        dist = self._to_dist(dist_params)
        return dist.sample(torch.Size([n_samples]))
    
    def predict(self, dist_params):
        if self.dist == 'StudentT':
            return dist_params[:, :, 0]
        else:
            return self._to_dist(dist_params).mean


class M5LightningModule(pl.LightningModule):
    def __init__(self, model, criterion, train_loader, val_loader, test_loader, args):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.args = args
        self.hparams = {key:value for key, value in args.items()}

    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        if self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise NotImplementedError()
        if self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.T_max, eta_min=self.args.lr*0.1)
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # warm up lr
        if self.trainer.global_step < 500:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.args.lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def train_dataloader(self):
        return self.train_loader
        
    def training_step(self, batch, batch_nb):
        x, y, weight = batch
        dist_params = self.forward(x)
        loss = self.criterion.get_loss(dist_params, y)
        score = self.criterion.get_rmse(dist_params, y)
        wloss = self.criterion.get_wloss(dist_params, y, weight)
        wscore = self.criterion.get_wrmse(dist_params, y, weight)
        log = {
            'train_loss': loss,
            'train_rmse': score,
            'train_wloss': wloss,
            'train_wrmse': wscore
        }
        return {
            'loss': log[self.args.objective],
            'log': log
        }
        
    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_nb):
        x, y, weight = batch
        dist_params = self.forward(x)
        loss = self.criterion.get_loss(dist_params, y)
        score = self.criterion.get_rmse(dist_params, y)
        wloss = self.criterion.get_wloss(dist_params, y, weight)
        wscore = self.criterion.get_wrmse(dist_params, y, weight)
        return {
            'val_loss': loss,
            'val_rmse': score,
            'val_wloss': wloss,
            'val_wrmse': wscore
        }

    def validation_epoch_end(self, outputs):
        loss = torch.tensor([output['val_loss'] for output in outputs]).mean()
        score = torch.tensor([output['val_rmse'] for output in outputs]).mean()
        wloss = torch.tensor([output['val_wloss'] for output in outputs]).mean()
        wscore = torch.tensor([output['val_wrmse'] for output in outputs]).mean()
        return {
            'log': {
                'val_loss': loss,
                'val_rmse': score,
                'val_wloss': wloss,
                'val_wrmse': wscore
            }
        }

    def test_dataloader(self):
        return self.test_loader

    def test_step(self, batch, batch_idx):
        _, x, y, weight = batch
        dist_params = self.forward(x)
        loss = self.criterion.get_loss(dist_params, y)
        score = self.criterion.get_rmse(dist_params, y)
        wloss = self.criterion.get_wloss(dist_params, y, weight)
        wscore = self.criterion.get_wrmse(dist_params, y, weight)
        return {
            'test_loss': loss,
            'test_rmse': score,
            'test_wloss': wloss,
            'test_wrmse': wscore
        }

    def test_epoch_end(self, outputs):
        loss = torch.tensor([output['test_loss'] for output in outputs]).mean()
        score = torch.tensor([output['test_rmse'] for output in outputs]).mean()
        wloss = torch.tensor([output['test_wloss'] for output in outputs]).mean()
        wscore = torch.tensor([output['test_wrmse'] for output in outputs]).mean()
        return {
            'log': {
                'test_loss': loss,
                'test_rmse': score,
                'test_wloss': wloss,
                'test_wrmse': wscore
            }
        }


class M5Trainer(pl.Trainer):
    def __init__(self, experiment, name, max_epochs, min_epochs, patience, val_check_interval):
        # logger = TensorBoardLogger('../logs', name=name)
        tags = {'mlflow.runName': name}
        logger = MLFlowLogger(experiment, 'file:../logs/mlruns', tags)
        if patience == 0:
            early_stopping = False
        else:
            early_stopping = EarlyStopping(
                patience=patience,
                monitor='val_loss',
                mode='min'
            )
        # filepath = pathlib.Path('../logs') / name / f'version_{logger.version}' / 'model'
        filepath = pathlib.Path('../models') / name / 'model'
        model_checkpoint = ModelCheckpoint(
            str(filepath),
            monitor='val_loss',
            mode='min'
        )
        super().__init__(
            default_save_path='../logs',
            gpus=1,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            early_stop_callback=early_stopping,
            logger=logger,
            row_log_interval=100,
            checkpoint_callback=model_checkpoint,
            val_check_interval=val_check_interval,
        )
