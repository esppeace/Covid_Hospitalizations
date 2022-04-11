import torch
import os
from argparse import ArgumentParser

from torch.nn.init import xavier_uniform_, xavier_normal_
import torch.nn.functional as F

from models.Transformer.Encoder import Encoder, EncoderLayer

import math
from loss import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()

        self.dropout = dropout
        if dropout > 0:
            self.dropout_layer = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        if self.dropout > 0:
            return self.dropout_layer(x)
        else:
            return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model_dim = args.model_dim
        self.input_dim = args.input_dim
        self.output_dim = args.output_dim
        self.dropout = args.dropout
        self.no_norm = args.no_norm
        self.num_head = args.num_head
        self.num_encoder_layer = args.num_encoder_layer
        self.feedforward_dim = args.feedforward_dim
        self.activation = args.activation
        self.seq_len = args.seq_len
        self.device = args.device
        self.optimizer = None
        self.scheduler = None
        self.learning_rate = args.learning_rate
        self.weight_decay = args.weight_decay
        self.half_epoch = args.half_epoch
        self.logging_interval = args.logging_interval
        self.gamma = args.gamma
        self.quantiles = args.quantiles

        self.input = nn.Linear(self.input_dim, self.model_dim)
        self.pos_encoder = PositionalEncoding(self.model_dim, self.dropout)
        if not self.no_norm:
            encoder_layers = EncoderLayer(self.model_dim,
                                                     self.num_head,
                                                     self.feedforward_dim,
                                                     self.dropout,
                                                     self.activation)
            encoder_norm = nn.LayerNorm(self.model_dim)
            self.encoder = Encoder(encoder_layers, self.num_encoder_layer, encoder_norm)
        else:
            encoder_layers = EncoderLayer(self.model_dim,
                                                     self.num_head,
                                                     self.feedforward_dim,
                                                     self.dropout,
                                                     self.activation)
            self.encoder = Encoder(encoder_layers, self.num_encoder_layer)
             
        self.decoder1 = nn.Linear(self.model_dim*self.seq_len, self.output_dim*1)
        self.decoder2 = nn.Linear(self.model_dim*self.seq_len, self.output_dim*(len(self.quantiles)))

        self._reset_parameters()
        self.configure_optimizers()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, x):
        # the transpose is there because they use len, batch, dim
        src = torch.transpose(x, 0, 1).contiguous()
        src = self.input(src) * math.sqrt(self.model_dim) # this is to prevent the original meaning of embedding doesnt loss afte the position embedding
        src = self.pos_encoder(src)
        output, _ = self.encoder(src)
        
        output = torch.transpose(output, 0, 1).contiguous()

        identity = torch.transpose(src, 0, 1).contiguous()
        output1 = F.relu(self.decoder1(torch.flatten(output+identity, start_dim=1)))
        output2 = F.relu(self.decoder2(torch.flatten(output+identity, start_dim=1)))

        return output1, output2

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.half_epoch, gamma=self.gamma)

def save_checkpoint(model, optimizer, loss, epoch, model_dir, best=False):
    output_dir = model_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not best:
        filename = output_dir +'/last.ckpt'
    else:
        filename = output_dir +'/best.ckpt'

    torch.save({'epoch': epoch, 
               'model_state_dict':model.state_dict(),
               'optimizer_state_dict':optimizer.state_dict(),
               'loss': loss}, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Trainer_Hospitalization:
    def __init__(self, args, logger=None):
        self.max_epochs = args.epochs
        self.loss = args.loss
        self.logging_interval = args.logging_interval
        self.huber_beta = args.huber_beta
        self.model_dir = args.model_dir
        self.best_validation_loss = 0.0
        self.num_of_week = args.num_of_week
        self.lambda1 = args.lambda1
        self.args = args
    

    def evaluate(self, model, device, valid_loader):
        model = model.to(device)
        model.eval()
        MAE = nn.L1Loss()
        losses = AverageMeter()

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                target_list_sum = torch.sum(target, dim=1)
                prediction, prediction2 = model(data)

                loss = MAE(prediction[:, 0:4], target_list_sum[:,1:1+self.num_of_week])
                
                losses.update(loss.item(), data.size(0))                
                
            print('Validation: MAE Loss:{:.6f}\n'.format(losses.avg))

        return losses.avg


    def fit(self, model, train_loader, valid_loader=None, valid_loader_state=None):
            optimizer = model.optimizer
            scheduler = model.scheduler
            device = model.device
            losses_point = AverageMeter()
            losses_quantile = AverageMeter()
            model = model.to(device) 
            for epoch in range(self.max_epochs):
                model.train()            
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    y_hat, y_hat2 = model(data)
                
                    loss_point = Huber_loss(y_hat[:, 0:4], target[:,:,1:1+self.num_of_week], self.huber_beta)
                                      
                    target_list_sum = torch.sum(target, dim=1)

                    wl1 = 0.25 * Quantile_loss(model.quantiles, y_hat2[:, 0:23], target_list_sum[:,1], model.device)
                    wl2 = 0.25 * Quantile_loss(model.quantiles, y_hat2[:, 23:46], target_list_sum[:,2], model.device)
                    wl3 = 0.25 * Quantile_loss(model.quantiles, y_hat2[:, 46:69], target_list_sum[:,3], model.device)
                    wl4 = 0.25 * Quantile_loss(model.quantiles, y_hat2[:, 69:95], target_list_sum[:,4], model.device)

                    loss  = self.lambda1*(wl1 + wl2 + wl3 + wl4)

                    losses_point.update(loss_point.item(), data.size(0))
                    losses_quantile.update(loss.item(), data.size(0))
                    
                    loss += loss_point

                    loss.backward()

                    optimizer.step()
                    
                    if self.logging_interval > 0 and batch_idx % self.logging_interval == 0:
                        print('Train Epoch: {} [{}/{}  ({:.0f}%)]\tLoss Point: {:.6f}\tLoss Quantile: {:.6f}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx/len(train_loader), losses_point.avg,losses_quantile.avg))
                    elif batch_idx == len(train_loader)-1 :
                        print('Train Epoch: {} [{}/{}  ({:.0f}%)]\tLoss Point: {:.6f}\tLoss Quantile: {:.6f}'.format(epoch, batch_idx, len(train_loader), 100. * batch_idx/len(train_loader), losses_point.avg,losses_quantile.avg))

                
                if epoch%10 == 0:
                    print('Running: ', self.model_dir)
                    if valid_loader_state is not None:
                        print('Evaluation .... \n')
                        _ = self.evaluate(model, device, valid_loader_state)
                        
                if valid_loader is not None:
                    valid_loss = self.evaluate(model, device, valid_loader)
                    if epoch == 0 or valid_loss < self.best_validation_loss:
                        self.best_validation_loss = valid_loss
                        save_checkpoint(model, optimizer, valid_loss, epoch, self.model_dir, best=True)
                
                scheduler.step()

            # saving checkpoint
            save_checkpoint(model, optimizer, losses_point.avg, epoch, self.model_dir)
 