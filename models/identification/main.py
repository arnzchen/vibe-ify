import os
import random
import time
import pickle
import sys
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn

# TODO imports
from util import save_checkpoint, ProgressMeter, AverageMeter, num_params
from constants import *
from data import Dataset
from model import Model


def train(model, dataset, optimizer, criterion, epoch, args, data_start_index):
    model.train()
    if data_start_index == 0:
        dataset.shuffle('train', seed=epoch + args.seed)
    if args.epoch_max_len is not None:
        data_end_index = min(data_start_index + args.epoch_max_len, len(dataset.splits['train']))
        loader = dataset.loader('train', batch_size=args.batch_size, num_workers=args.num_workers, indices=list(range(data_start_index, data_end_index)))
        data_start_index = data_end_index if data_end_index < len(dataset.splits['train']) else 0
    else:
        loader = dataset.loader('train', batch_size=args.batch_size, num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    indiv_keys = []
    indiv_meters = {key: AverageMeter(key, ':6.4f') for key in indiv_keys}
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter] + list(indiv_meters.values()), prefix='Training: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = [tensor.to(args.device) for tensor in batch]
        # batch = {key: value.to(args.device) if hasattr(value, 'to') else value for key, value in batch.items()}
        inputs, labels = batch # TODO adjust as needed
        scores = model(inputs)
        loss = criterion(scores, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.detach().item(), len(labels))
        # for key in indiv_keys:
        #     indiv_meters[key].update(losses[key], len(labels))
        #     # TODO update indiv meters
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return data_start_index


@torch.no_grad()
def validate(model, dataset, criterion, epoch, args):
    model.eval()
    loader = dataset.loader('dev', batch_size=args.batch_size, num_workers=args.num_workers)
    loss_meter = AverageMeter('loss', ':6.4f')
    indiv_keys = []
    indiv_meters = {key: AverageMeter(key, ':6.4f') for key in indiv_keys}
    total_length = len(loader)
    progress = ProgressMeter(total_length, [loss_meter] + list(indiv_meters.values()), prefix='Validation: ')
    for batch_num, batch in enumerate(tqdm(loader, total=len(loader))):
        batch = [tensor.to(args.device) for tensor in batch]
        # batch = {key: value.to(args.device) if hasattr(value, 'to') else value for key, value in batch.items()}
        inputs, labels = batch # TODO adjust as needed
        scores = model(inputs)
        loss = criterion(scores, labels)
        loss_meter.update(loss.item(), len(labels))
        # for key in indiv_keys:
        #     indiv_meters[key].update(losses[key], len(labels))
        #     # TODO update indiv meters
        if batch_num % args.train_print_freq == 0:
            progress.display(batch_num)
    progress.display(total_length)
    return loss_meter.avg


def main(args):
    dataset = Dataset(args)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'dataset_info'), 'wb') as wf:
        pickle.dump(dataset.dataset_info, wf)
    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=args.device)
        start_epoch = checkpoint['epoch'] + 1
        best_val_metric = checkpoint['best_metric']
        model_args = checkpoint['args']
        model = Model(model_args)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=model_args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        data_start_index = checkpoint['data_start_index']
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.ckpt, checkpoint['epoch']))
    else:
        model = Model(args)
        model = model.to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best_val_metric = 1e8 # lower is better for cross entropy
        data_start_index = 0
    print('num params', num_params(model))
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(args.device) # TODO adjust as needed
    
    if args.evaluate:
        epoch = 0
        validate(model, dataset, criterion, epoch, args)
        return
    for epoch in range(args.epochs):
        print("TRAINING: Epoch {} at {}".format(epoch, time.ctime()))
        data_start_index = train(model, dataset, optimizer, criterion, epoch, args, data_start_index)
        if epoch % args.validation_freq == 0:
            print("VALIDATION: Epoch {} at {}".format(epoch, time.ctime()))
            metric = validate(model, dataset, criterion, epoch, args)

            if metric < best_val_metric:
                print('new best val metric', metric)
                best_val_metric = metric
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': best_val_metric,
                    'optimizer': optimizer.state_dict(),
                    'data_start_index': data_start_index,
                    'args': args
                }, os.path.join(args.save_dir, 'model_best.pth.tar'))
            if args.save_individual_ckpts:
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_metric': metric,
                    'optimizer': optimizer.state_dict(),
                    'data_start_index': data_start_index,
                    'args': args
                }, os.path.join(args.save_dir, 'model_epoch' + str(epoch) + '.pth.tar'))
    


if __name__=='__main__':
    parser = ArgumentParser()

    # DATA
    parser.add_argument('--data-dir', type=str, required=True, help='where to load data from')
    parser.add_argument('--save-dir', type=str, required=True, help='where to save ckpts')
    parser.add_argument('--ckpt', type=str, default=None, help='load ckpt from file if given')
    parser.add_argument('--dataset-info', type=str, default=None, help='load dataset info from file if given')

    # MODEL ARCHITECTURE
    # TODO

    # TRAINING
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--epoch-max-len', type=int, default=None)
    parser.add_argument('--validation-freq', type=int, default=1, help='validate every X epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--num-workers', type=int, default=20, help='num workers for data loader')
    parser.add_argument('--save-individual-ckpts', action='store_true', default=False, help='save every validation epoch')
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)

    # PRINTING
    parser.add_argument('--train-print-freq', type=int, default=100, help='how often to print metrics (every X batches)')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.evaluate:
        assert args.ckpt is not None

    main(args)