from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import logging
import torch.nn.functional as F

from mmt import datasets
from mmt import models
from mmt.trainers import Train_search, Validation
from mmt.evaluators import Evaluator
from mmt.utils.data import IterLoader
from mmt.utils.data import transforms as T
from mmt.utils.data.sampler import RandomMultipleGallerySampler
from mmt.utils.data.preprocessor import Preprocessor
from mmt.utils.logging import Logger
from mmt.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from mmt.utils.lr_scheduler import WarmupMultiStepLR

from mmt.models.sun_model_search import sun_model_search
import mmt.NAS.cnnutils as utl
from mmt.NAS.architect import Architect

start_epoch = best_mAP = 0

def get_data(name, data_dir, height, width, batch_size, workers, num_instances, iters=200):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = sorted(dataset.train)
    val_set = sorted(dataset.val)
    num_classes = dataset.num_train_pids

    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer
         ])

    val_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer
         ])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler_train = RandomMultipleGallerySampler(train_set, num_instances)
        sampler_val = RandomMultipleGallerySampler(val_set, num_instances)
    else:
        sampler_train = None
        sampler_val = None

    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler_train,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    val_loader = IterLoader(
        DataLoader(Preprocessor(val_set, root=dataset.images_dir,
                                transform=val_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler_val,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main():
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loadersargws.
    iters = args.iters if (args.iters>0) else None
    dataset_source, num_classes, train_loader, val_loader, test_loader = \
        get_data(args.dataset_source, args.data_dir, args.height,
                 args.width, args.batch_size, args.workers, args.num_instances, iters)

    # Create model
    model = sun_model_search( args.arch, num_features=args.features, dropout=args.dropout, num_classes=num_classes)
    model.cuda()
    #model = nn.DataParallel(model)

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}".format(start_epoch, best_mAP))

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test on source domain:")
        evaluator.evaluate(test_loader, dataset_source.query, dataset_source.gallery, cmc_flag=True, rerank=args.rerank)
        return

    logging.info("param size = %fMB", utl.count_parameters_in_MB(model))

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    trainer = Train_search(model, num_classes, margin=args.margin)

    validator = Validation(model, num_classes, margin=args.margin)

    architect = Architect(model, optimizer, num_classes, args)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        train_loader.new_epoch()

        genotype = model.genotype()
        logging.info('genotype = %s', genotype)
        T = 1

        print(F.softmax(model.alphas_cell1/T, dim=-1))
        print(F.softmax(model.alphas_cell2/T, dim=-1))
        print(F.softmax(model.alphas_cell3/T, dim=-1))
        print(F.softmax(model.alphas_cell4 / T, dim=-1))

        #################train#####################
        trainer.train(epoch, T,train_loader,val_loader,  optimizer,architect,lr,
                    train_iters=len(train_loader), print_freq=args.print_freq)

        #################val#####################
        validator.val(epoch, T, val_loader,
                      val_iters=len(val_loader), print_freq=args.print_freq)

        # if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
        #
        #     _, mAP = evaluator.evaluate(test_loader, dataset_source.query, dataset_source.gallery, cmc_flag=True)
        #
        #     is_best = mAP > best_mAP
        #     best_mAP = max(mAP, best_mAP)
        #     save_checkpoint({
        #         'state_dict': model.state_dict(),
        #         'epoch': epoch + 1,
        #         'best_mAP': best_mAP,
        #     }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
        #
        #     print('\n * Finished epoch {:3d}  source mAP: {:5.1%}  best: {:5.1%}{}\n'.
        #           format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        # evaluator.evaluate(test_loader_source, dataset_source.query, dataset_source.gallery, cmc_flag=True,
        #                    rerank=args.rerank)

        utl.save(model, os.path.join(args.save, 'weights.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training on the source domain")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='dukemtmc',
                        choices=datasets.names())

    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    #parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    parser.add_argument('--gpu_devices', default='1', type=str,
                        help='gpu device ids for CUDA_VISIBLE_DEVICES')

    parser.add_argument('--save', type=str, default='EXP', help='experiment name')

###############################architect
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    args = parser.parse_args()
    main()
