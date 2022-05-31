from __future__ import print_function, absolute_import
import time

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy,  SimpleCSE
from .utils.meters import AverageMeter
from torch.autograd import Variable
class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, target)*(self.temp_factor**2)/input.size(0)
        return loss

temp = 8
omega = 0.5
kdloss = KDLoss(temp)
softmax = nn.Softmax(dim=1)
def sun_knowledge_ensemble(feats, logits, targets):
    batch_size = logits.size(0)
    id_num = len(targets.unique())
    instance = int(batch_size / id_num)

    masks = torch.eye(instance)
    masks = masks.cuda()

    feats11 = feats.chunk(id_num, 0)
    logits11 = logits.chunk(id_num, 0)
    Q1 = []
    for i in range(id_num):
        feats22 = nn.functional.normalize(feats11[i], p=2, dim=1)
        logits22 = nn.functional.softmax(logits11[i]/temp, dim=1)
        W = torch.matmul(feats22, feats22.permute(1, 0)) - masks * 1e9
        W = softmax(W)
        W = (1 - omega) * torch.inverse(masks - omega * W)
        Q = torch.matmul(W, logits22)
        Q1.append(Q)
    Q1 = torch.cat(Q1)
    return Q1


class Train_search(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Train_search, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def train(self, epoch, T, data_loader_source, val_loader, optimizer,architect, lr,train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_kd = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_source.next()
            val_inputs = val_loader.next()

            data_time.update(time.time() - end)

            inputs_train, targets_train = self._parse_data(train_inputs)
            inputs_val, targets_val = self._parse_data(val_inputs)

            inputs_train = Variable(inputs_train).cuda()
            targets_train = Variable(targets_train).cuda(async=True)

            inputs_val = Variable(inputs_val).cuda()
            targets_val = Variable(targets_val).cuda(async=True)

            train_features, train_cls_out = self.model(inputs_train,T)
            val_features, val_cls_out = self.model(inputs_val,T)

            architect.step(train_features, targets_train, val_features,val_cls_out, targets_val, lr, optimizer, unrolled=False)

            # backward main #
            with torch.no_grad():
                kd_targets = sun_knowledge_ensemble(train_features.detach(), train_cls_out.detach(),targets_train.detach())
            loss_kd = kdloss(train_cls_out, kd_targets.detach())

            loss_ce, loss_tr,prec1 = self._forward(train_features, train_cls_out, targets_train)
            loss = loss_ce + loss_tr+loss_kd

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_kd.update(loss_kd.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Train:\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Loss_kd {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_kd.val, losses_kd.avg,
                              precisions.val, precisions.avg))

   # 'Loss_tr {:.3f} ({:.3f})\t' losses_tr.val, losses_tr.avg,
    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce,loss_tr, prec


class Validation(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Validation, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()

    def val(self, epoch, T, val_loader, val_iters=200, print_freq=1):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_kd = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(val_iters):
            source_inputs = val_loader.next()

            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)

            s_inputs = Variable(s_inputs).cuda()
            targets = Variable(targets).cuda()

            s_features, s_cls_out = self.model(s_inputs,T)

            # backward main #
            loss_ce,loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)

            with torch.no_grad():
                kd_targets = sun_knowledge_ensemble(s_features.detach(), s_cls_out.detach(),targets.detach())
            loss_kd = kdloss(s_cls_out, kd_targets.detach())
            loss = loss_ce  + loss_tr + loss_kd

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_kd.update(loss_kd.item())
            precisions.update(prec1)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Validation:\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_kd {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, val_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_kd.val, losses_kd.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr,prec


class Trainer(object):
    def __init__(self, model, num_classes, margin=0.0):
        super(Trainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.simCSE = SimpleCSE().cuda()


    def train(self, epoch, data_loader_source, optimizer, train_iters=200, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        losses_kd = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()

            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)

            s_features, s_cls_out = self.model(s_inputs)

            with torch.no_grad():
                kd_targets = sun_knowledge_ensemble(s_features.detach(), s_cls_out.detach(), targets.detach())
            loss_kd = kdloss(s_cls_out, kd_targets.detach())
            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce +loss_tr+loss_kd


            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            losses_kd.update(loss_kd.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Train:\t'
                      'Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Loss_kd {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              losses_kd.val, losses_kd.avg,
                              precisions.val, precisions.avg))
              #  'Loss_tr {:.3f} ({:.3f})\t'    losses_tr.val, losses_tr.avg,

    def _parse_data(self, inputs):
        imgs, _, pids, _ = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)

        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce,loss_tr, prec


