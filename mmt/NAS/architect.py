import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from mmt.loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy


class KDLoss(nn.Module):
  def __init__(self, temp_factor):
    super(KDLoss, self).__init__()
    self.temp_factor = temp_factor
    self.kl_div = nn.KLDivLoss(reduction="sum")

  def forward(self, input, target):
    log_p = torch.log_softmax(input / self.temp_factor, dim=1)
    loss = self.kl_div(log_p, target) * (self.temp_factor ** 2) / input.size(0)
    return loss


temp = 8
omega = 0.5
kdloss = KDLoss(temp)
softmax = nn.Softmax(dim=1)


def sun_knowledge_ensemble(feats, logits, targets):
  batch_size = logits.size(0)
  id_num = len(targets.unique())
  instance = int(batch_size/id_num)

  masks = torch.eye(instance)
  masks = masks.cuda()

  feats11 = feats.chunk(id_num, 0)
  logits11 = logits.chunk(id_num, 0)
  Q1 = []
  for i in range(id_num):
    feats22 = nn.functional.normalize(feats11[i], p=2, dim=1)
    logits22 = nn.functional.softmax(logits11[i] / temp, dim=1)
    W = torch.matmul(feats22, feats22.permute(1, 0)) - masks * 1e9
    W = softmax(W)
    W = (1 - omega) * torch.inverse(masks - omega * W)
    Q = torch.matmul(W, logits22)
    Q1.append(Q)
  Q1 = torch.cat(Q1)
  return Q1

def Loss_kd(train_features, train_cls_out,targets_train):
  with torch.no_grad():
    kd_targets = sun_knowledge_ensemble(train_features.detach(), train_cls_out.detach(), targets_train.detach())
  loss_kd = kdloss(train_cls_out, kd_targets.detach())
  return loss_kd

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):
#alpha tidu
  def __init__(self, model,optimizer,num_classes, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
         lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    #self.optimizer = optimizer
    self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
    self.criterion_triple = SoftTripletLoss(margin=args.margin).cuda()


  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss_ce = self.criterion_ce(input, target)
    loss_tri = self.criterion_triple(input,input, target)
    loss = loss_ce + loss_tri

    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)

    a = torch.autograd.grad(loss, self.model.parameters())
    dtheta = _concat(a).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid,val_cls_out, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid,val_cls_out, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid,val_cls_out, target_valid):
    loss_ce = self.criterion_ce(input_valid, target_valid)
    loss_tri = self.criterion_triple(input_valid, input_valid,target_valid)
    loss_kd = Loss_kd(input_valid,val_cls_out,target_valid )
    loss = loss_ce +loss_tri+loss_kd
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    loss_ce = self.criterion_ce(input, target)
    loss_tri = self.criterion_triple(input, input, target)
    loss = loss_ce + loss_tri

    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input,input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

