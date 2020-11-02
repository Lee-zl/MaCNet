import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def append_params(params, module, prefix):
    for child in module.children():
        for k,p in child._parameters.items():
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError('Duplicated param name: {:s}'.format(name))

def append_params2(params, module, prefix):
    for child in module.children():
        for son in child.children():
            for k,p in son._parameters.items():
                if p is None: continue

                if isinstance(son, nn.BatchNorm2d):
                    name = prefix + '_bn_' + k
                else:
                    name = prefix + '_' + k

                if name not in params:
                    params[name] = p
                else:
                    raise RuntimeError('Duplicated param name: {:s}'.format(name))


def set_optimizer_F(model, lr_base, lr_mult, momentum=0.9, w_decay=0.0005):
    params = model.get_learnable_params()
    param_list = []
    lr = lr_base
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                if k.startswith('Ff') :
                    lr = lr_base * m
                else:
                    lr = 0
        param_list.append({'params': [p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
    return optimizer


class MALayer(nn.Module):
    def  __init__(self, num_conv=6, channel=6, k=1):
        super(MALayer, self).__init__()
        self.k = k
        self.num_conv = num_conv
        self.output = k * num_conv
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc1 = nn.Sequential(OrderedDict([
            ('ca_fc1', nn.Sequential(nn.Linear(channel, self.output*2),
                                     nn.ReLU()))]))
        self.ca_fc2 = nn.Sequential(OrderedDict([
            ('ca_fc2',nn.Sequential(nn.Linear(self.output*2, self.k * self.num_conv)))]))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc1(y)
        y = self.ca_fc2(y)
        y = y.view(-1, self.k, self.num_conv)
        return y



class MDNet(nn.Module):
    def __init__(self, model_path1=None, K=1, C=6):
        super(MDNet, self).__init__()
        self.K = K
        self._C = C


        #****************************RGB_branch***********************************#
        self.RGB_layers_conv1 = nn.Sequential(OrderedDict([
                ('Rconv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        ))]))

        self.RGB_layers_conv2 = nn.Sequential(OrderedDict([
                ('Rconv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        ))]))

        self.RGB_layers_conv3 = nn.Sequential(OrderedDict([
                ('Rconv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True)
                                        ))]))


        #*****************************T_branch************************************#
        self.T_layers_conv1 = nn.Sequential(OrderedDict([
                ('Tconv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        ))]))
        self.T_layers_conv2 = nn.Sequential(OrderedDict([
                ('Tconv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2)
                                        ))]))

        self.T_layers_conv3 = nn.Sequential(OrderedDict([
                ('Tconv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True)
                                        ))]))


        #*****************************Fusion_branch************************************#
        self.F_layers = nn.Sequential(OrderedDict([
                ('Ffc4',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(1024 * 3 * 3, 512),
                                        nn.ReLU(inplace=True))),
                ('Ffc5',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(512, 512),
                                        nn.ReLU(inplace=True)))]))


        #*****************************Bi-classification************************************#
        self.branches_F = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                       nn.Linear(512, 2)) for _ in range(K)])

        # *****************************MALayer************************************#
        self.attention = MALayer()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(OrderedDict([('conv1',nn.Sequential(nn.Conv2d(192, 96, 1, padding=0, bias=False),
                                                                       nn.ReLU()))]))
        self.conv2 = nn.Sequential(OrderedDict([('conv2',nn.Sequential(nn.Conv2d(512, 256, 1, padding=0, bias=False),
                                                                       nn.ReLU()))]))
        self.conv3 = nn.Sequential(OrderedDict([('conv3',nn.Sequential(nn.Conv2d(1024, 512, 1, padding=0, bias=False),
                                                                       nn.ReLU()))]))

        # ******************************load model**********************************#
        if model_path1 is not None:
            if os.path.splitext(model_path1)[1] == '.pth':
                self.load_model(model_path1)
            elif os.path.splitext(model_path1)[1] == '.mat':
                self.load_mat_model(model_path1)
            else:
                raise RuntimeError('Unkown model format: {:s}'.format(model_path1))
        self.build_param_dict()


    def build_param_dict(self):
        self.params = OrderedDict()

        for name, module in self.attention.named_children():
            append_params2(self.params, module, name)

        for name, module in self.conv1.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv2.named_children():
            append_params(self.params, module, name)
        for name, module in self.conv3.named_children():
            append_params(self.params, module, name)
        # ****************************RGB_branch***********************************#
        for name, module in self.RGB_layers_conv1.named_children():
            append_params(self.params, module, name)
        for name, module in self.RGB_layers_conv2.named_children():
            append_params(self.params, module, name)
        for name, module in self.RGB_layers_conv3.named_children():
            append_params(self.params, module, name)


        # *****************************T_branch************************************#
        for name, module in self.T_layers_conv1.named_children():
            append_params(self.params, module, name)
        for name, module in self.T_layers_conv2.named_children():
            append_params(self.params, module, name)
        for name, module in self.T_layers_conv3.named_children():
            append_params(self.params, module, name)

        # *****************************Fusion_branch************************************#
        for name, module in self.F_layers.named_children():
            append_params(self.params, module, name)


        # *****************************Bi-classification************************************#
        for k, module in enumerate(self.branches_F):
            append_params(self.params, module, 'Ffc6_{:d}'.format(k))

    def set_learnable_params(self, layers):
        for k, p in self.params.items():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.params.items():
            if p.requires_grad:
                params[k] = p
        return params


    def forward(self, xR=None, xT=None ,k=0, in_layer='Rconv1', out_layer='Ffc6',feat=None):
        # forward model from in_layer to out_layer

        # ****************************conv1***********************************#
        if in_layer == 'Rconv1':
            feat_att = torch.cat((xR, xT), dim=1)
            weights = self.attention(feat_att)
            weights = F.softmax(weights, dim=-1)

            feat_RGB_1 = self.RGB_layers_conv1(xR)
            feat_RGB_w1 = feat_RGB_1 * weights[:, :, 0].view([-1, 1, 1, 1])

            feat_T_1 = self.T_layers_conv1(xT)
            feat_T_w1 = feat_T_1 * weights[:, :, 1].view([-1, 1, 1, 1])

            feat_fuss_1 = torch.cat((feat_RGB_w1, feat_T_w1), dim=1)

            feat_RGB_1 = self.relu(self.conv1(feat_fuss_1) + feat_RGB_1)
            feat_T_1 = self.relu(self.conv1(feat_fuss_1) + feat_T_1)


        # ****************************conv2***********************************#
            feat_RGB_2 = self.RGB_layers_conv2(feat_RGB_1)
            feat_RGB_W2 = feat_RGB_2 * weights[:, :, 2].view([-1, 1, 1, 1])

            feat_T_2 = self.T_layers_conv2(feat_T_1)
            feat_T_w2 = feat_T_2 * weights[:, :, 3].view([-1, 1, 1, 1])

            feat_fuss_2 = torch.cat((feat_RGB_W2, feat_T_w2), dim=1)

            feat_RGB_2 = self.relu(self.conv2(feat_fuss_2) + feat_RGB_2)
            feat_T_2 = self.relu(self.conv2(feat_fuss_2) + feat_T_2)

        # ****************************conv3***********************************#
            feat_RGB_3 = self.RGB_layers_conv3(feat_RGB_2)
            feat_RGB_w3 = feat_RGB_3 * weights[:, :, 4].view([-1, 1, 1, 1])

            feat_T_3 = self.T_layers_conv3(feat_T_2)
            feat_T_w3 = feat_T_3 * weights[:, :, 5].view([-1, 1, 1, 1])

            feat_fuss_3 = torch.cat((feat_RGB_w3, feat_T_w3), dim=1)
            feat_RGB_3 = self.relu(self.conv3(feat_fuss_3) + feat_RGB_3)
            feat_T_3 = self.relu(self.conv3(feat_fuss_3) + feat_T_3)

            feat_F_3 = torch.cat((feat_RGB_3, feat_T_3), 1)
            feat_F_3 = feat_F_3.view(feat_F_3.size(0), -1)

            feat_F_fc = self.F_layers(feat_F_3)

        # ****************************fc***********************************#
        if in_layer == 'Ffc4':
            feat_F_fc = self.F_layers(feat)

        feat_F_end = self.branches_F[k](feat_F_fc)


        if out_layer=='Ffc6':
            return feat_F_end
        elif out_layer=='Ffc6_softmax':
            F.softmax(feat_F_end, dim=1)
        elif out_layer =='Fconv3':
            return feat_F_3


    def load_model(self, model_path):
        states = torch.load(model_path)

        conv1_layers = states['RGB_layers_conv1']
        self.RGB_layers_conv1.load_state_dict(conv1_layers,  strict=True)

        conv2_layers = states['RGB_layers_conv2']
        self.RGB_layers_conv2.load_state_dict(conv2_layers, strict=True)

        conv3_layers = states['RGB_layers_conv3']
        self.RGB_layers_conv3.load_state_dict(conv3_layers, strict=True)


        conv1_layers = states['T_layers_conv1']
        self.T_layers_conv1.load_state_dict(conv1_layers, strict=True)

        conv2_layers = states['T_layers_conv2']
        self.T_layers_conv2.load_state_dict(conv2_layers, strict=True)

        conv3_layers = states['T_layers_conv3']
        self.T_layers_conv3.load_state_dict(conv3_layers, strict=True)


        fc_layers = states['F_layers']
        self.F_layers.load_state_dict(fc_layers, strict=True)


        ca_fc1 = states['ca_fc1']
        self.attention.ca_fc1.load_state_dict(ca_fc1, strict=True)
        ca_fc2 = states['ca_fc2']
        self.attention.ca_fc2.load_state_dict(ca_fc2, strict=True)

        conv1 = states['conv1']
        self.conv1.load_state_dict(conv1, strict=True)
        conv2 = states['conv2']
        self.conv2.load_state_dict(conv2, strict=True)
        conv3 = states['conv3']
        self.conv3.load_state_dict(conv3, strict=True)



        print('load finish pth!')


    def load_mat_model(self, matfile):
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        weight, bias = mat_layers[0 * 4]['weights'].item()[0]
        self.RGB_layers_conv1[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.RGB_layers_conv1[0][0].bias.data = torch.from_numpy(bias[:, 0])
        self.T_layers_conv1[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.T_layers_conv1[0][0].bias.data = torch.from_numpy(bias[:, 0])

        weight, bias = mat_layers[1 * 4]['weights'].item()[0]
        self.RGB_layers_conv2[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.RGB_layers_conv2[0][0].bias.data = torch.from_numpy(bias[:, 0])
        self.T_layers_conv2[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.T_layers_conv2[0][0].bias.data = torch.from_numpy(bias[:, 0])

        weight, bias = mat_layers[2 * 4]['weights'].item()[0]
        self.RGB_layers_conv3[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.RGB_layers_conv3[0][0].bias.data = torch.from_numpy(bias[:, 0])
        self.T_layers_conv3[0][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
        self.T_layers_conv3[0][0].bias.data = torch.from_numpy(bias[:, 0])

        print('load mat finish!')



class BCELoss(nn.Module):
    def forward(self, pos_score, neg_score, average=True):
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]

        loss = pos_loss.sum() + neg_loss.sum()
        if average:
            loss /= (pos_loss.size(0) + neg_loss.size(0))
        return loss


class Accuracy():
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float()
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float()
        acc = (pos_correct + neg_correct) / (pos_score.size(0) + neg_score.size(0) + 1e-8)
        return acc.item()


class Precision():
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)
        topk = torch.topk(scores, pos_score.size(0))[1]
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)
        return prec.item()
