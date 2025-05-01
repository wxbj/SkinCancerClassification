from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from backend.core.models import BayesianLayers


class BayesianModule(nn.Module):
    def __init__(self):
        self.kl_list = []
        self.layers = []
        super(BayesianModule, self).__init__()

    def __setattr__(self, name, value):
        super(BayesianModule, self).__setattr__(name, value)
        # simple hack to collect bayesian layer automatically
        if isinstance(value, BayesianLayers.BayesianLayers) and not isinstance(value, BayesianLayers._ConvNdGroupNJ):
            self.kl_list.append(value)
            self.layers.append(value)

    def get_masks(self, thresholds):
        weight_masks = []
        mask = None
        for i, (layer, threshold) in enumerate(zip(self.kl_list, thresholds)):
            # compute dropout mask
            if mask is None:
                log_alpha = layer.get_log_dropout_rates().cpu().data.numpy()
                mask = log_alpha < threshold
            else:
                mask = np.copy(next_mask)

            try:
                log_alpha = self.layers[i + 1].get_log_dropout_rates().cpu().data.numpy()
                next_mask = log_alpha < thresholds[i + 1]
            except:
                # must be the last mask
                next_mask = np.ones(10)

            weight_mask = np.expand_dims(mask, axis=0) * np.expand_dims(next_mask, axis=1)
            weight_masks.append(weight_mask.astype(np.float))
        return weight_masks

    def kl_divergence(self):
        KLD = 0
        for layer in self.kl_list:
            KLD += layer.kl_divergence()
        return KLD




class MLP_Skin(BayesianModule):
    def __init__(self,dim_encoder_out, num_classes=7, use_cuda=True):
        super(MLP_Skin, self).__init__()

        self.fc1 = BayesianLayers.LinearGroupNJ(2048, 300, clip_var=0.04, cuda=use_cuda)
        self.fc2 = BayesianLayers.LinearGroupNJ(300, 100, cuda=use_cuda)
        self.fc3 = BayesianLayers.LinearGroupNJ(100, dim_encoder_out, cuda=use_cuda)

    def forward(self, x):
       # x = x.view(-1, 3 * 32 * 32)
        #print("进入mlp_skin里的第1层BayesianLayers.LinearGroupNJ前的数据流：",x.size())
        x = F.relu(self.fc1(x))
        #print("进入mlp_skin里的第2层BayesianLayers.LinearGroupNJ前的数据流：", x.size())
        x = F.relu(self.fc2(x))
        #print("进入mlp_skin里的第3层BayesianLayers.LinearGroupNJ前的数据流：", x.size())
        x = self.fc3(x)
        #print("经过mlp_skin里的第3层BayesianLayers.LinearGroupNJ后的数据流：", x.size())
        return x

if __name__ == "__main__":
    data = torch.randn([1, 3, 32, 32])
    data = Variable(data)

