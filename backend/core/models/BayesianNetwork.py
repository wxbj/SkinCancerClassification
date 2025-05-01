import torch
from torch import nn
from backend.core.models.BayesianModule import MLP_Skin
from backend.core.models.edl import EDLClassifier
from backend.core.models.resnet_attn import ResNet50_Self_Attn

"""
最终皮肤癌识别模型：
    1、用带有50自注意力机制的50层残差网络进行皮损部位的特征提取。resnet_self_attn
    2、然后传入mlp_skin贝叶斯线性变换层，从而实现模型的不确定性估计。贝叶斯层可以用来对模型的参数进行后验分布的估计。
    3、最后通过EDLclassifier进行置信度区间的计算，输出皮肤癌的种类，置信度，及其置信度区间以及准确率。
"""

class CombinedBayesianNetwork(nn.Module):
    def __init__(self,dim_encoder_out=2048, num_classes=7, use_cuda=True,pretrained=True):
        super(CombinedBayesianNetwork, self).__init__()
        self.resnet_self_attn = ResNet50_Self_Attn(pretrained=pretrained, out_features=2048)
        self.mlp_skin = MLP_Skin(dim_encoder_out=dim_encoder_out,num_classes=num_classes, use_cuda=use_cuda)
        self.edl_classifier = EDLClassifier(encoder=self.mlp_skin, dim_encoder_out=dim_encoder_out, dim_hidden=50,
                                            num_classes=num_classes, dropout=0.5)

    def forward(self, x):
        x = self.resnet_self_attn(x)
        x = self.edl_classifier(x)

        return x

    @torch.inference_mode()
    def predict(self, x, return_uncertainty=True):
        x = self.resnet_self_attn(x)
        return self.edl_classifier.predict(x, return_uncertainty=return_uncertainty)