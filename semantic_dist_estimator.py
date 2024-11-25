import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn

import torch.nn.functional as F


def bilinear_interpolate_1d(tensor, target_size):
    """
    对一个2D tensor进行双线性插值，仅改变第二个维度的大小。

    参数:
    tensor (torch.Tensor): 输入tensor，形状为 (H, W)
    target_size (int): 目标大小，即第二个维度的新大小

    返回:
    torch.Tensor: 插值后的tensor，形状为 (H, target_size)
    """
    # 获取原始tensor的形状
    original_shape = tensor.shape

    # 添加两个维度以满足interpolate函数的输入要求
    tensor_unsqueezed = tensor.unsqueeze(0).unsqueeze(0)  # 新形状: (1, 1, H, W)

    # 使用双线性插值调整大小
    resized_tensor = F.interpolate(tensor_unsqueezed,
                                   size=(original_shape[0], target_size),
                                   mode='bilinear',
                                   align_corners=False)

    # 移除之前添加的维度
    final_tensor = resized_tensor.squeeze(0).squeeze(0)  # 最终形状: (H, target_size)

    return final_tensor

class semantic_dist_estimator():
    def __init__(self, feature_num = 2048):
        super(semantic_dist_estimator, self).__init__()

        # self.cfg = cfg
        self.class_num = 13
        # _, backbone_name = cfg.MODEL.NAME.split('_')
        # self.feature_num = 2048 if backbone_name.startswith('resnet') else 1024
        self.feature_num = feature_num
        # init mean and covariance
        self.init(feature_num=self.feature_num)

    def init(self, feature_num,):

        self.CoVariance = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
        self.Mean = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True)
        self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)
        self.dist_vec  = torch.zeros(self.class_num, 512).cuda(non_blocking=True)

    def update_dist_vec(self, new_data):
        if not self.dist_vec.shape == new_data.shape:
            raise ValueError("Shape of new_data must match dist_vec!")
        alpha = 0.9
        self.dist_vec = alpha * new_data + (1 - alpha) * self.dist_vec

    def get_dist_vec(self):
        return self.dist_vec

    def get(self,target_size = 512):
        # Mean = F.normalize(self.Mean, p=2, dim=1)
        # CoVariance = F.normalize(self.CoVariance, p=2, dim=1)
        # fea_unsqueezed = fea.unsqueeze(0).unsqueeze(0)  # 新维度: 1x1x13x2048
        #
        # # 使用双线性插值调整大小
        # resized_fea = F.interpolate(fea_unsqueezed, size=(13, 512), mode='bilinear', align_corners=False)
        #
        # # 移除之前添加的维度
        # final_fea = resized_fea.squeeze(0).squeeze(0)  # 最终维度: 13x512
        Mean = bilinear_interpolate_1d(self.Mean,target_size)
        CoVariance = bilinear_interpolate_1d(self.CoVariance,target_size)
        return Mean,CoVariance
    def update(self, features, labels):

        # label_mask = (labels == self.cfg.INPUT.IGNORE_LABEL).long()
        # labels = ((1 - label_mask).mul(labels) + label_mask * self.cfg.MODEL.NUM_CLASSES).long()

        # mask = (labels != self.cfg.INPUT.IGNORE_LABEL)
        # # remove IGNORE_LABEL pixels
        # labels = labels[mask]
        # features = features[mask]




        N, A = features.size()
        C = self.class_num

        NxCxA_Features = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )

        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxA_Features.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        mean_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - mean_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Mean - mean_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp.mul(
            weight_CV)).detach() + additional_CV.detach()

        self.Mean = (self.Mean.mul(1 - weight_CV) + mean_CxA.mul(weight_CV)).detach()

        self.Amount = self.Amount + onehot.sum(0)

    def save(self, name):
        torch.save({'CoVariance': self.CoVariance.cpu(),
                    'Mean': self.Mean.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.cfg.OUTPUT_DIR, name))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
