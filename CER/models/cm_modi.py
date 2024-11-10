from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
import numpy as np
from collections import defaultdict


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        # uni_target = np.unique(targets.cpu())
        # for target in uni_target:
        #     indexes = torch.nonzero(targets == target).squeeze(-1)
        #     target_mean_feature = inputs[indexes].mean(0)
        #     ctx.features[target] = ctx.momentum * ctx.features[target] + (1. - ctx.momentum) * target_mean_feature
        #     ctx.features[target] /= ctx.features[target].norm()

        for x, y in zip(inputs, targets):    #取个mean取update
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        # ctx.topk = topk
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            # grad_inputs = grad_outputs.mm(ctx.features.half())
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            # topk_index = np.argsort(np.array(distances))[0:ctx.topk]
            # topk_feature = features[topk_index].mean(0)
            # ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * topk_feature
            # ctx.features[index] /= ctx.features[index].norm()

            # topk_index = np.argsort(np.array(distances))[0:ctx.topk]
            # for idx in topk_index:
            #     ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * features[idx]
            #     ctx.features[index] /= ctx.features[index].norm()
        return grad_inputs, None, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.use_hard = use_hard
        self.neg_samp_topk = 50

        self.register_buffer('features', torch.zeros(num_samples, num_features))

    def forward(self, inputs, targets):
        inputs = F.normalize(inputs, dim=1).cuda()
        # if self.use_hard:
        outputs = cm_hard(inputs, targets, self.features, self.momentum)
        # uni_target = np.unique(targets.cpu())
        # temp_sims_cluster = outputs.detach().clone()
        # outputs /= 1.0
        # cluster_loss = 0.
        # for uni_index in uni_target:
        #     index = torch.nonzero(targets == uni_index).squeeze(-1)
        #     uni_index_min = np.argmin(temp_sims_cluster[index, uni_index].cpu())
        #     for idx in index:
        #         temp_sims_cluster[idx, uni_index] = -10000.0
        #
        #     concated_neg = torch.cat(
        #         [outputs[idx, torch.sort(temp_sims_cluster[idx])[1][-self.neg_samp_topk:]] for idx in
        #          index], dim=0)
        #     concated_input = torch.cat((outputs[index[uni_index_min], uni_index].unsqueeze(0), concated_neg),
        #                                dim=0)
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #         torch.device('cuda'))
        #     concated_target[0] = 1.0
        #     cluster_loss += -1 * (
        #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()

        outputs /= self.temp
        loss = F.cross_entropy(outputs, targets)
        return loss
