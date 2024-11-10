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
        # ctx.save_for_backward(inputs, targets)
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

        for x, y in zip(inputs, targets):  # 取个mean取update
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


class CM_Hard(autograd.Function):

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

        return grad_inputs, None, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard_proxy(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, camids, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)

        proxy_centroid = []
        proxy_target = []
        for label in torch.unique(targets):
            for cam in torch.unique(camids):
                inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
                        val in torch.nonzero(camids == cam).squeeze(-1)]
                if len(inds) != 0:
                    inds = torch.tensor(inds)
                    proxy = inputs[inds].mean(0).unsqueeze(0)  # 不错
                    proxy_centroid.append(proxy)
                    proxy_target.append(label.unsqueeze(0))
        proxy_centroid = torch.cat(proxy_centroid, dim=0).cuda()
        proxy_target = torch.cat(proxy_target, dim=0).cuda()
        outputs = proxy_centroid.mm(ctx.features.t())
        return outputs, proxy_target

    @staticmethod
    def backward(ctx, grad_outputs, grad_target):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)
        #
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

        # for x, y in zip(inputs, targets):  # 取个mean取update
        #     ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
        #     ctx.features[y] /= ctx.features[y].norm()
        # return grad_inputs, None, None, None, None


def cm_hard_proxy(inputs, indexes, camids, features, momentum=0.5):
    return CM_Hard_proxy.apply(inputs, indexes, camids, features, torch.Tensor([momentum]).to(inputs.device))


class ClusterMemory(nn.Module, ABC):
    def __init__(self, args, num_features, num_samples, temp=0.05, momentum=0.2, use_hard=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.beta = args.beta
        self.use_hard = use_hard
        self.neg_samp_topk = 50
        self.beta = 1.0
        self.neg_samp_cluster_topk = args.neg_samp_cluster_topk
        self.kl = nn.KLDivLoss(reduction='batchmean')
        self.register_buffer('features', torch.zeros(num_samples, num_features))
        self.lam = args.lam

    def forward(self, inputs, targets, camids, isClusterC):
        inputs = F.normalize(inputs, dim=1).cuda()
        neg_num = len(self.features) - 1
        proxy_cluster_loss = 0.
        outputs = cm_hard(inputs, targets, self.features, self.momentum)

        # outputs, proxy_target = cm_hard_proxy(inputs, targets, camids, self.features, self.momentum)

        # sim = outputs.detach().clone()
        outputs /= 0.05

        # loss = F.cross_entropy(outputs, targets)

        proxy_centroid = []
        proxy_target = []
        for label in torch.unique(targets):
            for cam in torch.unique(camids):
                inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
                        val in torch.nonzero(camids == cam).squeeze(-1)]
                if len(inds) != 0:
                    inds = torch.tensor(inds)
                    proxy = outputs[inds].mean(0).unsqueeze(0)  # 不错
                    proxy_centroid.append(proxy)
                    proxy_target.append(label.unsqueeze(0))
        proxy_centroid = torch.cat(proxy_centroid, dim=0).cuda()
        proxy_target = torch.cat(proxy_target, dim=0).cuda()
        # loss = F.cross_entropy(proxy_centroid, proxy_target)
        proxy_sim = proxy_centroid.detach().clone()

        len_feature = proxy_sim.size()[1] - 1
        for label in torch.unique(proxy_target):
            inds = torch.nonzero(proxy_target == label).squeeze(-1)
            uni_index_min = np.argmin(proxy_sim[inds, label].cpu())
            concated_pos = proxy_centroid[inds[uni_index_min], label].unsqueeze(0)
            concated_neg = torch.cat(
                [proxy_centroid[idx, torch.sort(proxy_sim[idx])[1][-len_feature:]]
                 for idx in inds], dim=0)

            concated_input = torch.cat(
                (concated_pos, concated_neg),
                dim=0)

            concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                torch.device('cuda'))
            concated_target[0] = 1.0
            proxy_cluster_loss += -1 * (
                    F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        loss = proxy_cluster_loss / len(torch.unique(proxy_target))




        # loss = torch.tensor([0.]).cuda()
        # unique_label = torch.unique(targets)
        # for label in unique_label:
        #     proxy_sim = sim.detach().clone()
        #     inds = torch.nonzero(targets == label).squeeze(-1)
        #     uni_index_min = np.argmin(proxy_sim[inds, label].cpu())
        #     proxy_sim[inds, label] = -10000.
        #     concated_pos = outputs[inds[uni_index_min], label].unsqueeze(0)
        #     concated_neg = torch.cat(
        #         [outputs[idx, torch.sort(proxy_sim[idx])[1][-15:]]
        #          for idx in inds], dim=0)
        #
        #     concated_input = torch.cat(
        #         (concated_pos, concated_neg),
        #         dim=0)
        #
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #         torch.device('cuda'))
        #     concated_target[0] = 1.0
        #     proxy_cluster_loss += -1 * (
        #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss = proxy_cluster_loss / len(unique_label)
        #
        # proxy_cluster_loss = 0.
        # proxy_cout = 0
        #
        # length_features = len(self.features) - 1
        #
        # for label in torch.unique(targets):
        #     for cam in torch.unique(camids):
        #         inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #                 val in torch.nonzero(camids == cam).squeeze(-1)]
        #         if len(inds) == 0:
        #             continue
        #         else:
        #             proxy_cout += 1
        #             inds = torch.tensor(inds)
        #
        #             # uni_index_min = np.argmin(sim[inds, label].cpu())
        #             # proxy_pos = outputs[inds[uni_index_min], label].unsqueeze(0)
        #
        #             proxy_pos = outputs[inds, label].mean(0).unsqueeze(0)  # 不错
        #             sim[inds, label] = -10000.0
        #
        #             # length_instance = int(length_features * 0.5)
        #
        #             length_instance = 50
        #
        #             proxy_neg = torch.cat(
        #                 [outputs[
        #                      idx, torch.sort(sim[idx])[1][-length_instance:]] for idx in inds], dim=0)
        #
        #             concated_input = torch.cat(
        #                 (proxy_pos, proxy_neg),
        #                 dim=0)
        #
        #             concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #                 torch.device('cuda'))
        #             concated_target[0] = 1.0
        #             proxy_cluster_loss += -1 * (
        #                     F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #             # proxy_cout += 1
        #             # inds = torch.tensor(inds)
        #             #
        #             # min = np.argmin(sim[inds, label].cpu().numpy())
        #             # proxy_pos = outputs[inds[min], label].unsqueeze(0)
        #             # # proxy_pos = outputs[inds, label].mean(0).unsqueeze(0)  # 不错
        #             # # proxy_pos = outputs[inds, label]
        #             # sim[inds[min], label] = -10000.0
        #             # # proxy_neg = outputs[inds, :].mean(0)[torch.sort(sim[inds, :].mean(0))[1][-10:]]
        #             # # self.neg_samp_cluster_topk
        #             # # proxy_neg = torch.cat(
        #             # #     [outputs[
        #             # #          idx, torch.sort(sim[idx])[1][-self.neg_samp_cluster_topk:]] for idx in inds],
        #             # #     dim=0)
        #             #
        #             # proxy_neg = outputs[inds[min], torch.sort(sim[inds[min]])[1][-neg_num:]]
        #             #
        #             # concated_input = torch.cat(
        #             #     (proxy_pos, proxy_neg),
        #             #     dim=0)
        #             # concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #             #     torch.device('cuda'))
        #             # concated_target[0] = 1.0
        #             # # concated_target[0: len(inds)] = 1.0 / len(inds)
        #             # proxy_cluster_loss += -1 * (
        #             #         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss = proxy_cluster_loss / proxy_cout

        return loss
        # if isClusterC:
        #     # outputs = inputs.mm(self.features.t())
        #     sim = outputs.detach().clone()
        #     outputs /= self.temp
        #     # # outputs /= self.beta
        #     proxy_cluster_loss = 0.
        #     proxy_cout = 0
        #     for label in torch.unique(targets):
        #         for cam in torch.unique(camids):
        #             inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #                     val in torch.nonzero(camids == cam).squeeze(-1)]
        #             if len(inds) == 0:
        #                 continue
        #             else:
        #                 proxy_cout += 1
        #                 # proxy_sim = sim.detach().clone()
        #                 inds = torch.tensor(inds)
        #                 # proxy_pos = outputs[inds[np.argmin(proxy_sim[inds, label].cpu().numpy())], label].unsqueeze(0)
        #                 proxy_pos = outputs[inds, label].mean(0).unsqueeze(0)  # 不错
        #                 # proxy_pos = outputs[inds, label]
        #                 sim[inds, label] = -10000.0
        #                 # proxy_neg = outputs[inds, :].mean(0)[torch.sort(sim[inds, :].mean(0))[1][-10:]]
        #
        #                 # proxy_neg = torch.cat(
        #                 #     [outputs[
        #                 #          idx, torch.sort(proxy_sim[idx])[1][-20:]] for idx in inds], dim=0)
        #
        #                 proxy_neg = torch.cat(
        #                             [outputs[
        #                             idx, torch.sort(sim[idx])[1][-neg_num:]] for idx in inds], dim=0)
        #
        #                 concated_final_neg = proxy_neg[torch.sort(proxy_neg)[1][-neg_num:]]
        #
        #                 concated_input = torch.cat(
        #                     (proxy_pos, concated_final_neg),
        #                     dim=0)
        #
        #                 concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #                     torch.device('cuda'))
        #                 concated_target[0] = 1.0
        #                 proxy_cluster_loss += -1 * (
        #                         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #                 # proxy_cout += 1
        #                 # inds = torch.tensor(inds)
        #                 #
        #                 # min = np.argmin(sim[inds, label].cpu().numpy())
        #                 # proxy_pos = outputs[inds[min], label].unsqueeze(0)
        #                 # # proxy_pos = outputs[inds, label].mean(0).unsqueeze(0)  # 不错
        #                 # # proxy_pos = outputs[inds, label]
        #                 # sim[inds[min], label] = -10000.0
        #                 # # proxy_neg = outputs[inds, :].mean(0)[torch.sort(sim[inds, :].mean(0))[1][-10:]]
        #                 # # self.neg_samp_cluster_topk
        #                 # # proxy_neg = torch.cat(
        #                 # #     [outputs[
        #                 # #          idx, torch.sort(sim[idx])[1][-self.neg_samp_cluster_topk:]] for idx in inds],
        #                 # #     dim=0)
        #                 #
        #                 # proxy_neg = outputs[inds[min], torch.sort(sim[inds[min]])[1][-neg_num:]]
        #                 #
        #                 # concated_input = torch.cat(
        #                 #     (proxy_pos, proxy_neg),
        #                 #     dim=0)
        #                 # concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #                 #     torch.device('cuda'))
        #                 # concated_target[0] = 1.0
        #                 # # concated_target[0: len(inds)] = 1.0 / len(inds)
        #                 # proxy_cluster_loss += -1 * (
        #                 #         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #     loss = proxy_cluster_loss / proxy_cout
        #
        #     #############################
        #     # neg_num = len(self.features) - 1
        #     # for label in torch.unique(targets):
        #     #     proxy_pos = []
        #     #     hardest_indexs = []
        #     #     for cam in torch.unique(camids):
        #     #         inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #     #                 val in torch.nonzero(camids == cam).squeeze(-1)]
        #     #         if len(inds) == 0:
        #     #             continue
        #     #         else:
        #     #             # proxy_cout += 1
        #     #             inds = torch.tensor(inds)
        #     #             # per_proxy_index = np.argmin(sim[inds, label].cpu().numpy())
        #     #             # # hardest_indexs.append(per_proxy_index)
        #     #             # proxy_pos.append(outputs[inds[per_proxy_index],].unsqueeze(0))
        #     #             # proxy_mean = outputs[inds, label].mean(0).unsqueeze(0)  # 不错
        #     #             proxy_pos.append(outputs[inds,].mean(0).unsqueeze(0))
        #     #             # sim[per_proxy_index, label] = -10000.0
        #     #     proxy_pos_tensor = torch.cat(proxy_pos)
        #     #     proxy_sim = proxy_pos_tensor.detach().clone()
        #     #     proxy_pos_tensor /= self.temp
        #     #     pos = torch.cat(
        #     #         [proxy_pos_tensor[
        #     #              idx, label].unsqueeze(0) for idx, _ in enumerate(proxy_pos_tensor)],
        #     #         dim=0)
        #     #
        #     #     for idx, _ in enumerate(proxy_pos_tensor):
        #     #         proxy_sim[idx, label] = -10000
        #     #
        #     #     neg = torch.cat(
        #     #         [proxy_pos_tensor[
        #     #              idx, torch.sort(proxy_sim[idx])[1][-neg_num:]] for idx, _ in
        #     #          enumerate(proxy_pos_tensor)],
        #     #         dim=0)
        #     #
        #     #     concated_final_neg = neg[torch.sort(neg)[1][-50:]]
        #     #     # proxy_neg = torch.cat(
        #     #     #                     [outputs[
        #     #     #                          idx, torch.sort(sim[idx])[1][-10:]] for idx in hardest_indexs],
        #     #     #                     dim=0)
        #     #     concated_input = torch.cat(
        #     #         (pos, concated_final_neg),
        #     #         dim=0)
        #     #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #     #         torch.device('cuda'))
        #     #     # concated_target[0] = 1.0
        #     #     concated_target[0: len(proxy_pos)] = 1.0 / len(proxy_pos)
        #     #     proxy_cluster_loss += -1 * (
        #     #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #     # loss = proxy_cluster_loss / len(torch.unique(targets))
        #     #############################
        #     # outputs /= self.temp
        #     # #
        #     # # # proxy_pos = []
        #     # length = outputs.size()[1] - 1
        #     # for label in torch.unique(targets):
        #     #     # for cam in torch.unique(camids):
        #     #     #     inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #     #     #             val in torch.nonzero(camids == cam).squeeze(-1)]
        #     #     #     if len(inds) == 0:
        #     #     #         continue
        #     #     #     else:
        #     #     #         proxy_cout += 1
        #     #     #         inds = torch.tensor(inds)
        #     #     #         # proxy_pos = outputs[inds[np.argmin(sim[inds, label].cpu().numpy())], label].unsqueeze(0)
        #     #     #         proxy_mean = outputs[inds, label].mean(0).unsqueeze(0)  # 不错
        #     #     #         proxy_pos.append(proxy_mean.unsqueeze(0))
        #     #     # proxy_pos = torch.cat(proxy_pos)
        #     #     inds = [val for val in torch.nonzero(targets == label).squeeze(-1)]
        #     #     # cluster_pos = outputs[inds, label]
        #     #     # cluster_pos = outputs[inds, label].mean(0).unsqueeze(0)
        #     #     cluster_pos = outputs[inds[np.argmin(sim[inds, label].cpu().numpy())], label].unsqueeze(0)
        #     #     sim[inds, label] = -10000.0
        #     #
        #     #     # if epoch < 10:
        #     #     #     cluster_neg = torch.cat(
        #     #     #         [outputs[
        #     #     #              idx, torch.sort(sim[idx])[1][-30:-20]] for idx in inds],
        #     #     #         dim=0)
        #     #     # elif 10 <= epoch < 20:
        #     #     #     cluster_neg = torch.cat(
        #     #     #         [outputs[
        #     #     #              idx, torch.sort(sim[idx])[1][-20:-10]] for idx in inds],
        #     #     #         dim=0)
        #     #     # else:
        #     #     cluster_neg = torch.cat(
        #     #         [outputs[
        #     #              idx, torch.sort(sim[idx])[1][-20:]] for idx in inds],
        #     #         dim=0)
        #     #
        #     #     concated_input = torch.cat(
        #     #         (cluster_pos, cluster_neg),
        #     #         dim=0)
        #     #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #     #         torch.device('cuda'))
        #     #     # concated_target[0: proxy_cout] = 1.0 / proxy_cout
        #     #     concated_target[0] = 1.0
        #     #     # concated_target[0: len(inds)] = 1.0 / len(inds)
        #     #     proxy_cluster_loss += -1 * (
        #     #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #     # loss = proxy_cluster_loss / len(torch.unique(targets))
        #     ####################################
        #     # for label in torch.unique(targets):
        #     #     for cam in torch.unique(camids):
        #     #         inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #     #                 val in torch.nonzero(camids == cam).squeeze(-1)]
        #     #         if len(inds) == 0:
        #     #             continue
        #     #         else:
        #     #             proxy_cout += 1
        #     #             # proxy_sim = sim.detach().clone()
        #     #             inds = torch.tensor(inds)
        #     #
        #     #             # sim[inds, label] = -10000
        #     #             # proxy_neg = torch.cat(
        #     #             #     [outputs[
        #     #             #          idx, torch.sort(sim[idx])[1][-neg_num:]] for idx in inds], dim=0)
        #     #             # concated_final_neg = proxy_neg[torch.sort(proxy_neg)[1][-neg_num:]]
        #     #
        #     #             for index in inds:
        #     #                 proxy_pos = outputs[index, label].unsqueeze(0)
        #     #
        #     #                 sim[index, label] = -10000
        #     #                 proxy_neg = outputs[
        #     #                         index, torch.sort(sim[index])[1][-neg_num:]]
        #     #                 concated_input = torch.cat(
        #     #                     (proxy_pos, proxy_neg),
        #     #                     dim=0)
        #     #
        #     #                 # concated_input = torch.cat(
        #     #                 #     (proxy_pos, concated_final_neg),
        #     #                 #     dim=0)
        #     #
        #     #                 concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #     #                     torch.device('cuda'))
        #     #                 concated_target[0] = 1.0
        #     #                 proxy_cluster_loss += -1 * (
        #     #                         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
        #     #                     0)).sum()
        #     #
        #     #
        #     #             # proxy_pos = outputs[inds, label]
        #     #             # sim[inds, label] = -10000.0
        #     #             # # proxy_neg = outputs[inds, :].mean(0)[torch.sort(sim[inds, :].mean(0))[1][-10:]]
        #     #             #
        #     #             # proxy_neg = torch.cat(
        #     #             #     [outputs[
        #     #             #          idx, torch.sort(sim[idx])[1][-neg_num:]] for idx in inds], dim=0)
        #     #             #
        #     #             # concated_final_neg = proxy_neg[torch.sort(proxy_neg)[1][-neg_num:]]
        #     #             #
        #     #             # concated_input = torch.cat(
        #     #             #     (proxy_pos, concated_final_neg),
        #     #             #     dim=0)
        #     #             #
        #     #             # concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #     #             #     torch.device('cuda'))
        #     #             # # concated_target[0] = 1.0
        #     #             # concated_target[0:len(inds)] = 1.0 / len(inds)
        #     #             # proxy_cluster_loss += -1 * (
        #     #             #         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #     # loss = proxy_cluster_loss / len(targets)
        #     ###########################
        #
        #     # outputs /= 0.05
        #     # loss = F.cross_entropy(outputs, targets)
        # else:
        # outputs /= self.temp
        # output_cluster = []
        # for label in torch.unique(targets):
        #     inds = [val for val in torch.nonzero(targets == label).squeeze(-1)]
        #     cluster_pos = outputs[inds, ].mean(0)
        #     output_cluster.append(cluster_pos.unsqueeze(0))
        # output_cluster = torch.cat(output_cluster)
        # loss = F.cross_entropy(output_cluster, torch.unique(targets))

        # sim = outputs.detach().clone()
        # outputs /= self.temp
        # output_proxy = []
        # ce_target = []
        # for label in torch.unique(targets):
        #     for cam in torch.unique(camids):
        #         inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #                 val in torch.nonzero(camids == cam).squeeze(-1)]
        #         if len(inds) == 0:
        #             continue
        #         else:
        #             # proxy_pos = outputs[inds[np.argmin(sim[inds, label].cpu().numpy())],]
        #             proxy_pos = outputs[inds, ].mean(0)
        #             output_proxy.append(proxy_pos.unsqueeze(0))
        #             ce_target.append(label.unsqueeze(0))
        # output_proxy = torch.cat(output_proxy)
        # ce_target = torch.cat(ce_target)
        # loss = F.cross_entropy(output_proxy, ce_target)

        # proxy_cluster_loss = 0.
        # sim = outputs.detach().clone()
        # outputs /= 0.07
        # length = outputs.size()[1] - 1
        #
        # proxy_cout = 0
        # for label in torch.unique(targets):
        #     for cam in torch.unique(camids):
        #         inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #                 val in torch.nonzero(camids == cam).squeeze(-1)]
        #         if len(inds) == 0:
        #             continue
        #         else:
        #             proxy_cout += 1
        #             # proxy_sim = sim.detach().clone()
        #             inds = torch.tensor(inds)
        #
        #             proxy_mean = outputs[inds,].mean(0)
        #             sim[inds, label] = -10000
        #             proxy_sim = sim[inds,].mean(0)
        #             proxy_pos = proxy_mean[label].unsqueeze(0)
        #             proxy_neg = proxy_mean[torch.sort(proxy_sim)[1][-50:]]
        #
        #             # min_index = np.argmax(sim[inds, label].cpu().numpy())
        #             # proxy_pos = outputs[inds[min_index], label].unsqueeze(0)
        #             # sim[inds, label] = -10000
        #             # proxy_neg = outputs[inds[min_index], torch.sort(sim[inds[min_index]])[1][-length:]]
        #
        #             concated_input = torch.cat(
        #                 (proxy_pos, proxy_neg),
        #                 dim=0)
        #
        #             concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #                 torch.device('cuda'))
        #             concated_target[0] = 1.0
        #             # concated_target[0:len(inds)] = 1.0 / len(inds)
        #             proxy_cluster_loss += -1 * (
        #                     F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss = proxy_cluster_loss / proxy_cout

        #             for index in inds:
        #                 proxy_pos = outputs[index, label].unsqueeze(0)
        #             # proxy_pos = outputs[inds, label]
        #                 sim[index, label] = -10000.0
        #             # proxy_neg = outputs[inds, :].mean(0)[torch.sort(sim[inds, :].mean(0))[1][-10:]]
        #
        #                 proxy_neg = outputs[
        #                          index, torch.sort(sim[index])[1][-length:]]
        #                 # proxy_neg = torch.cat(
        #                 #     [outputs[
        #                 #          idx, torch.sort(sim[idx])[1][-length:]] for idx in inds], dim=0)
        #
        #                 concated_input = torch.cat(
        #                     (proxy_pos, proxy_neg),
        #                     dim=0)
        #
        #                 concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #                     torch.device('cuda'))
        #                 concated_target[0] = 1.0
        #                 # concated_target[0:len(inds)] = 1.0 / len(inds)
        #                 proxy_cluster_loss += -1 * (
        #                         F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #
        # loss = proxy_cluster_loss / len(targets)

        # for index, output in enumerate(outputs):
        #     target = targets[index]
        #     instance_pos = output[target].unsqueeze(0)
        #     sim[index, target] = -10000
        #     instance_neg = outputs[
        #         index, torch.sort(sim[index,])[1][-length:]]
        #     concated_input = torch.cat(
        #         (instance_pos, instance_neg),
        #         dim=0)
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #         torch.device('cuda'))
        #     concated_target[0] = 1.0
        #     proxy_cluster_loss += -1 * (
        #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss = proxy_cluster_loss / len(outputs)

        #     outputs /= 0.05
        #     loss = F.cross_entropy(outputs, targets)
        #
        # # kl_loss = self.kl(inputs, clusterfeature)
        # return self.lam * loss

        #
        # for label in unique_label:
        #     inds = torch.nonzero(targets == label).squeeze(-1)
        #     targets1 = targets[inds]
        #     outputs1 = outputs[inds, :]
        #     proxy_cluster_loss += F.cross_entropy(outputs1, targets1)
        # proxy_loss = proxy_cluster_loss / len(unique_label)
        #
        #
        # for label in unique_label:
        #     for cam in unique_cam:
        #         inds = [val for val in torch.nonzero(targets == label).squeeze(-1) if
        #                 val in torch.nonzero(camids == cam).squeeze(-1)]
        #         if len(inds) == 0:
        #             continue
        #         else:
        #             inds = torch.tensor(inds)
        #             uni_index_min = np.argmin(sim[inds, label].cpu())
        #             for idx in inds:
        #                 sim[idx, label] = -10000.0
        #
        #             concated_pos = outputs[inds[uni_index_min], label].unsqueeze(0)
        #
        #             # concated_pos = outputs[inds, label].mean(0).unsqueeze(0)
        #
        #             concated_neg = torch.cat(
        #                 [outputs[idx, torch.sort(sim[idx])[1][-10:]]
        #                  for idx in inds], dim=0)
        #
        #             concated_input = torch.cat(
        #                 (concated_pos, concated_neg),
        #                 dim=0)
        #
        #             concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #                 torch.device('cuda'))
        #
        #             concated_target[0] = 1.0
        #             proxy_cluster_loss += -1 * (
        #                     F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        #     label_loss += proxy_cluster_loss / len(unique_cam)
        # loss1 += proxy_cluster_loss / len(unique_label)
        # #proxy 对比 cluster
        #####################################################################

        # for label in unique_label:
        #     proxy_sim = sim.detach().clone()
        #     inds = torch.nonzero(targets == label).squeeze(-1)
        #     uni_index_min = np.argmin(proxy_sim[inds, label].cpu())
        #     proxy_sim[inds, label] = -10000.
        #     concated_pos = outputs[inds[uni_index_min], label].unsqueeze(0)
        #     concated_neg = torch.cat(
        #         [outputs[idx, torch.sort(proxy_sim[idx])[1][-15:]]
        #          for idx in inds], dim=0)
        #
        #     concated_input = torch.cat(
        #         (concated_pos, concated_neg),
        #         dim=0)
        #
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #         torch.device('cuda'))
        #     concated_target[0] = 1.0
        #     # proxy_cluster_loss += F.cross_entropy(outputs1, targets1)
        #     proxy_cluster_loss += -1 * (
        #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # proxy_loss = proxy_cluster_loss / len(unique_label)
        #####################################################################

        #     min_index = np.argmin(perlabel_proxy)
        #     concated_pos = perlabel_proxy[min_index].unsqueeze(0)
        #
        #     # concated_pos = torch.cat(perlabel_proxy.squeeze(-1), dim=0)
        #
        #     concated_neg = torch.cat(
        #         [others_proxy[idx][torch.sort(others_proxy[idx])[1][-20:]]
        #          for idx in range(len(perlabel_proxy))], dim=0)  # inds = 0
        #
        #     concated_input = torch.cat(
        #         (concated_pos, concated_neg),
        #         dim=0)
        #
        #     concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
        #         torch.device('cuda'))
        #
        #     # concated_target[0:len(perlabel_proxy)] = 1.0/len(perlabel_proxy)
        #     concated_target[0] = 1.0
        #     proxy_cluster_loss += -1 * (
        #             F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
        # loss += proxy_cluster_loss / len(unique_label)
