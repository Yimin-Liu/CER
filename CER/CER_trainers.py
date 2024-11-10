from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch import nn, autograd
import numpy as np
import collections
from CER.loss.contrastive import ViewContrastiveLoss
import math
from collections import defaultdict
from CER.loss.lp_loss import CriterionLP_intra


def del_tensor_ele(arr, index):
    arr1 = arr[0:index]
    arr2 = arr[index + 1:]
    return torch.cat((arr1, arr2), dim=0)


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
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


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

        batch_centers = collections.defaultdict(list)
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

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CamMemory(Function):

    @staticmethod
    def forward(self, inputs, targets, features, momentum):
        self.momentum = momentum
        self.features = features
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.features.t())
        return outputs

    @staticmethod
    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.features)
        for x, y in zip(inputs, targets):
            self.features[y] = self.momentum * self.features[y] + (1.0 - self.momentum) * x
            self.features[y] /= self.features[y].norm()
        return grad_inputs, None, None, None


class CamMemory_hard(Function):
    @staticmethod
    def forward(self, inputs, targets, features, momentum):
        self.momentum = momentum
        self.features = features
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(features.t())
        return outputs

    @staticmethod
    def backward(self, grad_outputs):

        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(self.features[index].unsqueeze(0).t().cuda())
                distances.append(distance.cpu().numpy())

            min_index = np.argmin(np.array(distances))
            self.features[index] = self.features[index] * self.momentum + (1 - self.momentum) * features[min_index]
            self.features[index] /= self.features[index].norm()
        return grad_inputs, None, None, None


def CM(inputs, indexes, features, momentum=0.5):
    return CamMemory.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def CM_hard(inputs, indexes, features, momentum=0.5):
    return CamMemory_hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


def get_min(a, b):
    if a >= b:
        return b
    else:
        return a


@torch.no_grad()
def getClusterMean(features, labels, uniquecams):
    sum_cams_cluster = 0
    centers = collections.defaultdict(list)
    for cc in uniquecams:
        perCamClusterLength = len(features[cc])
        for cluster, label in zip(features[cc], labels[cc]):
            if label == -1:
                continue
            centers[label].append(cluster)
        sum_cams_cluster += perCamClusterLength
    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]
    return centers, sum_cams_cluster


@torch.no_grad()
def getHardClusterMean(features, labels, uniquecams, inputs, pseudo_labels):
    inputs = F.normalize(inputs, dim=1).cuda()
    sum_cams_cluster = 0
    centers = collections.defaultdict(list)
    for cc in uniquecams:
        perCamClusterLength = len(features[cc])
        for cluster, label in zip(features[cc], labels[cc]):
            if label == -1:
                continue
            centers[label].append(cluster)
        sum_cams_cluster += perCamClusterLength
    centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]

    for label in torch.unique(pseudo_labels):
        inds = torch.nonzero(pseudo_labels == label).squeeze(-1)
        inputs_id = inputs[inds]
        distances = []
        for id in inputs_id:
            distance = id.unsqueeze(0).mm(centers[label].unsqueeze(0).t().cuda())
            distances.append(distance.cpu().numpy())
        min_index = np.argmin(np.array(distances))
        centers[label] = inputs[min_index]
    return centers, sum_cams_cluster


class HCCLTrainer(object):
    def __init__(self, args, encoder):
        super(HCCLTrainer, self).__init__()
        self.encoder = encoder
        self.crosscam_epoch = 0
        self.beta = args.beta
        self.bg_knn = 50  # 50
        self.args = args
        self.alpha = 0.01
        self.temp = args.temp
        self.neg_samp_intra_topk = args.neg_samp_intra_topk
        self.neg_samp_inter_topk = args.neg_samp_inter_topk
        # self.isIntraCons = args.intraC
        # self.isInterCons = args.interC
        self.isorigin = False
        self.kl = nn.KLDivLoss(reduction='batchmean')  # 加一个kl散度   f_out和memory
        self.vcloss = ViewContrastiveLoss(num_instance=16, T=0.1)
        # self.criterion_lp = CriterionLP(args)

        self.criterion_lp_intra = CriterionLP_intra(args)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def batchPLoss(self, input, proxy, proxy_targets, target):
        loss_caminter = torch.tensor([0.]).cuda()
        output = F.normalize(input).mm(F.normalize(proxy).t())
        temp_sims = output.detach().clone()
        output /= 0.07
        associate_loss = 0.
        uni_target = np.unique(target.cpu())
        for uni_index in uni_target:
            index = torch.nonzero(target == uni_index).squeeze(-1)
            ori_asso_ind = torch.nonzero(proxy_targets == uni_index).squeeze(-1)

            concated_inter_pos = torch.cat(
                [output[index, ori].mean(0).unsqueeze(0)
                 for ori in ori_asso_ind], dim=0)

            for ind in ori_asso_ind:
                temp_sims[index, ind] = -10000.0

            concated_inter_neg = torch.cat(
                [output[
                     idx, torch.sort(temp_sims[idx])[1][-(len(proxy_targets) - len(ori_asso_ind)):]] for idx
                 in
                 index], dim=0)

            concated_inter_input = torch.cat((concated_inter_pos, concated_inter_neg), dim=0)
            concated_inter_target = torch.zeros((len(concated_inter_input)),
                                                dtype=concated_inter_input.dtype).to(
                torch.device('cuda'))
            concated_inter_target[0: len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)

            associate_loss += -1 * (
                    F.log_softmax(concated_inter_input.unsqueeze(0),
                                  dim=1) * concated_inter_target.unsqueeze(
                0)).sum()
        loss_caminter += associate_loss / len(uni_target)  # 这里缩紧了
        return loss_caminter

    def calculate_loss_camintra_con(self, args, labels, camids, f_out, epoch, memory_class_mapper, init_intra_id_feat,
                                    percam_tempV, cams_num):

        loss_camintra_con = torch.tensor([0.]).cuda()
        uniquepercams = torch.unique(camids)
        uniquecams = torch.unique(self.all_img_cams)
        for cc in uniquepercams:
            inds = torch.nonzero(camids == cc).squeeze(-1)
            if len(inds) == 0:
                continue
            percam_targets = labels[inds]
            percam_feat = f_out[inds]

            intra_cam_proxy_features = percam_tempV[cams_num[cc]:cams_num[cc + 1]]

            uni_target = np.unique(percam_targets.cpu())
            # if self.isIntraCons:
            target_intra_inputs = torch.matmul(percam_feat,
                                               intra_cam_proxy_features.cuda().t())
            temp_sims_intra = target_intra_inputs.detach().clone()
            target_intra_inputs /= self.beta2
            intra_loss = 0
            for uni_index in uni_target:
                index = torch.nonzero(percam_targets == uni_index).squeeze(-1)
                pos_ind = memory_class_mapper[cc][int(uni_index)]

                uni_index_min = np.argmin(temp_sims_intra[index, pos_ind].cpu())

                for idx in index:
                    temp_sims_intra[idx, pos_ind] = -10000.0

                neg_sec = get_min(len(intra_cam_proxy_features) - 1, self.neg_samp_intra_topk)
                concated_neg = torch.cat(
                    [target_intra_inputs[idx, torch.sort(temp_sims_intra[idx])[1][-neg_sec:]]
                     for idx in index], dim=0)  # 小于最大数的预值

                concated_pos = target_intra_inputs[index[uni_index_min], pos_ind].unsqueeze(0)
                concated_input = torch.cat(
                    (concated_pos, concated_neg),
                    dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))

                concated_target[0] = 1.0
                # concated_target[0] = 1.0 / len(index)
                intra_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(0)).sum()
            loss_camintra_con += intra_loss / len(uni_target)
        return loss_camintra_con

    def calculate_loss_camintra(self, args, labels, camids, f_out, epoch, memory_class_mapper, init_intra_id_feat,
                                percam_tempV, cams_num):
        loss_camintra = torch.tensor([0.]).cuda()
        uniquepercams = torch.unique(camids)
        uniquecams = torch.unique(self.all_img_cams)

        for cc in uniquepercams:
            intra_cam_proxy_features = percam_tempV[cams_num[cc]:cams_num[cc + 1]]
            inds = torch.nonzero(camids == cc).squeeze(-1)
            if len(inds) == 0:
                continue
            percam_targets = labels[inds]
            percam_feat = f_out[inds]
            mapped_targets = [memory_class_mapper[cc][int(k)] for k in percam_targets]
            mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
            percam_inputs = torch.matmul(percam_feat, intra_cam_proxy_features.t().cuda())
            # percam_inputs = kendalltau(percam_feat, intra_cam_proxy_features)
            percam_inputs /= self.beta  # similarity score before softmax
            loss_camintra += F.cross_entropy(percam_inputs, mapped_targets.long())
        return loss_camintra

    # def calculate_loss_camintra_hard(self, args, labels, camids, f_out, epoch, memory_class_mapper, init_intra_id_feat,
    #                             percam_tempV, cams_num):
    #     loss_camintra = torch.tensor([0.]).cuda()
    #     uniquepercams = torch.unique(camids)
    #     uniquecams = torch.unique(self.all_img_cams)
    #
    #     for cc in uniquepercams:
    #         intra_cam_proxy_features = percam_tempV[cams_num[cc]:cams_num[cc + 1]]
    #         inds = torch.nonzero(camids == cc).squeeze(-1)
    #         if len(inds) == 0:
    #             continue
    #         percam_targets = labels[inds]
    #         percam_feat = f_out[inds]
    #         mapped_targets = [memory_class_mapper[cc][int(k)] for k in percam_targets]
    #         mapped_targets = torch.tensor(mapped_targets).to(torch.device('cuda'))
    #         percam_inputs = torch.matmul(percam_feat, intra_cam_proxy_features.t().cuda())
    #         # percam_inputs = kendalltau(percam_feat, intra_cam_proxy_features)
    #         percam_inputs /= self.beta  # similarity score before softmax
    #         loss = 0.
    #         percam_inputs = torch.exp(percam_inputs)
    #         for idx, (exp_i_s, lb) in enumerate(zip(percam_inputs, mapped_targets)):
    #             pos_sim, pos_ind, neg_sim, neg_ind = self.find_hardest_support2(exp_i_s, idx, lb, labels_s)
    #             loss += (-torch.log((pos_sim.sum() / (pos_sim.sum() + neg_sim.sum() + 1e-6)) + 1e-6))
    #         loss_camintra = loss / len(percam_inputs)
    #
    #         loss_camintra += F.cross_entropy(percam_inputs, mapped_targets.long())
    #     return loss_camintra

    def calculate_loss_caminter_con(self, args, labels, camids, f_out, epoch, memory_class_mapper, init_intra_id_feat,
                                    percam_tempV):
        loss_caminter_con = torch.tensor([0.]).cuda()
        uniquepercams = torch.unique(camids)
        uniquecams = torch.unique(self.all_img_cams)
        for cc in uniquepercams:
            inds = torch.nonzero(camids == cc).squeeze(-1)
            if len(inds) == 0:
                continue
            percam_targets = labels[inds]
            percam_feat = f_out[inds]
            uni_target = np.unique(percam_targets.cpu())
            associate_loss = 0
            target_inter_inputs = torch.matmul(percam_feat, percam_tempV.t().clone())
            # lenneg = len(percam_tempV) - 1
            temp_sims_inter = target_inter_inputs.detach().clone()
            target_inter_inputs /= self.beta2
            for uni_index in uni_target:
                index = torch.nonzero(percam_targets == uni_index).squeeze(-1)
                ori_asso_ind = torch.nonzero(self.concate_intra_class == uni_index).squeeze(-1)

                if len(ori_asso_ind) == 0:
                    continue
                concated_inter_pos = torch.cat(
                    [target_inter_inputs[index, ori].mean(0).unsqueeze(0)
                     for ori in ori_asso_ind], dim=0)
                for ind in ori_asso_ind:
                    temp_sims_inter[index, ind] = -10000.0

                concated_inter_neg = torch.cat(
                    [target_inter_inputs[
                         idx, torch.sort(temp_sims_inter[idx])[1][-self.neg_samp_inter_topk:]] for idx
                     in
                     index], dim=0)

                concated_inter_input = torch.cat((concated_inter_pos, concated_inter_neg), dim=0)
                concated_inter_target = torch.zeros((len(concated_inter_input)),
                                                    dtype=concated_inter_input.dtype).to(
                    torch.device('cuda'))
                concated_inter_target[0: len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)

                associate_loss += -1 * (
                        F.log_softmax(concated_inter_input.unsqueeze(0),
                                      dim=1) * concated_inter_target.unsqueeze(
                    0)).sum()
            loss_caminter_con += associate_loss / len(uni_target)  # 这里缩紧了
        return loss_caminter_con

    def calculate_loss_caminter(self, args, labels, camids, f_out, epoch, memory_class_mapper, init_intra_id_feat,
                                percam_tempV):
        loss_caminter = torch.tensor([0.]).cuda()
        uniquepercams = torch.unique(camids)
        for cc in uniquepercams:
            inds = torch.nonzero(camids == cc).squeeze(-1)
            if len(inds) == 0:
                continue
            percam_targets = labels[inds]
            percam_feat = f_out[inds]

            associate_loss = 0
            target_inter_inputs = torch.matmul(percam_feat, percam_tempV.t().clone())
            temp_sims = target_inter_inputs.detach().clone()
            target_inter_inputs /= self.beta
            for k in range(len(percam_feat)):
                ori_asso_ind = torch.nonzero(self.concate_intra_class == percam_targets[k]).squeeze(-1)
                if len(ori_asso_ind) == 0:
                    continue
                temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
                sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
                concated_input = torch.cat(
                    (target_inter_inputs[k, ori_asso_ind], target_inter_inputs[k, sel_ind]),
                    dim=0)
                concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                    torch.device('cuda'))
                concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
                associate_loss += -1 * (
                        F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                    0)).sum()
            loss_caminter += associate_loss / len(percam_feat)
        return loss_caminter

    def getCamLoss(self, args, labels, camids, f_out, epoch, memory_class_mapper, init_intra_id_feat, percam_tempV,
                   cams_num):

        self.beta2 = 0.07
        # loss_camintra_con = self.calculate_loss_camintra_con(args, labels, camids, f_out, epoch, memory_class_mapper,
        #                                                      init_intra_id_feat, percam_tempV, cams_num)
        # loss_caminter_con = self.calculate_loss_caminter_con(args, labels, camids, f_out, epoch, memory_class_mapper,
        #                                                      init_intra_id_feat, percam_tempV)
        loss_camintra_con = self.calculate_loss_camintra(args, labels, camids, f_out, epoch, memory_class_mapper,
                                                         init_intra_id_feat, percam_tempV, cams_num)
        loss_caminter_con = self.calculate_loss_caminter(args, labels, camids, f_out, epoch, memory_class_mapper,
                                                         init_intra_id_feat, percam_tempV)

        return loss_camintra_con, loss_caminter_con

    def getCamLoss_cross(self, args, f_out_inter, labels_inter, camids_inter, f_out_intra, labels_intra, camids_intra,
                         epoch, memory_class_mapper, init_intra_id_feat, percam_tempV, cams_num):

        self.beta2 = 0.07
        # loss_camintra_con = self.calculate_loss_camintra_con(args, labels_intra, camids_intra, f_out_intra, epoch,
        #                                                      memory_class_mapper,
        #                                                      init_intra_id_feat, percam_tempV, cams_num)
        # loss_caminter_con = self.calculate_loss_caminter_con(args, labels_inter, camids_inter, f_out_inter, epoch,
        #                                                      memory_class_mapper,
        #                                                      init_intra_id_feat, percam_tempV)

        # loss_camintra_con = self.calculate_loss_camintra_con(args, labels_inter, camids_inter, f_out_inter, epoch,
        #                                                  memory_class_mapper,
        #                                                  init_intra_id_feat, percam_tempV, cams_num)
        # loss_caminter_con = self.calculate_loss_caminter_con(args, labels_intra, camids_intra, f_out_intra, epoch,
        #                                                  memory_class_mapper,
        #                                                  init_intra_id_feat, percam_tempV)

        loss_camintra_con = self.calculate_loss_camintra(args, labels_intra, camids_intra, f_out_intra, epoch,
                                                         memory_class_mapper,
                                                         init_intra_id_feat, percam_tempV, cams_num)
        loss_caminter_con = self.calculate_loss_caminter(args, labels_inter, camids_inter, f_out_inter, epoch,
                                                         memory_class_mapper,
                                                         init_intra_id_feat, percam_tempV)
        return loss_camintra_con, loss_caminter_con

    def getProMemLoss(self, args, percam_tempV, percam_tempV_epoch, concate_intra_class):
        self.beta_con = self.beta
        ProMemLoss = torch.tensor([0.]).cuda()

        target_inter_inputs = torch.matmul(percam_tempV, percam_tempV_epoch.t())
        temp_sims = target_inter_inputs.detach().clone()
        target_inter_inputs /= 0.07

        associate_loss = 0.
        for k in range(len(percam_tempV)):
            ori_asso_ind = torch.nonzero(concate_intra_class == concate_intra_class[k]).squeeze(-1)
            if len(ori_asso_ind) == 0:
                continue
            temp_sims[k, ori_asso_ind] = -10000.0  # mask out positive
            sel_ind = torch.sort(temp_sims[k])[1][-self.bg_knn:]
            concated_input = torch.cat(
                (target_inter_inputs[k, ori_asso_ind], target_inter_inputs[k, sel_ind]),
                dim=0)
            concated_target = torch.zeros((len(concated_input)), dtype=concated_input.dtype).to(
                torch.device('cuda'))
            concated_target[0:len(ori_asso_ind)] = 1.0 / len(ori_asso_ind)
            associate_loss += -1 * (
                    F.log_softmax(concated_input.unsqueeze(0), dim=1) * concated_target.unsqueeze(
                0)).sum()
        ProMemLoss += associate_loss / len(percam_tempV)
        return ProMemLoss

    def proxy_memory_update(self, f_out, feature_id, percam_tempV, momentum):
        batch_centers = defaultdict(list)
        for instance_feature, index in zip(f_out, feature_id.tolist()):
            batch_centers[index].append(instance_feature)
        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(percam_tempV[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())
            median = np.argmin(np.array(distances))
            percam_tempV[index] = percam_tempV[index] * momentum + (1 - momentum) * features[median]
            percam_tempV[index] /= percam_tempV[index].norm()

    def proxy_memory_update_normal(self, f_out, feature_id, percam_tempV, momentum):
        batch_centers = defaultdict(list)
        for instance_feature, index in zip(f_out, feature_id.tolist()):
            batch_centers[index].append(instance_feature)
        for index, features in batch_centers.items():
            for feature in features:
                percam_tempV[index] = percam_tempV[index] * momentum + (1 - momentum) * feature
                percam_tempV[index] /= percam_tempV[index].norm()

    def torch_intersect(self, t1, t2, use_unique=False):
        t1 = t1.cuda()
        t2 = t2.cuda()
        t1 = t1.unique()
        t2 = t2.unique()
        return torch.tensor(np.intersect1d(t1.cpu().numpy(), t2.cpu().numpy()))

    def camAug_inter_aug_all(self, args, features, percam_memory, memory_class_mapper, features_labels, features_camids,
                             proxy_memory_ids,
                             proxy_memory_camids,
                             uniquecams, cur_lambda, cams_nums):

        feature_aug = []
        feature_aug_labels = []
        feature_aug_camids = []

        # TTF
        degree = args.degree
        for feature, label, camid in zip(features, features_labels, features_camids):
            inds_diff_cams = torch.nonzero(proxy_memory_ids == label)
            inds_same_cams = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                  torch.nonzero(proxy_memory_camids == camid))
            diff_cam_all_proxy_features = []
            diff_cam_same_id_features = []
            for index in inds_diff_cams:
                diff_camid = proxy_memory_camids[index]
                diff_cam_proxy_features = percam_memory[cams_nums[diff_camid]:cams_nums[diff_camid + 1]]
                target_index = torch.tensor(memory_class_mapper[diff_camid][int(label)])
                inds_same = torch.nonzero(
                    torch.arange(0, len(memory_class_mapper[diff_camid]), 1) != target_index).squeeze(
                    -1)  # 这里，没有就是全部
                if diff_camid == camid:  # DS
                    sim = feature.unsqueeze(0).mm(diff_cam_proxy_features[inds_same].t())
                    _, indexes = torch.sort(sim, dim=1, descending=False)
                    uni_index_max = indexes[:, :1].squeeze(0)
                    aug_feature = feature + degree * cur_lambda * (
                            diff_cam_proxy_features[inds_same][uni_index_max].squeeze(0) - percam_memory[
                        inds_same_cams].squeeze(0))
                    feature_aug.append(aug_feature.unsqueeze(0))
                    feature_aug_labels.append(label.unsqueeze(0))
                    feature_aug_camids.append(camid.unsqueeze(0))
                else:
                    diff_cam_same_id_features.append(percam_memory[index])
                    diff_cam_all_proxy_features.append(diff_cam_proxy_features[inds_same])
            if len(diff_cam_all_proxy_features) != 0:  # DD
                diff_cam_all_proxy = torch.cat(diff_cam_all_proxy_features, dim=0)
                sim = feature.unsqueeze(0).mm(diff_cam_all_proxy.t())
                _, indexes = torch.sort(sim, dim=1, descending=False)
                uni_index_max = indexes[:, :1].squeeze(0)
                aug_feature = feature + degree * cur_lambda * (
                        diff_cam_all_proxy[uni_index_max].squeeze(0) - percam_memory[
                    inds_same_cams].squeeze(0))
                feature_aug.append(aug_feature.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
            if len(diff_cam_same_id_features) != 0:  # SD
                diff_cam_same_id_proxy = torch.cat(diff_cam_same_id_features, dim=0)
                sim = feature.unsqueeze(0).mm(diff_cam_same_id_proxy.t())
                _, indexes = torch.sort(sim, dim=1, descending=True)
                uni_index_max = indexes[:, :1].squeeze(0)
                aug_feature = feature - degree * cur_lambda * (
                        diff_cam_same_id_proxy[uni_index_max].squeeze(0) - percam_memory[
                    inds_same_cams].squeeze(0))
                feature_aug.append(aug_feature.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
        feature_aug_1 = torch.cat(feature_aug, dim=0).cuda()
        feature_aug_labels_1 = torch.cat(feature_aug_labels, dim=0).cuda()
        feature_aug_camids_1 = torch.cat(feature_aug_camids, dim=0).cuda()
        return feature_aug_1, feature_aug_labels_1, feature_aug_camids_1

    def camAug_support(self, args, features, percam_memory, memory_class_mapper, features_labels, features_camids,
                       proxy_memory_ids,
                       proxy_memory_camids,
                       uniquecams, cur_lambda, cams_nums):

        feature_aug = []
        feature_aug_labels = []
        feature_aug_camids = []
        degree = args.degree
        for feature, label, camid in zip(features, features_labels, features_camids):
            inds_same_cams_diff_ids = self.torch_intersect(torch.nonzero(proxy_memory_ids != label),
                                                           torch.nonzero(proxy_memory_camids == camid))

            inds_diff_cams_same_ids = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                           torch.nonzero(proxy_memory_camids != camid))

            inds_diff_cams_diff_ids = self.torch_intersect(torch.nonzero(proxy_memory_ids != label),
                                                           torch.nonzero(proxy_memory_camids != camid))

            inds_feature = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                torch.nonzero(proxy_memory_camids == camid))

            sim_same_cams_diff_ids = feature.unsqueeze(0).mm(percam_memory[inds_same_cams_diff_ids].t())
            _, indexes = torch.sort(sim_same_cams_diff_ids, dim=1, descending=True)
            uni_index_max = indexes[:, :1].squeeze(0)
            aug_feature_diff_person = feature + degree * cur_lambda * (
                    percam_memory[inds_same_cams_diff_ids][uni_index_max].squeeze(0) - percam_memory[
                inds_feature].squeeze(0))
            feature_aug.append(aug_feature_diff_person.unsqueeze(0))
            feature_aug_labels.append(label.unsqueeze(0))
            feature_aug_camids.append(camid.unsqueeze(0))

            sim_diff_cams_diff_ids = feature.unsqueeze(0).mm(percam_memory[inds_diff_cams_diff_ids].t())
            _, indexes = torch.sort(sim_diff_cams_diff_ids, dim=1, descending=True)
            uni_index_max = indexes[:, :1].squeeze(0)
            aug_feature = feature + degree * cur_lambda * (
                    percam_memory[inds_diff_cams_diff_ids][uni_index_max].squeeze(0) - percam_memory[
                inds_feature].squeeze(0))
            feature_aug.append(aug_feature.unsqueeze(0))
            feature_aug_labels.append(label.unsqueeze(0))
            feature_aug_camids.append(camid.unsqueeze(0))

            if len(inds_diff_cams_same_ids) != 0:
                sim_diff_cams_same_ids = feature.unsqueeze(0).mm(percam_memory[inds_diff_cams_same_ids].t())
                _, indexes = torch.sort(sim_diff_cams_same_ids, dim=1, descending=False)
                uni_index_max = indexes[:, :1].squeeze(0)
                aug_feature_same_person = feature - degree * cur_lambda * (
                        percam_memory[inds_diff_cams_same_ids][uni_index_max].squeeze(0) - percam_memory[
                    inds_feature].squeeze(0))
                feature_aug.append(aug_feature_same_person.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
        feature_aug_1 = torch.cat(feature_aug, dim=0).cuda()
        feature_aug_labels_1 = torch.cat(feature_aug_labels, dim=0).cuda()
        feature_aug_camids_1 = torch.cat(feature_aug_camids, dim=0).cuda()
        return feature_aug_1, feature_aug_labels_1, feature_aug_camids_1

    def camera_aware_support_sd(self, args, features, percam_memory, memory_class_mapper, features_labels,
                                features_camids,
                                proxy_memory_ids,
                                proxy_memory_camids,
                                cur_lambda, degree, cams_nums):

        feature_aug = []
        feature_aug_labels = []
        feature_aug_camids = []
        for feature, label, camid in zip(features, features_labels, features_camids):
            inds_same_cams_diff_ids = self.torch_intersect(torch.nonzero(proxy_memory_ids != label),
                                                           torch.nonzero(proxy_memory_camids == camid))
            inds_feature = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                torch.nonzero(proxy_memory_camids == camid))
            sim_same_cams_diff_ids = feature.unsqueeze(0).mm(percam_memory[inds_same_cams_diff_ids].t())
            _, indexes = torch.sort(sim_same_cams_diff_ids, dim=1, descending=True)
            uni_index_max = indexes[:, :1].squeeze(0)
            aug_feature = feature + degree * cur_lambda * (
                    percam_memory[inds_same_cams_diff_ids][uni_index_max].squeeze(0) - percam_memory[
                inds_feature].squeeze(0))
            feature_aug.append(aug_feature.unsqueeze(0))
            feature_aug_labels.append(label.unsqueeze(0))
            feature_aug_camids.append(camid.unsqueeze(0))
        feature_aug_1 = torch.cat(feature_aug, dim=0).cuda()
        feature_aug_labels_1 = torch.cat(feature_aug_labels, dim=0).cuda()
        feature_aug_camids_1 = torch.cat(feature_aug_camids, dim=0).cuda()
        return feature_aug_1, feature_aug_labels_1, feature_aug_camids_1

    def camera_aware_support_ds(self, args, features, percam_memory, memory_class_mapper, features_labels,
                                features_camids,
                                proxy_memory_ids,
                                proxy_memory_camids,
                                cur_lambda, degree, cams_nums):

        feature_aug = []
        feature_aug_labels = []
        feature_aug_camids = []
        for feature, label, camid in zip(features, features_labels, features_camids):
            inds_diff_cams_same_ids = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                           torch.nonzero(proxy_memory_camids != camid))

            inds_feature = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                torch.nonzero(proxy_memory_camids == camid))

            if len(inds_diff_cams_same_ids) != 0:
                sim_diff_cams_same_ids = feature.unsqueeze(0).mm(percam_memory[inds_diff_cams_same_ids].t())
                _, indexes = torch.sort(sim_diff_cams_same_ids, dim=1, descending=False)
                uni_index_max = indexes[:, :1].squeeze(0)
                aug_feature = feature - degree * cur_lambda * (
                        percam_memory[inds_diff_cams_same_ids][uni_index_max].squeeze(0) - percam_memory[
                    inds_feature].squeeze(0))
                feature_aug.append(aug_feature.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
            else:
                feature_aug.append(feature.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
        feature_aug_1 = torch.cat(feature_aug, dim=0).cuda()
        feature_aug_labels_1 = torch.cat(feature_aug_labels, dim=0).cuda()
        feature_aug_camids_1 = torch.cat(feature_aug_camids, dim=0).cuda()
        return feature_aug_1, feature_aug_labels_1, feature_aug_camids_1

    def camera_aware_support_dd(self, args, features, percam_memory, memory_class_mapper, features_labels,
                                features_camids,
                                proxy_memory_ids,
                                proxy_memory_camids,
                                cur_lambda, degree, cams_nums):

        feature_aug = []
        feature_aug_labels = []
        feature_aug_camids = []
        for feature, label, camid in zip(features, features_labels, features_camids):
            inds_diff_cams_same_ids = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                           torch.nonzero(proxy_memory_camids != camid))

            inds_feature = self.torch_intersect(torch.nonzero(proxy_memory_ids == label),
                                                torch.nonzero(proxy_memory_camids == camid))

            diff_cam_all_proxy_features = []
            for index in inds_diff_cams_same_ids:
                diff_camid = proxy_memory_camids[index]
                diff_cam_proxy_features = percam_memory[cams_nums[diff_camid]:cams_nums[diff_camid + 1]]
                target_index = torch.tensor(memory_class_mapper[diff_camid][int(label)])
                inds_same = torch.nonzero(
                    torch.arange(0, len(memory_class_mapper[diff_camid]), 1) != target_index).squeeze(
                    -1)
                diff_cam_all_proxy_features.append(diff_cam_proxy_features[inds_same])

            if len(diff_cam_all_proxy_features) != 0:
                diff_cam_all_proxy = torch.cat(diff_cam_all_proxy_features, dim=0)
                sim = feature.unsqueeze(0).mm(diff_cam_all_proxy.t())
                _, indexes = torch.sort(sim, dim=1, descending=True)
                uni_index_max = indexes[:, :1].squeeze(0)
                aug_feature = feature + degree * cur_lambda * (
                        diff_cam_all_proxy[uni_index_max].squeeze(0) - percam_memory[
                    inds_feature].squeeze(0))
                feature_aug.append(aug_feature.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
            else:
                feature_aug.append(feature.unsqueeze(0))
                feature_aug_labels.append(label.unsqueeze(0))
                feature_aug_camids.append(camid.unsqueeze(0))
        feature_aug_1 = torch.cat(feature_aug, dim=0).cuda()
        feature_aug_labels_1 = torch.cat(feature_aug_labels, dim=0).cuda()
        feature_aug_camids_1 = torch.cat(feature_aug_camids, dim=0).cuda()
        return feature_aug_1, feature_aug_labels_1, feature_aug_camids_1

    def train(self, args, epoch, data_loader, optimizer, print_freq=10, train_iters=400, intra_id_labels=None,
              intra_id_features=None,
              cams=None, concate_intra_class_list=None, concate_intra_cams=None, uniquecams=None,
              memory_class_mapper=None):
        global cur_lambda
        self.encoder.train()
        # self.momentum = 0.5

        batch_time = AverageMeter()
        data_time = AverageMeter()

        cluster_losses = AverageMeter()
        cam_inter = AverageMeter()
        cam_intra = AverageMeter()

        self.all_img_cams = torch.tensor(cams).cuda()
        self.init_intra_id_feat = intra_id_features

        cams_nums = []
        cam_last_id = 0
        cams_nums.append(cam_last_id)
        self.percam_memory = []
        for cc in uniquecams:
            if len(intra_id_features) > 0:
                # print('initializing ID memory from updated embedding features...')
                proto_memory = intra_id_features[cc].to(torch.device('cuda'))
                cam_last_id += len(proto_memory)
                cams_nums.append(cam_last_id)
                self.percam_memory.append(proto_memory.detach())
        self.concate_intra_class = torch.cat(concate_intra_class_list)
        self.percam_tempV = torch.cat(self.percam_memory, dim=0).cuda()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            lamda_type = 'loge'

            if lamda_type == 'loge':
                cur_lambda = np.log((math.e - 1) * (epoch * train_iters + i) / (
                        train_iters * self.args.epochs) + 1)
            elif lamda_type == 'log2':
                cur_lambda = np.log2((2 - 1) * (epoch * train_iters + i) / (
                        train_iters * self.args.epochs) + 1)
            elif lamda_type == 'log10':
                cur_lambda = np.log10((10 - 1) * (epoch * train_iters + i) / (
                        train_iters * self.args.epochs) + 1)
            elif lamda_type == 'square':
                cur_lambda = ((epoch * train_iters + i) / (
                        train_iters * self.args.epochs)) ** 2
            elif lamda_type == 'cubed':
                cur_lambda = ((epoch * train_iters + i) / (
                        train_iters * self.args.epochs)) ** 3

            # process inputs
            inputs, labels, camids, indexes = self._parse_data(inputs)

            if args.dataset == 'msmt17':
                camids = camids - 1

            f_out = self.encoder(inputs, cam_label=camids)

            f_out_support_inter, support_labels_inter, support_camids_inter = self.camAug_inter_aug_all(args,
                                                                                                        f_out,
                                                                                                        self.percam_tempV,
                                                                                                        memory_class_mapper,
                                                                                                        labels, camids,
                                                                                                        self.concate_intra_class,
                                                                                                        concate_intra_cams,
                                                                                                        torch.from_numpy(
                                                                                                            uniquecams).cuda(),
                                                                                                        cur_lambda,
                                                                                                        cams_nums)

            feature_id = []
            for person_id, camid in zip(support_labels_inter, support_camids_inter):
                feature_id.append(
                    cams_nums[camid] + torch.nonzero(concate_intra_class_list[camid] == person_id).squeeze(0))
            feature_id = torch.cat(feature_id, dim=0).cuda()

            loss_camintra_con, loss_caminter_con = self.getCamLoss(args, support_labels_inter, support_camids_inter,
                                                                   f_out_support_inter, epoch,
                                                                   memory_class_mapper,
                                                                   self.percam_memory,
                                                                   self.percam_tempV,
                                                                   cams_nums)

            loss_lp_intra = self.criterion_lp_intra(f_out, f_out_support_inter, labels,
                                                    support_labels_inter)

            loss = 0.2 * loss_camintra_con + loss_caminter_con + loss_lp_intra
            optimizer.zero_grad()
            loss.backward()
            with torch.no_grad():
                if args.dataset == 'market1501':
                    self.proxy_memory_update(f_out_support_inter, feature_id, self.percam_tempV, 0.1)
                else:
                    self.proxy_memory_update_normal(f_out_support_inter, feature_id, self.percam_tempV, 0.1)
            optimizer.step()

            cluster_losses.update(loss_caminter_con.item())
            cam_inter.update(loss_caminter_con.item())
            cam_intra.update(loss_camintra_con.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'ClusterLoss {:.3f} ({:.3f})\t'
                      'CaminterLoss {:.3f} ({:.3f})\t'
                      'CamintraLoss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              cluster_losses.val, cluster_losses.avg,
                              cam_inter.val, cam_inter.avg,
                              cam_intra.val, cam_intra.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, camids, indexes = inputs
        return imgs.cuda(), pids.cuda(), camids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
