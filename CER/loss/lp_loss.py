from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriterionLP(nn.Module):
    """Label Preserving Loss in ISE"""

    def __init__(self, args):
        super(CriterionLP, self).__init__()
        self.args = args

    def forward(self, feats, feats_s, labels, labels_s):
        B = feats.size(0)
        C = feats.size(-1)
        topk = self.args.topk

        B_s = feats_s.size(0)
        C_s = feats_s.size(-1)

        feats_s = feats_s.reshape(B_s * topk, C_s)
        sim_instance_support = feats.mm(feats_s.t()) / self.args.temp_lp_loss
        exp_instance_support = torch.exp(sim_instance_support)
        loss = 0
        for idx, (exp_i_s, lb) in enumerate(zip(exp_instance_support, labels)):
            pos_sim, pos_ind, neg_sim, neg_ind = self.find_hardest_support(exp_i_s, idx, lb, labels_s)
            loss += (-torch.log((pos_sim.sum() / (pos_sim.sum() + neg_sim.sum() + 1e-6)) + 1e-6))
        loss = loss / B
        return loss

    def find_hardest_support(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        sim_negs = sim[is_neg]
        try:
            is_neg = is_neg.reshape(-1, self.args.topk * self.args.num_instances)
            sim_negs = sim_negs.reshape(-1, self.args.topk * self.args.num_instances)
            neg_sim, relative_n_inds = torch.max(sim_negs.contiguous(), 1, keepdim=False)
        except:
            print("is_neg shape is: {}".format(is_neg.size()))
            print("sim_negs shape is: {}".format(sim_negs.size()))
            print("use all negative samples")
            relative_n_inds = None
            neg_sim = sim_negs

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds

    def find_hardest_support2(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        neg_sim = [], relative_n_inds = []
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        sim_negs = sim[is_neg]

        diff_labels = torch.unique(labels_s[is_neg])

        for diff_label in diff_labels:
            sim_negs = sim[torch.nonzero(labels_s == diff_label)]
            neg = torch.max(sim_negs.contiguous(), 1, keepdim=False)
            neg_sim.append(neg[0])
            relative_n_inds.append(neg[1])
        neg_sim = torch.cat(neg_sim, dim=0).cuda()
        relative_n_inds = torch.cat(relative_n_inds, dim=0).cuda()

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds


class CriterionLP_all(nn.Module):
    """Label Preserving Loss in ISE"""

    def __init__(self, args):
        super(CriterionLP_all, self).__init__()
        self.args = args

    def forward(self, feats, feats_s, labels, labels_s):
        B = feats.size(0)
        C = feats.size(-1)
        topk = self.args.intra_topk
        B_s = feats_s.size(0)
        C_s = feats_s.size(-1)

        feats_s = feats_s.reshape(B_s * topk, C_s)
        sim_instance_support = feats.mm(feats_s.t()) / self.args.temp_lp_loss
        # exp_instance_support = torch.exp(sim_instance_support)
        loss = 0
        associate_loss = 0
        for idx, (exp_i_s, lb) in enumerate(zip(sim_instance_support, labels)):
            pos_sim, neg_sim = self.find_all_support(exp_i_s, idx, lb, labels_s)

            concated_input = torch.cat(
                (pos_sim, neg_sim),
                dim=1)
            concated_target = torch.zeros((len(concated_input[0])), dtype=concated_input.dtype).to(
                torch.device('cuda'))
            concated_target[0:len(pos_sim[0])] = 1.0 / len(pos_sim[0])
            associate_loss += -1 * (
                    F.log_softmax(concated_input, dim=1) * concated_target.unsqueeze(
                0)).sum()
        loss = associate_loss / B
        return loss

    def find_all_support(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim = sim[is_pos].contiguous().view(1, -1)
        neg_sim = sim[is_neg].contiguous().view(1, -1)
        return pos_sim, neg_sim


class CriterionLP_intra(nn.Module):
    """Label Preserving Loss in ISE"""

    def __init__(self, args):
        super(CriterionLP_intra, self).__init__()
        self.args = args

    def forward(self, feats, feats_s, labels, labels_s):
        B = feats.size(0)
        C = feats.size(-1)
        topk = self.args.intra_topk
        B_s = feats_s.size(0)
        C_s = feats_s.size(-1)

        feats_s = feats_s.reshape(B_s * topk, C_s)
        sim_instance_support = feats.mm(feats_s.t()) / self.args.temp_lp_loss
        exp_instance_support = torch.exp(sim_instance_support)
        loss = 0
        for idx, (exp_i_s, lb) in enumerate(zip(exp_instance_support, labels)):
            pos_sim, pos_ind, neg_sim, neg_ind = self.find_hardest_support5(exp_i_s, idx, lb, labels_s)
            loss += (-torch.log((pos_sim.sum() / (pos_sim.sum() + neg_sim.sum() + 1e-6)) + 1e-6))
            # loss += (-torch.log((pos_sim.sum() / (pos_sim.sum() + neg_sim.sum()))))
        loss = loss / B
        return loss


    def find_hardest_support(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        sim_negs = sim[is_neg]
        # labels_neg = labels_s[is_neg]
        try:
            is_neg = is_neg.reshape(-1, 1 * self.args.num_instances)
            sim_negs = sim_negs.reshape(-1, 1 * self.args.num_instances)
            neg_sim, relative_n_inds = torch.max(sim_negs.contiguous(), 1, keepdim=False)
        except:
            print("is_neg shape is: {}".format(is_neg.size()))
            print("sim_negs shape is: {}".format(sim_negs.size()))
            print("use all negative samples")
            relative_n_inds = None
            neg_sim = sim_negs

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds

    def find_hardest_support_3(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        sim_negs = sim[is_neg]
        # labels_neg = labels_s[is_neg]
        try:
            is_neg = is_neg.reshape(-1, 1 * self.args.num_instances * 3)
            sim_negs = sim_negs.reshape(-1, 1 * self.args.num_instances * 3)
            # labels_neg = labels_neg.reshape(-1, 1 * self.args.num_instances * 3)
            is_neg = is_neg.reshape(-1, 1 * self.args.num_instances)
            sim_negs = sim_negs.reshape(-1, 1 * self.args.num_instances)
            # labels_neg = labels_neg.reshape(-1, 1 * self.args.num_instances)
            neg_sim, relative_n_inds = torch.max(sim_negs.contiguous(), 1, keepdim=False)
        except:
            print("is_neg shape is: {}".format(is_neg.size()))
            print("sim_negs shape is: {}".format(sim_negs.size()))
            print("use all negative samples")
            relative_n_inds = None
            neg_sim = sim_negs

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds

    def find_hardest_support5(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        neg_sim = sim[is_neg]
        relative_n_inds = None

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds

    def find_hardest_support2(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        neg_sim = []
        relative_n_inds = []
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)
        # sim_negs = sim[is_neg]

        diff_labels = torch.unique(labels_s[is_neg])

        for diff_label in diff_labels:
            sim_negs = sim[torch.nonzero(labels_s == diff_label)]
            neg = torch.max(sim_negs.squeeze(-1).contiguous(), 0, keepdim=False)
            neg_sim.append(neg[0].unsqueeze(0))
            relative_n_inds.append(neg[1].unsqueeze(0))
        neg_sim = torch.cat(neg_sim, dim=0).cuda()
        relative_n_inds = torch.cat(relative_n_inds, dim=0).cuda()

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds


class CriterionLP_inter(nn.Module):
    """Label Preserving Loss in ISE"""

    def __init__(self, args):
        super(CriterionLP_inter, self).__init__()
        self.args = args

    def forward(self, feats, feats_s, labels, labels_s):
        B = feats.size(0)
        C = feats.size(-1)
        topk = self.args.inter_topk
        B_s = feats_s.size(0)
        C_s = feats_s.size(-1)

        feats_s = feats_s.reshape(B_s, C_s)
        # feats_s = feats_s.reshape(B * topk, C)
        sim_instance_support = feats.mm(feats_s.t()) / self.args.temp_lp_loss
        exp_instance_support = torch.exp(sim_instance_support)
        loss = 0
        for idx, (exp_i_s, lb) in enumerate(zip(exp_instance_support, labels)):
            pos_sim, pos_ind, neg_sim, neg_ind = self.find_hardest_support(exp_i_s, idx, lb, labels_s)
            loss += (-torch.log((pos_sim.sum() / (pos_sim.sum() + neg_sim.sum() + 1e-6)) + 1e-6))
        loss = loss / B

        return loss

    def find_hardest_support(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)

        sim_negs = sim[is_neg]
        try:
            is_neg = is_neg.reshape(-1, 1 * self.args.num_instances)
            sim_negs = sim_negs.reshape(-1, 1 * self.args.num_instances)
            neg_sim, relative_n_inds = torch.max(sim_negs.contiguous(), 1, keepdim=False)
        except:
            print("is_neg shape is: {}".format(is_neg.size()))
            print("sim_negs shape is: {}".format(sim_negs.size()))
            print("use all negative samples")
            relative_n_inds = None
            neg_sim = sim_negs

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds

    def find_hardest_support2(self, sim, idx, label, labels_s):
        is_pos = labels_s == label
        is_neg = ~is_pos
        neg_sim = []
        relative_n_inds = []
        pos_sim, relative_p_inds = torch.min(sim[is_pos].contiguous().view(1, -1), 1, keepdim=False)
        # sim_negs = sim[is_neg]

        diff_labels = torch.unique(labels_s[is_neg])

        for diff_label in diff_labels:
            sim_negs = sim[torch.nonzero(labels_s == diff_label)]
            neg = torch.max(sim_negs.squeeze(-1).contiguous(), 0, keepdim=False)
            neg_sim.append(neg[0].unsqueeze(0))
            relative_n_inds.append(neg[1].unsqueeze(0))
        neg_sim = torch.cat(neg_sim, dim=0).cuda()
        relative_n_inds = torch.cat(relative_n_inds, dim=0).cuda()

        return pos_sim, relative_p_inds, neg_sim, relative_n_inds
