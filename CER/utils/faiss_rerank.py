#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

import os, sys
import time
import numpy as np
from scipy.spatial.distance import cdist
import gc
import faiss

import torch
import torch.nn.functional as F

from .faiss_utils import search_index_pytorch, search_raw_array_pytorch, \
                            index_init_gpu, index_init_cpu


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]


def compute_jaccard_distance(target_features, k1=20, k2=6, print_flag=True, search_option=2, use_float16=False):
    end = time.time()
    if print_flag:
        print('Computing jaccard distance...')

    ngpus = faiss.get_num_gpus()
    N = target_features.size(0)
    mat_type = np.float16 if use_float16 else np.float32

    if (search_option==0):
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==1):
        # GPU + PyTorch CUDA Tensors (2)
        # cfg = faiss.GpuIndexFlatConfig()
        # cfg.useFloat16 = False
        # cfg.device = 0, 1, 2, 3
        # res = faiss.StandardGpuResources()
        # index = faiss.GpuIndexFlatL2(res, target_features.size(-1), cfg)
        # index.add(target_features.cpu().numpy())
        # _, initial_rank = search_index_pytorch(index, target_features, k1)
        # res.syncDefaultStreamCurrentDevice()
        # initial_rank = initial_rank.cpu().numpy()
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, target_features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif (search_option==2):
        # GPU
        index = index_init_gpu(ngpus, target_features.size(-1))
        index.add(target_features.cpu().numpy())
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)
    elif (search_option==3):
        # 节省内存占用
        res = faiss.StandardGpuResources()
        res.setTempMemory(4000 * 1024 * 1024)
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, target_features, target_features, k1)
        initial_rank = initial_rank.cpu().numpy()
    else:
        # CPU
        index = index_init_cpu(target_features.size(-1))
        xb = target_features.cpu().numpy()
        assert not index.is_trained
        index.train(xb)
        assert index.is_trained
        index.add(xb)
        _, initial_rank = index.search(target_features.cpu().numpy(), k1)


    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1/2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index)) > 2/3*len(candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)  ## element-wise unique
        dist = 2-2*torch.mm(target_features[i].unsqueeze(0).contiguous(), target_features[k_reciprocal_expansion_index].t())
        if use_float16:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
        else:
            V[i,k_reciprocal_expansion_index] = F.softmax(-dist, dim=1).view(-1).cpu().numpy()

    del nn_k1, nn_k1_half

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:,i] != 0)[0])  #len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        # temp_max = np.zeros((1,N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]]+np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
            # temp_max[0,indImages[j]] = temp_max[0,indImages[j]]+np.maximum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])

        jaccard_dist[i] = 1-temp_min/(2-temp_min)
        # jaccard_dist[i] = 1-temp_min/(temp_max+1e-6)

    del invIndex, V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Jaccard distance computing time cost: {}".format(time.time()-end))

    return jaccard_dist

def compute_jaccard_dist_cpu(target_features, k1=20, k2=6, print_flag=True,
                         lambda_value=0, source_features=None, use_gpu=False):
    end = time.time()
    N = target_features.size(0)
    if (use_gpu):
        # accelerate matrix distance computing
        target_features = target_features.cuda()
        if (source_features is not None):
            source_features = source_features.cuda()

    if ((lambda_value > 0) and (source_features is not None)):
        print('use source labels')
        M = source_features.size(0)
        sour_tar_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True).expand(N, M) + \
                        torch.pow(source_features, 2).sum(dim=1, keepdim=True).expand(M, N).t()
        sour_tar_dist.addmm_(1, -2, target_features, source_features.t())
        sour_tar_dist = 1 - torch.exp(-sour_tar_dist)
        sour_tar_dist = sour_tar_dist.cpu()
        source_dist_vec = sour_tar_dist.min(1)[0]
        del sour_tar_dist
        source_dist_vec /= source_dist_vec.max()
        source_dist = torch.zeros(N, N)
        for i in range(N):
            source_dist[i, :] = source_dist_vec + source_dist_vec[i]
        del source_dist_vec

    if print_flag:
        print('Computing original distance...')

    original_dist = torch.pow(target_features, 2).sum(dim=1, keepdim=True) * 2
    original_dist = original_dist.expand(N, N) - 2 * torch.mm(target_features, target_features.t())
    original_dist /= original_dist.max(0)[0]
    original_dist = original_dist.t()
    initial_rank = torch.argsort(original_dist, dim=-1)

    original_dist = original_dist.cpu()
    initial_rank = initial_rank.cpu()
    all_num = gallery_num = original_dist.size(0)

    del target_features
    if (source_features is not None):
        del source_features

    if print_flag:
        print('Computing Jaccard distance...')

    nn_k1 = []
    nn_k1_half = []
    for i in range(all_num):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = torch.zeros(all_num, all_num)
    for i in range(all_num):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if (len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index)):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)  ## element-wise unique
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

    if k2 != 1:
        k2_rank = initial_rank[:, :k2].clone().view(-1)
        V_qe = V[k2_rank]
        V_qe = V_qe.view(initial_rank.size(0), k2, -1).sum(1)
        V_qe /= k2
        V = V_qe
        del V_qe
    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.nonzero(V[:, i])[:, 0])  # len(invIndex)=all_num

    jaccard_dist = torch.zeros_like(original_dist)
    for i in range(all_num):
        temp_min = torch.zeros(1, gallery_num)
        indNonZero = torch.nonzero(V[i, :])[:, 0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.min(V[i, indNonZero[j]],
                                                                              V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)
    del invIndex

    del V

    pos_bool = (jaccard_dist < 0)
    jaccard_dist[pos_bool] = 0.0
    if print_flag:
        print("Time cost: {}".format(time.time() - end))

    if (lambda_value > 0):
        return jaccard_dist * (1 - lambda_value) + source_dist * lambda_value
    else:
        return jaccard_dist
