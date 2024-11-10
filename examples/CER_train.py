# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import parser
import random
import numpy as np
import os
import sys
import collections
import time
from datetime import timedelta
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, adjusted_mutual_info_score, v_measure_score, \
    davies_bouldin_score

from bisect import bisect_right
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
sys.path.append(os.getcwd())
from CER import datasets
from CER import models
# from HCCL.models.cm import ClusterMemory
from CER.CER_trainers import HCCLTrainer
from CER.evaluation_metrics import Evaluator, extract_features
from CER.utils.data import IterLoader
from CER.utils.data import transforms as T
from CER.utils.data.sampler import RandomMultipleGallerySamplerProxy
from CER.utils.data.preprocessor import Preprocessor
from CER.utils.logging import Logger
from CER.utils.serialization import load_checkpoint, save_checkpoint
from CER.utils.faiss_rerank import compute_jaccard_distance
from sklearn.manifold import TSNE

start_epoch = best_mAP = 0


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3, warmup_iters=500,
                 warmup_method="linear", last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones, )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / float(self.warmup_iters)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def plot_with_labels(args, lowDWeights, labels, i, name, visualization_labels_num=None):
    plt.cla()
    plt.figure(dpi=300)
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    if visualization_labels_num is not None:
        for x, y, s in zip(X, Y, labels):
            c = cm.get_cmap('rainbow')(
                int(255 / visualization_labels_num * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
            plt.text(x, y, s, color=c, fontsize=7)
    else:
        for x, y, s in zip(X, Y, labels):
            c = cm.get_cmap('rainbow')(int(255 / 10 * s))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
            plt.text(x, y, s, color=c, fontsize=7)
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # plt.title('Visualize in {}'.format(i))
    dirs = osp.join(args.logs_dir,
                    'images_db{}_bs{}_ins{}_eps{}_intraLam{}_it{}_lam{}'.format(
                        args.dataset,
                        args.batch_size,
                        args.num_instances,
                        args.eps,
                        args.intra_lam,
                        args.iters,
                        args.lam))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    # plt.xticks(())  # 不显示坐标刻度
    # plt.yticks(())
    plt.savefig(osp.join(dirs, "{}_{}.jpg".format(name, i)))


def plot_with_labels_for_ab(args, lowDWeights, instance_labels, centroid_labels, i, name):
    plt.cla()
    plt.figure(dpi=300)
    X_ins, Y_ins = lowDWeights[:len(instance_labels), 0], lowDWeights[:len(instance_labels), 1]
    X_cen, Y_cen = lowDWeights[len(instance_labels):, 0], lowDWeights[len(instance_labels):, 1]
    # palette = np.array(sns.color_palette('hls', len(torch.unique(instance_labels))))
    for x, y, s in zip(X_ins, Y_ins, instance_labels):
        c = cm.get_cmap('gist_rainbow')(
            int(255 / len(torch.unique(instance_labels)) *
                np.where(torch.unique(instance_labels).cpu().numpy() == int(s)
                         )[0]))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区
        plt.plot(x, y, '.', c=c, markersize=2)
        # plt.plot(x, y, '.', c=palette[np.where(torch.unique(instance_labels).cpu().numpy() == s)[0]])

    for x, y, s in zip(X_cen, Y_cen, centroid_labels):
        c = cm.get_cmap('gist_rainbow')(
            int(255 / len(torch.unique(centroid_labels)) *
                np.where(torch.unique(centroid_labels).cpu().numpy() == int(s))[
                    0]))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
        plt.plot(x, y, '*', c=c, markersize=5)
    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    dirs = osp.join(args.logs_dir,
                    'images_db{}_bs{}_ins{}_eps{}_intraLam{}_it{}_lam{}'.format(
                        args.dataset,
                        args.batch_size,
                        args.num_instances,
                        args.eps,
                        args.intra_lam,
                        args.iters,
                        args.lam))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    plt.savefig(osp.join(dirs, "{}_{}.jpg".format(name, i)))


def plot_with_labels_ps_gt(args, lowDWeights, labels, pesudo_labels, i, name, visualization_labels_num=None):
    plt.cla()
    plt.figure(dpi=300)
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    if visualization_labels_num is not None:
        for x, y, s, pes in zip(X, Y, labels, pesudo_labels):
            c = cm.get_cmap('rainbow')(
                int(255 / visualization_labels_num * pes))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
            plt.text(x, y, s, color=c, fontsize=7)
    else:
        for x, y, s, pes in zip(X, Y, labels, pesudo_labels):
            c = cm.get_cmap('rainbow')(int(255 / 10 * pes))  # 为了使得颜色有区分度，把0-255颜色区间分为9分,然后把标签映射到一个区间
            plt.text(x, y, s, color=c, fontsize=7)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    # plt.title('Visualize in {}'.format(i))
    dirs = osp.join(args.logs_dir,
                    'images_db{}_bs{}_ins{}_eps{}_intraLam{}_it{}_lam{}_des{}'.format(
                        args.dataset,
                        args.batch_size,
                        args.num_instances,
                        args.eps,
                        args.intra_lam,
                        args.iters,
                        args.lam, args.des))
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    plt.savefig(osp.join(dirs, "{}_{}.jpg".format(name, i)))


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        # sampler = MoreCameraSampler(train_set, num_instances)  # RandomMultipleGallerySamplerProxy
        sampler = RandomMultipleGallerySamplerProxy(train_set, args.num_instances, 4)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    if 'resnet' in args.arch:
        model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                              num_classes=0, pooling_type=args.pooling_type)
    else:
        model = models.create(args.arch, img_size=(args.height, args.width), drop_path_rate=args.drop_path_rate
                              , pretrained_path=args.pretrained_path, hw_ratio=1, conv_stem=False)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def tsne_visualization(args, tsne, index, features, real_labels, real_cam_labels, epoch, name,
                       visualization_labels_num=None):
    select_features = features[index]
    select_ids = real_labels[index]
    select_camids = real_cam_labels[index]
    low_dim_embs = tsne.fit_transform(select_features)
    tene_labels = select_ids.numpy()
    tene_cam_labels = select_camids.numpy()
    plot_with_labels(args, low_dim_embs, tene_labels, "epoch{}".format(epoch), name, visualization_labels_num)
    plot_with_labels(args, low_dim_embs, tene_cam_labels, "epoch{}".format(epoch), 'cam' + name,
                     visualization_labels_num)


def tsne_visualization_gt_ps(args, tsne, index, features, real_labels, pseudo_labels, epoch, name,
                             visualization_labels_num=None):
    select_features = features[index]
    select_ids = real_labels
    select_pseudo_labels = pseudo_labels
    low_dim_embs = tsne.fit_transform(select_features)
    tene_labels = select_ids.numpy()
    tene_pseudo_labels = select_pseudo_labels.numpy()
    plot_with_labels_ps_gt(args, low_dim_embs, tene_labels, tene_pseudo_labels, "epoch{}".format(epoch), name,
                           visualization_labels_num)


def tsne_visualization_per(tsne, index, features, real_labels, epoch, name):
    select_features = features[index]
    select_camids = real_labels[index]
    low_dim_embs = tsne.fit_transform(select_features)
    tene_labels = select_camids.numpy()
    plot_with_labels(low_dim_embs, tene_labels, "epoch{}".format(epoch), name)


def compute_global_variance(inputs, mem, pseudo_labels, real_camids, uniquecams):
    pseudo_labels = torch.from_numpy(pseudo_labels)
    uniquelabels = torch.unique(pseudo_labels)
    mean_variance = 0.
    proxy_num = 0
    miss_num = 0
    for label in uniquelabels:
        for camid in uniquecams:
            indexs = [val for val in
                      torch.nonzero(pseudo_labels == label) if
                      val in torch.nonzero(real_camids == camid)]
            if len(indexs) > 0:
                features = inputs[torch.cat(indexs)]
                sim = torch.matmul(F.normalize(features).cuda(), F.normalize(mem.t().clone()))
                variance = np.array(sim.cpu()).std(axis=0)
                mean_variance += np.mean(variance)
                proxy_num += 1
            else:
                miss_num += 1
                continue
    mean_variance = mean_variance / proxy_num
    print(miss_num)
    return mean_variance


def sim_between_batch_pyototype(args, tsne, features, intra_id_features, intra_id_labels, pseudo_labels, real_camids,
                                concate_intra_class_list, epoch, uniquecams):
    pseudo_id_label = intra_id_labels[0][:10]
    inds_list = []
    centorid_label_list = []
    intra_id_features_list = []
    for cam in uniquecams:
        per_feat = [val.unsqueeze(-1) for val in
                    torch.from_numpy(np.where(np.isin(pseudo_labels, pseudo_id_label.cpu()))[0]) if
                    val in torch.nonzero(real_camids == cam)]
        if len(per_feat) > 0:
            inds_list.append(torch.cat(per_feat))
        centorid_label_list.append(
            intra_id_labels[cam][np.where(np.isin(intra_id_labels[cam], pseudo_id_label.cpu()))[0]])
        intra_id_features_list.append(
            intra_id_features[cam][np.where(np.isin(intra_id_labels[cam], pseudo_id_label.cpu()))[0]])

    inds = torch.cat(inds_list, dim=0)
    centroid_labels = torch.cat(centorid_label_list, dim=0)
    select_feat = features[inds]

    # intra_id_features = intra_id_features[cam_label]
    select_feat = torch.cat((select_feat, torch.cat(intra_id_features_list)), dim=0)
    low_dim_embs = tsne.fit_transform(select_feat)

    instance_labels = torch.from_numpy(pseudo_labels[inds]).cuda()
    # centroid_labels = centorid_label_list
    plot_with_labels_for_ab(args, low_dim_embs, instance_labels, centroid_labels, "epoch{}".format(epoch),
                            'visi_for_ab')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir,
                                 'HCCL_step_db{}_bs{}_ins{}_eps{}_intraLam{}_it{}_lam{}_des{}.txt'.format(
                                     args.dataset,
                                     args.batch_size,
                                     args.num_instances,
                                     args.eps,
                                     args.intra_lam,
                                     args.iters,
                                     args.lam,
                                     args.des)))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)

    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_choose = 'WarmUp'
    if lr_choose == 'WarmUp':
        lr_scheduler = WarmupMultiStepLR(optimizer, [20, 40], gamma=0.1, warmup_factor=0.01, warmup_iters=10)
        print('lr_sch is warmup')
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
        print('lr_sch is step')

    # params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    # optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    # Trainer
    trainer = HCCLTrainer(args, model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            random.shuffle(dataset.train)
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))
            features, real_labels, real_camids = extract_features(model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            real_labels = torch.cat([real_labels[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            real_camids = torch.cat([real_camids[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            visualization_labels_num = 20

            if args.is_supervised:
                num_cluster = len(torch.unique(real_labels))
            else:
                rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
                if epoch == 0:
                    # DBSCAN cluster
                    eps = args.eps
                    print('Clustering criterion: eps: {:.3f}'.format(eps))
                    cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

                # select & cluster images as training set of this epochs
                pseudo_labels = cluster.fit_predict(rerank_dist)
                index_pseudo_labels = np.where(pseudo_labels != -1)
                # index_gt_labels = torch.nonzero(real_labels < visualization_labels_num).squeeze(-1)

                select_pseudo_labels = torch.index_select(torch.from_numpy(pseudo_labels), dim=0,
                                                          index=torch.from_numpy(index_pseudo_labels[0]))
                select_real_labels = torch.index_select(real_labels, dim=0,
                                                        index=torch.from_numpy(index_pseudo_labels[0]))

                # 两种都显示处理，shown
                ARS = adjusted_rand_score(select_real_labels, select_pseudo_labels)
                print("ARS is {} at epoch {}".format(ARS, epoch))
                FMS = fowlkes_mallows_score(select_real_labels, select_pseudo_labels)
                print("FMS is {} at epoch {}".format(FMS, epoch))
                AMIS = adjusted_mutual_info_score(select_real_labels, select_pseudo_labels)
                print("AMIS is {} at epoch {}".format(AMIS, epoch))
                VMS = v_measure_score(select_real_labels, select_pseudo_labels)
                print("VMS is {} at epoch {}".format(VMS, epoch))
                num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
                del rerank_dist

        # generate new dataset and calculate cluster centers
        pseudo_labeled_dataset = []
        cams = []
        labels_true = []
        if args.is_supervised:
            labels = real_labels.numpy()
        else:
            labels = pseudo_labels
        for i, ((fname, pid, cid), label) in enumerate(
                zip(sorted(dataset.train), labels)):  # pseudo_labels & real_labels
            labels_true.append(pid)
            cams.append(cid)
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))
        cams = np.asarray(cams)

        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        # cluster_features = generate_cluster_features(labels, features)
        del cluster_loader

        # Create Per memory
        intra_id_features = []
        intra_id_labels = []
        intra_labels = []
        concate_intra_class_list = []
        memory_class_mapper = []
        concate_intra_cams = []
        all_mapper = []
        class_number = []
        uniquecams = np.unique(cams)
        init_number = 0
        for cc in uniquecams:
            percam_ind = np.where(cams == cc)[0]
            percam_feature = features[percam_ind].numpy()
            percam_label = labels[percam_ind]
            percam_class = np.unique(percam_label[percam_label >= 0])
            concate_intra_class_list.append(torch.tensor(percam_class).cuda())
            concate_intra_cams.append(cc * torch.ones(len(percam_class), dtype=int).cuda())
            percam_class_num = len(percam_class)
            percam_id_feature = np.zeros((percam_class_num, percam_feature.shape[1]), dtype=np.float32)
            percam_id_label = np.zeros((percam_class_num, 1), dtype=np.int8)
            cls_mapper = {int(percam_class[j]): j for j in range(len(percam_class))}
            memory_class_mapper.append(cls_mapper)
            for j in range(len(percam_class)):
                all_mapper.append(int(percam_class[j]))
            class_number.append(init_number)
            cnt = 0
            for lbl in np.unique(percam_label):
                if lbl >= 0:
                    ind = np.where(percam_label == lbl)[0]
                    id_feat = np.mean(percam_feature[ind], axis=0)
                    percam_id_feature[cnt, :] = id_feat
                    percam_id_label[cnt, :] = lbl
                    intra_id_labels.append(lbl)
                    cnt += 1
            percam_id_feature = percam_id_feature / np.linalg.norm(percam_id_feature, axis=1, keepdims=True)
            intra_id_features.append(torch.from_numpy(percam_id_feature))
            intra_labels.append(torch.from_numpy(percam_id_label))
            init_number += len(percam_class)
        concate_intra_cams = torch.cat(concate_intra_cams)

        # memory = ClusterMemory(args, model.module.num_features, num_cluster, temp=args.temp,
        #                        momentum=args.momentum, use_hard=args.use_hard).cuda()
        # cluster_features = F.normalize(cluster_features, dim=1).cuda()
        # memory.features = cluster_features
        # trainer.memory = memory

        if args.dataset == 'msmt17':
            uniquecams = uniquecams - 1
            cams = cams - 1
            concate_intra_cams = concate_intra_cams - 1

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(args, epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader),
                      intra_id_labels=intra_id_labels, intra_id_features=intra_id_features, cams=cams,
                      concate_intra_class_list=concate_intra_class_list, concate_intra_cams=concate_intra_cams,
                      uniquecams=uniquecams,
                      memory_class_mapper=memory_class_mapper)

        if epoch == args.epochs - 1:
            sim_between_batch_pyototype(args, tsne, features, intra_id_features, intra_labels, labels,
                                        real_camids,
                                        concate_intra_class_list, epoch,
                                        uniquecams)

        del features
        # if epoch > 0:
        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            if args.dataset == 'msmt171':
                is_best = True
                save_checkpoint(args, {
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth_{}.tar'.format(args.start_epoch)))
            else:
                mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
                is_best = (mAP > best_mAP)
                best_mAP = max(mAP, best_mAP)
                save_checkpoint(args, {
                    'state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': best_mAP,
                }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth_{}.tar'.format(args.start_epoch)))

                print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                      format(epoch, mAP, best_mAP, ' *' if is_best else ''))
        # print learning rate in each epoch
        print('epoch{}_learning_rate is {}'.format(epoch, lr_scheduler.get_last_lr()[0]))
        lr_scheduler.step()

    if args.dataset != 'msmt171':
        print('==> Test with the best model:')
        checkpoint = load_checkpoint(
            osp.join(args.logs_dir,
                     'model_best.pth_db{}_eps{}_bs{}_lam{}_ins{}_beta{}_iters{}_des{}_end.tar'.format(
                         args.dataset,
                         args.eps,
                         args.batch_size,
                         args.lam,
                         args.num_instances,
                         args.beta,
                         args.iters, args.des)))
        model.load_state_dict(checkpoint['state_dict'])
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print-freq', type=int, default=50)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")  # 0.07
    parser.add_argument('--lam', type=float, default=1.0)  # 0.6
    parser.add_argument('--intra-lam', type=float, default=0.1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/home/dsp/data/lym/')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/market_super_1'))
    parser.add_argument('--pooling-type', type=str, default='gem')  # avg
    parser.add_argument('--use-hard', action="store_true", default=True)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--beta', type=float, default=0.07)
    # Proxy-aware Consistency Contrastive Loss
    parser.add_argument('--neg-samp-intra-topk', type=int, default=20)
    parser.add_argument('--neg-samp-inter-topk', type=int, default=20)
    parser.add_argument('--neg-samp-cluster-topk', type=int, default=20)
    parser.add_argument('--is-supervised', action="store_true")
    #
    parser.add_argument('--intra_topk', type=int, default=1, help='find top-k nearest center')
    parser.add_argument('--inter_topk', type=int, default=5, help='find top-k nearest center')
    # label_preserving (LP) loss parameters
    parser.add_argument('--temp_lp_loss', type=float, default=0.6,
                        help="temperature for LP loss")
    parser.add_argument('--degree', type=float, default=0.5,
                        help="degree")
    parser.add_argument('--lp_loss_weight', type=float, default=0.1,
                        help="loss weight for LP loss")
    parser.add_argument('--des', action="store_true")
    main()
