from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def setToArray(setInput, dtype='<U40'):
    arrayOutput = np.zeros((len(setInput)), dtype=dtype)
    index = 0
    for every in setInput:
        arrayOutput[index] = every
        index += 1
    return arrayOutput


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []
        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])  # 类里选一张图

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]  # 选一个人
            cams = self.pid_cam[pid_i]  # 选这个人的摄像机索引
            index = self.pid_index[pid_i]  # 选这个人的索引
            select_cams = No_index(cams, i_cam)  #

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if not select_indexes:
                    continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return iter(ret)


# class RandomMultipleGallerySampler_2(Sampler):
#     def __init__(self, data_source, batch_size, num_instances=4, num_cameras=4):
#         super().__init__(data_source)
#         self.data_source = data_source
#         self.index_pid = defaultdict(int)
#         self.pid_cam = defaultdict(list)
#         self.pid_index = defaultdict(list)
#         self.num_instances = num_instances
#         self.batch_size = batch_size
#         self.num_cameras = num_cameras
#
#         for index, (_, pid, cam) in enumerate(data_source):
#             if pid < 0:
#                 continue
#             self.index_pid[index] = pid
#             self.pid_cam[pid].append(cam)
#             self.pid_index[pid].append(index)
#
#         self.pids = list(self.pid_index.keys())
#         self.num_samples = len(self.pids)
#
#     def __len__(self):
#         return self.num_samples * self.num_instances
#
#     def get_unique_numbers(self, numbers):
#         list_of_unique_numbers = []
#         unique_numbers = set(numbers)
#
#         for number in unique_numbers:
#             list_of_unique_numbers.append(number)
#
#         return list_of_unique_numbers
#
#     def __iter__(self):
#         indices = torch.randperm(len(self.pids)).tolist()
#         ret = []
#         for kid in indices:
#             i = random.choice(self.pid_index[self.pids[kid]])  # 类里选一张图
#
#             _, i_pid, i_cam = self.data_source[i]
#
#             ret.append(i)
#
#             pid_i = self.index_pid[i]  # 选一个人
#             cams = self.pid_cam[pid_i]  # 选这个人的摄像机索引
#             index = self.pid_index[pid_i]  # 选这个人的索引
#             unique_cams = self.get_unique_numbers(cams)
#             # select_cams = No_index(cams, i_cam)  #
#
#             if len(unique_cams) < self.num_cameras:
#                 perid_cams_num = len(unique_cams)
#                 perid_percam_pic_num = self.num_instances / perid_cams_num
#
#
#
#
#             else:
#                 # perid_cams_num = self.num_cameras
#                 perid_percam_pic_num = self.num_instances / self.num_cameras
#
#
#
#             if len(cams) > self.num_cameras:
#                 select_cam_id = np.random.choice(cams, size=self.num_cameras, replace=False)
#             else:
#                 select_cam_id = cams
#
#             per_cam_select_nums = self.num_instances / len(select_cam_id)
#
#             # for cam_id in select_cam_id:
#
#
#
#
#             if select_cams:
#                 if len(select_cams) >= self.num_instances:
#                     cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=False)
#                 else:
#                     cam_indexes = np.random.choice(select_cams, size=self.num_instances - 1, replace=True)
#
#                 for kk in cam_indexes:
#                     ret.append(index[kk])
#
#             else:
#                 select_indexes = No_index(index, i)
#                 if not select_indexes:
#                     continue
#                 if len(select_indexes) >= self.num_instances:
#                     ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=False)
#                 else:
#                     ind_indexes = np.random.choice(select_indexes, size=self.num_instances - 1, replace=True)
#
#                 for kk in ind_indexes:
#                     ret.append(index[kk])
#
#         return iter(ret)


class RandomMultipleGallerySamplerProxy(Sampler):
    def __init__(self, data_source, num_instances=4, select_cam_num=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.cid_index = defaultdict(list)
        self.num_instances = num_instances
        # self.num_proxy = num_proxy

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)
            self.cid_index[cam].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)
        self.select_cam_num = select_cam_num
        self.proxy_num = int(self.num_instances / self.select_cam_num)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []
        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]

            cams = self.pid_cam[i_pid]
            index = self.pid_index[i_pid]

            unique_cams = set(cams)
            cams = np.array(cams)
            index = np.array(index)

            if len(unique_cams) >= self.select_cam_num:
                select_cam = np.random.choice(setToArray(unique_cams), size=self.select_cam_num, replace=False)
            else:
                select_cam = np.random.choice(setToArray(unique_cams), size=self.select_cam_num, replace=True)
            select_indexes = []
            for cam in select_cam:
                cam = int(cam)
                if len(index[cams == cam]) >= self.proxy_num:
                    select_indexes.append(np.random.choice(index[cams == cam], size=self.proxy_num, replace=False))
                else:
                    select_indexes.append(np.random.choice(index[cams == cam], size=self.proxy_num, replace=True))
            select_indexes = np.concatenate(select_indexes)
            ret.extend(select_indexes)
        return iter(ret)


class RandomMultipleGallerySamplerProxy1(Sampler):
    def __init__(self, data_source, num_instances=4):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.cid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)
            self.cid_index[cam].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []
        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]
            index = self.pid_index[i_pid]
            c_index = self.cid_index[i_cam]
            common_index = list(set(index).intersection(set(c_index)))
            if len(common_index) >= self.num_instances:
                cam_indexes = np.random.choice(common_index, size=self.num_instances, replace=False)
            else:
                cam_indexes = np.random.choice(common_index, size=self.num_instances, replace=True)
            ret.extend(cam_indexes)
        return iter(ret)


class MoreCameraSampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0: continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]

            cams = self.pid_cam[i_pid]
            index = self.pid_index[i_pid]

            unique_cams = set(cams)
            cams = np.array(cams)
            index = np.array(index)
            select_indexes = []
            for cam in unique_cams:
                select_indexes.append(np.random.choice(index[cams == cam], size=1, replace=False))  # 每个人每个摄像头选一张
            select_indexes = np.concatenate(select_indexes)
            if len(select_indexes) < self.num_instances:
                diff_indexes = np.setdiff1d(index, select_indexes)  # 从不同的index里面挑选
                if len(diff_indexes) == 0:
                    select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=True)
                elif len(diff_indexes) >= (self.num_instances - len(select_indexes)):
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances - len(select_indexes)),
                                                    replace=False)
                else:
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances - len(select_indexes)),
                                                    replace=True)
                select_indexes = np.concatenate([select_indexes, diff_indexes])
            else:
                select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=False)
            ret.extend(select_indexes)
        return iter(ret)


class MoreCameraSampler1(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid < 0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])
            _, i_pid, i_cam = self.data_source[i]

            cams = self.pid_cam[i_pid]
            index = self.pid_index[i_pid]

            unique_cams = set(cams)

            cams = np.array(cams)
            index = np.array(index)
            select_indexes = []
            for cam in unique_cams:
                select_indexes.append(np.random.choice(index[cams == cam], size=1, replace=False))  # 每个人每个摄像头选一张
            select_indexes = np.concatenate(select_indexes)
            if len(select_indexes) < self.num_instances:
                diff_indexes = np.setdiff1d(index, select_indexes)  # 从不同的index里面挑选
                if len(diff_indexes) == 0:
                    select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=True)
                elif len(diff_indexes) >= (self.num_instances - len(select_indexes)):
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances - len(select_indexes)),
                                                    replace=False)
                else:
                    diff_indexes = np.random.choice(diff_indexes, size=(self.num_instances - len(select_indexes)),
                                                    replace=True)
                select_indexes = np.concatenate([select_indexes, diff_indexes])
            else:
                select_indexes = np.random.choice(select_indexes, size=self.num_instances, replace=False)
            ret.extend(select_indexes)
        return iter(ret)


class RandomIdentitySamplerCamera(Sampler):
    def __init__(self, data_source, num_instances=4, batchsize=64):
        super().__init__(data_source)
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.index_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.cam_index = defaultdict(list)
        self.cam_pid = defaultdict(list)
        self.num_instances = num_instances
        self.batchsize = batchsize
        for index, (_, pid, cam) in enumerate(data_source):
            if pid < 0:
                continue
            self.index_pid[index] = pid
            self.index_cam[index] = cam
            self.pid_cam[pid].append(cam)
            self.cam_pid[cam].append(pid)
            self.pid_index[pid].append(index)
            self.cam_index[cam].append(index)

        self.pids = list(self.pid_index.keys())
        self.cams = list(self.cam_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.cams)).tolist()
        ret = []

        for cid in indices:
            i = random.choice(self.cam_index[self.cams[cid]])
            _, i_pid, i_cam = self.data_source[i]

            ids = self.cam_pid[i_cam]
            # index = self.pid_index[i_pid]
            index = self.cam_index[i_cam]

            unique_ids = np.unique(ids)
            ids = np.array(ids)
            index = np.array(index)
            select_indexes = []
            random_index = np.random.choice(len(unique_ids), size=int(self.batchsize / self.num_instances),
                                            replace=True)
            select_ids = unique_ids[random_index]

            for id in select_ids:
                select_indexes.append(np.random.choice(index[ids == id], size=len(index[ids == id]), replace=False))
            select_indexes = np.concatenate(select_indexes)
            if len(select_indexes) >= self.batchsize:
                select_indexes = np.random.choice(select_indexes, size=self.batchsize, replace=False)
            else:
                select_indexes = np.random.choice(select_indexes, size=self.batchsize, replace=True)
            # if len(select_indexes) < self.batchsize:
            #     diff_indexes = np.setdiff1d(index, select_indexes)
            #     if len(diff_indexes) == 0:
            #         select_indexes = np.random.choice(select_indexes, size=self.batchsize, replace=True)
            #     elif len(diff_indexes) >= (self.num_instances - len(select_indexes)):
            #         diff_indexes = np.random.choice(diff_indexes, size=(self.batchsize - len(select_indexes)),
            #                                         replace=False)
            #     else:
            #         diff_indexes = np.random.choice(diff_indexes, size=(self.batchsize - len(select_indexes)),
            #                                         replace=True)
            #     select_indexes = np.concatenate([select_indexes, diff_indexes])
            # else:
            #     select_indexes = np.random.choice(select_indexes, size=self.batchsize, replace=False)
            ret.extend(select_indexes)
        return iter(ret)


class ClassUniformlySampler(Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, samples, class_position, k, has_outlier=False, cam_num=0):

        self.samples = samples
        self.class_position = class_position
        self.k = k
        self.has_outlier = has_outlier
        self.cam_num = cam_num
        self.class_dict = self._tuple2dict(self.samples)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (image_path_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        id_dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]  # from which index to obtain the label
            if class_index not in list(id_dict.keys()):
                id_dict[class_index] = [index]
            else:
                id_dict[class_index].append(index)
        return id_dict

    def _generate_list(self, id_dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''
        sample_list = []

        dict_copy = id_dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        outlier_cnt = 0
        for key in keys:
            value = dict_copy[key]
            if self.has_outlier and len(value) <= self.cam_num:
                random.shuffle(value)
                sample_list.append(value[0])  # sample outlier only one time
                outlier_cnt += 1
            elif len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k  # copy a person's image list for k-times
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
        if outlier_cnt > 0:
            print('in Sampler: outlier number= {}'.format(outlier_cnt))
        return sample_list
