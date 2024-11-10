from __future__ import absolute_import
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


class GraphLayer(nn.Module):
    """
    graph block with residual learning.
    """

    def __init__(self, in_features, out_features, learn_graph=True, use_pose=False,
                 dist_method='l2', gamma=0.1, k=4, **kwargs):
        """
        :param in_features: input feature size.
        :param out_features: output feature size.
        :param learn_graph: learn a affinity graph or not.
        :param use_pose: use graph from pose estimation or not.
        :param dist_method: calculate the similarity between the vertex.
        :param k: nearest neighbor size.
        :param kwargs:
        """
        super(GraphLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.learn_graph = learn_graph
        self.use_pose = use_pose
        self.dist_method = dist_method
        self.gamma = gamma

        assert use_pose or learn_graph
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.LeakyReLU(0.1)

        if self.learn_graph and dist_method == 'dot':
            num_hid = self.in_features // 8
            self.emb_q = nn.Linear(out_features, num_hid)
            self.emb_k = nn.Linear(out_features, num_hid)

        self._init_params()

    def get_sim_matrix(self, v_feats):
        """
        generate similarity matrix
        :param v_feats: (batch, num_vertex, num_hid)
        :return: sim_matrix: (batch, num_vertex, num_vertex)
        """
        if self.dist_method == 'dot':
            emb_q = self.emb_q(v_feats)
            emb_k = self.emb_k(v_feats)
            sim_matrix = torch.bmm(emb_q, emb_k.transpose(1, 2))
        elif self.dist_method == 'l2':
            # calculate the pairwise distance with exp(x) - 1 / exp(x) + 1
            distmat = torch.pow(v_feats, 2).sum(dim=2).unsqueeze(1) + \
                      torch.pow(v_feats, 2).sum(dim=2).unsqueeze(2)
            distmat -= 2 * torch.bmm(v_feats, v_feats.transpose(1, 2))
            distmat = distmat.clamp(1e-12).sqrt()  # numerical stability
            sim_matrix = 2 / (distmat.exp() + 1)
        else:
            raise NotImplementedError
        return sim_matrix

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    #def forward(self, input, adj):
    def forward(self, input):
        """
        :param input: (b, num_vertex, num_hid), where num_vertex = num_scale * seq_len * num_splits
        :param adj: (b, num_vertex, num_vertex), the pose-driven graph
        :return:
        """
        h = self.linear(input)
        N, V, C = h.size()

        # mask = torch.ones((N, V, V)).to(h.device)
        # for i in range(mask.size(1)):
        #     mask[:, i, i] = 0

        # if self.use_pose:
        #     # adj = mask * adj
        #     adj = F.normalize(adj, p=1, dim=2)

        if self.learn_graph:
            graph = self.get_sim_matrix(input)
            # graph = mask * graph
            graph = F.normalize(graph, p=1, dim=2)
            # if self.use_pose:
            #     graph = (adj + graph) / 2
        # else:
        #     graph = adj

        h_prime = torch.bmm(graph, h)
        h_prime = self.bn(h_prime.view(N * V, -1)).view(N, V, -1)
        h_prime = self.relu(h_prime)

        return (1 - self.gamma) * input + self.gamma * h_prime