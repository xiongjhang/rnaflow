import torch
import torch.nn as nn
from torch.nn.modules.distance import CosineSimilarity
from models.modules.mlp import MLP
import models.modules.basic_gnn_with_edge as basic_gnn_with_edge


class Net_new_new(nn.Module):
    def __init__(self,
                 hand_NodeEncoder_dic={},
                 learned_NodeEncoder_dic={},
                 intialize_EdgeEncoder_dic={},
                 message_passing={},
                 edge_classifier_dic={}
                 ):
        super(Net_new_new, self).__init__()

        self.distance = CosineSimilarity()
        self.handcrafted_node_embedding = MLP(**hand_NodeEncoder_dic)
        self.learned_node_embedding = MLP(**learned_NodeEncoder_dic)
        self.learned_edge_embedding = MLP(**intialize_EdgeEncoder_dic)
        basic_gnn_class = getattr(basic_gnn_with_edge, message_passing.target)
        self.message_passing = basic_gnn_class(**message_passing.kwargs)
        self.edge_classifier = MLP(**edge_classifier_dic)

    def forward(self, x, edge_index, edge_feat):
        x1, x2 = x
        x_init = torch.cat((x1, x2), dim=-1)
        src, trg = edge_index
        similarity1 = self.distance(x_init[src], x_init[trg])
        abs_init = torch.abs(x_init[src] - x_init[trg])
        x1 = self.handcrafted_node_embedding(x1)
        x2 = self.learned_node_embedding(x2)
        x = torch.cat((x1, x2), dim=-1)  # cat of the 2 types of features
        src, trg = edge_index
        similarity2 = self.distance(x[src], x[trg])
        edge_feat_in = torch.cat((abs_init, similarity1[:, None], x[src], x[trg], torch.abs(x[src] - x[trg]), similarity2[:, None]), dim=-1)
        edge_init_features = self.learned_edge_embedding(edge_feat_in)
        x = self.message_passing(x, edge_index, edge_init_features)
        x = self.edge_classifier(x).squeeze()
        return x









