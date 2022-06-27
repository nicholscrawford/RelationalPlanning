### Acknowledgement 
"""
The general code base including the model is edited from the code of CoRL 2022 Planning for Multi Object Manipulation with Graph Neural Network Relational Classifiers
Which was edited from the CoRL 2020 Relational Learning for Skill Preconditions.
Thanks for Mohit Sharma for sharing this code base with Yixuan Huang,
and thanks to Yixuan Huang for sharing this code base with me.

Utility function for PointConv from : https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py by Wenxuan Wu from OSU.
Licenses about these two softwares in the liscense directory.
"""

import numpy as np
from collections import OrderedDict
from itertools import chain, permutations, product
from numbers import Number

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Data, Batch
from torch.utils.data import SubsetRandomSampler

from relational_precond.model.pointconv_util_groupnorm import PointConvDensitySetAbstraction




class PointConv(nn.Module):
    def __init__(self, normal_channel=False):
        super(PointConv, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        
        
        self.sa1 = PointConvDensitySetAbstraction(npoint=128, nsample=8, in_channel=6+additional_channel, mlp=[32], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=64, nsample=16, in_channel= 32 + 3, mlp=[64], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel= 64 + 3, mlp=[128], bandwidth = 0.4, group_all=True) # version 3-5

        
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)
        self.drop1 = nn.Dropout(0.5)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.drop3 = nn.Dropout(0.5)    

        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.drop4 = nn.Dropout(0.5)

        self.fc5 = nn.Linear(32, 3)


    def forward(self, xyz):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)

        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        
        
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 128)
        

        return x

class GNNModelOptionalEdge(MessagePassing):
    def __init__(self, 
                 in_channels, 
                 edge_inp_size,
                 node_output_size, 
                 relation_output_size, 
                 max_objects = 5, 
                 graph_output_emb_size=16, 
                 node_emb_size=32, 
                 edge_emb_size=32,
                 message_output_hidden_layer_size=128,  
                 message_output_size=128, 
                 node_output_hidden_layer_size=64,
                 edge_output_size=16,
                 use_latent_action = True,
                 latent_action_dim = 128, 
                 all_classifier = False,
                 predict_obj_masks=False,
                 predict_graph_output=False,
                 use_edge_embedding=False,
                 predict_edge_output=False,
                 use_edge_input=False,
                 node_embedding = False):
        self.relation_output_size = relation_output_size
        # define the relation_output_size by hand for all baselines. 
        # Make sure all the planning stuff keeps the same for all our comparison approaches. 
        super(GNNModelOptionalEdge, self).__init__(aggr='mean')
        # all edge output will be classifier
        self.all_classifier = all_classifier

        self.node_inp_size = in_channels
        # Predict if an object moved or not
        self._predict_obj_masks = predict_obj_masks
        # predict any graph level output
        self._predict_graph_output = predict_graph_output

        self.latent_action_dim = latent_action_dim
        self.use_latent_action = use_latent_action
        
        
        self.use_one_hot_embedding = True
        if self.use_one_hot_embedding: 
            self.one_hot_encoding_dim = 128

        
        total_objects = max_objects
        # print('max-objects', max_objects)
        action_dim = total_objects + 3
        if self.use_latent_action:
            self._in_channels = self.latent_action_dim
            self.action_emb = nn.Sequential(
                nn.Linear(action_dim, self.latent_action_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.latent_action_dim, self.latent_action_dim)
            )
        else:
            self._in_channels = action_dim

        if self.use_one_hot_embedding: 
            self.one_hot_encoding_embed = nn.Sequential(
                    nn.Linear(total_objects, self.one_hot_encoding_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.one_hot_encoding_dim, self.one_hot_encoding_dim)
                )

        
        

        self._use_edge_dynamics = True

        self.use_edge_input = use_edge_input
        if self.use_edge_input:
            self.use_one_hot_embedding = False
        
        if use_edge_input == False:
            edge_inp_size = 0
            use_edge_embedding = False
            self._use_edge_dynamics = False
        self._edge_inp_size = edge_inp_size

        self._node_emb_size = node_emb_size
        self.node_embedding = node_embedding
        if self.node_embedding:
            self.node_emb = nn.Sequential(
                nn.Linear(in_channels, self._node_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(self._node_emb_size, self._node_emb_size)
            )
        if not self.node_embedding:
            if self.use_one_hot_embedding:
                self.node_inp_size += self.one_hot_encoding_dim
                self._node_emb_size = self.node_inp_size
                
            else:
                self._node_emb_size = self.node_inp_size

        self.edge_emb_size = edge_emb_size
        self._use_edge_embedding = use_edge_embedding
        self._test_edge_embedding = False
        if use_edge_embedding:
            self.edge_emb = nn.Sequential(
                nn.Linear(edge_inp_size, edge_emb_size),
                nn.ReLU(inplace=True),
                nn.Linear(edge_emb_size, edge_emb_size)
            )

        self._message_layer_size = message_output_hidden_layer_size
        self._message_output_size = message_output_size
        #print('node input size', self.node_inp_size)
        if self.node_embedding:
            message_inp_size = 2*self._node_emb_size + edge_emb_size if use_edge_embedding else \
                2 * self._node_emb_size + edge_inp_size
        else:
            message_inp_size = 2*self.node_inp_size + edge_emb_size if use_edge_embedding else \
                2 * self.node_inp_size + edge_inp_size
        # if use_edge_input == False:
        #     message_inp_size = 2 * self._node_emb_size
        self.message_info_mlp = nn.Sequential(
            nn.Linear(message_inp_size, self._message_layer_size),
            nn.ReLU(),
            # nn.Linear(self._message_layer_size, self._message_layer_size),
            # nn.ReLU(),
            nn.Linear(self._message_layer_size, self._message_output_size)
            )

        self._node_output_layer_size = node_output_hidden_layer_size
        self._per_node_output_size = node_output_size
        graph_output_emb_size = 0
        self._per_node_graph_output_size = graph_output_emb_size
        self.node_output_mlp = nn.Sequential(
            nn.Linear(self._node_emb_size + self._message_output_size, self._node_output_layer_size),
            nn.ReLU(),
            nn.Linear(self._node_output_layer_size, node_output_size + graph_output_emb_size)
        )

        action_dim = self._in_channels
        self.action_dim = action_dim
        self.dynamics =  nn.Sequential(
            nn.Linear(self._in_channels+action_dim, 128),  # larger value
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self._in_channels)
        )

        if self._use_edge_dynamics:
            self.edge_dynamics =  nn.Sequential(
                nn.Linear(self._edge_inp_size+action_dim, 128),  # larger value
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, self._edge_inp_size)
            )

        
        self.graph_dynamics = nn.Sequential(
            nn.Linear(node_output_size+action_dim, 512),  # larger value
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, node_output_size)
        )

        self.graph_edge_dynamics = nn.Sequential(
            nn.Linear(edge_output_size+action_dim, 512),  # larger value
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, edge_output_size)
        )

        if self._predict_graph_output:
            self._graph_pred_mlp = nn.Sequential(
                nn.Linear(graph_output_emb_size, 32),
                nn.ReLU(),
                nn.Linear(32, 3),
            )
        
        self._should_predict_edge_output = predict_edge_output
        if predict_edge_output:
            self._edge_output_size = edge_output_size
            # TODO: Add edge attributes as well, should be easy
            if True:
                self._edge_output_mlp = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, edge_output_size)
                )
                self._edge_output_sigmoid = nn.Sequential(
                    nn.Linear(edge_inp_size + 2 * self._node_emb_size + 2 * self._message_output_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.relation_output_size),
                    nn.Sigmoid()
                )
            self._pred_edge_output = None


    def forward(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        #print(x.shape)
        # print(self.node_emb)
        self._test_edge_embedding = False
        if self.use_latent_action:
            # print(action.shape)
            # print(self.action_emb)
            action = self.action_emb(action)
        if self.node_embedding:
            x = self.node_emb(x)

        # Begin the message passing scheme
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        graph_node_action = torch.cat((state_pred_out, action), axis = 1)
        pred_node_embedding = self.graph_dynamics(graph_node_action)

        #edge_action = torch.stack([action[0][:], action[0][:], action[0][:], action[0][:], action[0][:], action[0][:]])
        edge_num = self._pred_edge_output.shape[0]
        edge_action_list = []
        for _ in range(edge_num):
            edge_action_list.append(action[0][:])
        edge_action = torch.stack(edge_action_list)
        graph_edge_node_action = torch.cat((self._pred_edge_output, edge_action), axis = 1)
        pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        return_dict = {'pred': state_pred_out,
        'current_embed': state_pred_out, 'pred_embedding':pred_node_embedding, 'edge_embed': self._pred_edge_output, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        
        return return_dict

    def forward_decoder(self, x, edge_index, edge_attr, batch, action):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # edge_x has shape [E, edge_features]

        # Get node embeddings for input features
        # print(x.shape)
        # print(self.node_emb)
        #x = self.node_emb(x)
        

        # Begin the message passing scheme
        self._test_edge_embedding = True
        total_out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        #print(total_out)

        # Get outputs for every ndoe vs overall graph
        node_out_index = torch.arange(self._per_node_output_size).to(x.device)
        graph_out_index = torch.arange(
            self._per_node_output_size, 
            self._per_node_output_size+self._per_node_graph_output_size).to(x.device)

        # Get node level outputs, that is [0..node_out_index-1] values from total_out
        out = torch.index_select(total_out, dim=1, index=node_out_index)
        #import pdb; pdb.set_trace()
        if self._predict_obj_masks:
            mask_index = [out.size(1) - 1]
            state_pred_index = [i for i in range(out.size(1)-1)]

            state_pred_out = torch.index_select(out, 1, torch.LongTensor(state_pred_index).to(x.device))
            mask_out = torch.index_select(out, 1, torch.LongTensor(mask_index).to(x.device))[:, 0]
        else:
            state_pred_out = out
            mask_out = None

        # Get graph level outputs, i.e., [node_out_index, end] values from total_out
        if self._predict_graph_output:
            graph_out = torch.index_select(total_out, dim=1, index=graph_out_index)
            graph_out = global_add_pool(graph_out, batch)
            graph_preds = self._graph_pred_mlp(graph_out)
        else:
            graph_preds = None

        state_action = torch.cat((state_pred_out, action), axis = 1)
        #print(state_action.shape)
        pred_state = self.dynamics(state_action)

        edge_action = torch.zeros((self._pred_edge_output.shape[0], self._pred_edge_output.shape[1] + self.action_dim))
        edge_action[:,:self._pred_edge_output.shape[1]] = self._pred_edge_output
        edge_action[:,self._pred_edge_output.shape[1]:] = action[0]
        edge_action = edge_action.to(x.device)
        
        
        if self._use_edge_dynamics:
            dynamics_edge = self.edge_dynamics(edge_action)

        graph_node_action = torch.cat((x, action), axis = 1)
        pred_node_embedding = self.graph_dynamics(graph_node_action)

        
        edge_num = self._edge_inp.shape[0]
        edge_action_list = []
        for _ in range(edge_num):
            edge_action_list.append(action[0][:])
        edge_action = torch.stack(edge_action_list)
        graph_edge_node_action = torch.cat((self._edge_inp, edge_action), axis = 1)
        pred_graph_edge_embedding = self.graph_edge_dynamics(graph_edge_node_action)
        return_dict = {'pred': state_pred_out, 'object_mask': mask_out, 'graph_pred': graph_preds, 'pred_state': pred_state, 
        'current_embed': x, 'pred_embedding':pred_node_embedding, 'edge_embed': self._edge_inp, 'pred_edge_embed': pred_graph_edge_embedding}
        if self._should_predict_edge_output:
            return_dict['pred_edge'] = self._pred_edge_output
            return_dict['pred_sigmoid'] = self._pred_edge_output_sigmoid
        if self.all_classifier:
            return_dict['pred_edge_classifier'] = self._pred_edge_classifier
        if self._use_edge_dynamics:
            return_dict['dynamics_edge'] = dynamics_edge

        return return_dict

    
    def message(self, x_i, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_attr is the edge attribute between x_i and x_j

        # x_i is the central node that aggregates information
        # x_j is the neighboring node that passes on information.

        # Concatenate features for sender node (x_j) and receiver x_i and get the message from them
        # Maybe there is a better way to get this message information?

        if self._test_edge_embedding:
            edge_inp = edge_attr
        else:
            if self._use_edge_embedding:
                assert self.edge_emb is not None, "Edge embedding model cannot be none"
                # print(edge_attr.shape)
                # print(self.edge_emb)
                edge_inp = self.edge_emb(edge_attr)
            else:
                edge_inp = edge_attr
        self._edge_inp = edge_inp
        #print('edge in GNN', self._edge_inp)

        #print(edge_inp.shape)
        if self.use_edge_input:
            x_ij = torch.cat([x_i, x_j, edge_inp], dim=1)
            
            out = self.message_info_mlp(x_ij)
        else:
            x_ij = torch.cat([x_i, x_j], dim=1)
            
            out = self.message_info_mlp(x_ij)
        
        return out

    def update(self, x_ij_aggr, x, edge_index, edge_attr):
        # We can transform the node embedding, or use the transformed embedding directly as well.
        inp = torch.cat([x, x_ij_aggr], dim=1)
        if self._should_predict_edge_output:
            source_node_idxs, target_node_idxs = edge_index[0, :], edge_index[1, :]
            if self.use_edge_input:
                edge_inp = torch.cat([
                    self._edge_inp,
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            else:
                edge_inp = torch.cat([
                    x[source_node_idxs], x[target_node_idxs],
                    x_ij_aggr[source_node_idxs], x_ij_aggr[target_node_idxs]], dim=1)
            
            self._pred_edge_output = self._edge_output_mlp(edge_inp)
            self._pred_edge_output_sigmoid = self._edge_output_sigmoid(edge_inp)
            if self.all_classifier:
                self._pred_edge_classifier = []
                for pred_classifier in self.all_classifier_list:
                    pred_classifier = pred_classifier.to(x.device)
                    self._pred_edge_classifier.append(F.softmax(pred_classifier(edge_inp), dim = 1))

        return self.node_output_mlp(inp)

    def edge_decoder_result(self):
        if self._should_predict_edge_output:
            return self._pred_edge_output
        else:
            return None

def create_graph(num_nodes, node_inp_size, node_pose, edge_size, edge_feature, action):
    nodes = list(range(num_nodes))
    # Create a completely connected graph
    edges = list(permutations(nodes, 2))
    edge_index = torch.LongTensor(np.array(edges).T)
    x = node_pose#torch.zeros((num_nodes, node_inp_size))#torch.eye(node_inp_size).float()
    edge_attr = edge_feature #torch.rand(len(edges), edge_size)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, action = action)
    # Recreate x as target
    data.y = x
    return data


## the initial setup for the input graph, latent graph and output graph. 
def main(): 

    node_emb_size, edge_emb_size = 128, 128

    node_inp_size = 128
    
    edge_inp_size = 128

    node_output_size = 128

    edge_output_size = 128

    max_objects = 8

    z_dim = 7


    classif_model = GNNModelOptionalEdge(
            node_inp_size, 
            edge_inp_size,
            relation_output_size = z_dim, 
            node_output_size = node_emb_size, 
            predict_edge_output = True,
            edge_output_size = edge_emb_size,
            graph_output_emb_size=16, 
            node_emb_size=node_emb_size, 
            edge_emb_size=edge_emb_size,
            message_output_hidden_layer_size=128,  
            message_output_size=128, 
            node_output_hidden_layer_size=64,
            predict_obj_masks=False,
            predict_graph_output=False,
            use_edge_embedding = False,
            use_edge_input = False, 
            max_objects = max_objects
        )
    classif_model_decoder = GNNModelOptionalEdge(
            node_emb_size, 
            edge_emb_size,
            relation_output_size = z_dim, 
            node_output_size = node_output_size, 
            predict_edge_output = True,
            edge_output_size = edge_output_size,
            graph_output_emb_size=16, 
            node_emb_size=node_emb_size, 
            edge_emb_size=edge_emb_size,
            message_output_hidden_layer_size=128,  
            message_output_size=128, 
            node_output_hidden_layer_size=64,
            predict_obj_masks=False,
            predict_graph_output=False,
            use_edge_embedding = False,
            use_edge_input = True, 
            max_objects = max_objects
        )
                    

    emb_model = PointConv(normal_channel=False)

    num_nodes = 3

    ## initial point cloud
    input_pointcloud = torch.rand(num_nodes,3,128) 

    ## get the pointconv features for each object which forms the node embedding for the input graph.
    img_emb_single = emb_model(input_pointcloud) 

    ##Encodes object identity
    one_hot_encoding = np.zeros((num_nodes, max_objects))
    one_hot_encoding[0][0] = 1
    one_hot_encoding[1][1] = 1
    one_hot_encoding[2][2] = 1  ## one hot encoding 
    one_hot_encoding_tensor = torch.Tensor(one_hot_encoding)
    latent_one_hot_encoding = classif_model.one_hot_encoding_embed(one_hot_encoding_tensor)
    node_pose = torch.cat([img_emb_single, latent_one_hot_encoding], dim = 1)

    ## the action for this step. 
    action_torch = torch.rand((num_nodes,max_objects + 3)) 

    #Create graph, stored as https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
    data = create_graph(num_nodes, node_inp_size, node_pose, 0, None, action_torch)

    #Creates a batch object, which is a set of regular graphs treated as a disconnected graph.
    batch = Batch.from_data_list([data])

    #Returns a dict of presumably the set of predicted latent space nodes and edges
    #and the set of predicted latent space nodes and edges given an action.
    outs = classif_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.action)
    
    
    data_1_decoder = create_graph(num_nodes, node_emb_size, outs['pred'], edge_emb_size, outs['pred_edge'], action_torch)
    batch_decoder = Batch.from_data_list([data_1_decoder])
    outs_decoder = classif_model_decoder(batch_decoder.x, batch_decoder.edge_index, batch_decoder.edge_attr, batch_decoder.batch, batch_decoder.action)

    data_2_decoder_edge = create_graph(num_nodes, node_emb_size, outs['pred_embedding'], edge_emb_size, outs['pred_edge_embed'], action_torch)
    batch_decoder_2_edge = Batch.from_data_list([data_2_decoder_edge])
    outs_decoder_2_edge = classif_model_decoder(batch_decoder_2_edge.x, batch_decoder_2_edge.edge_index, batch_decoder_2_edge.edge_attr, batch_decoder_2_edge.batch, batch_decoder_2_edge.action)

    print(outs_decoder['pred_sigmoid'][:].shape)  ## the predicted relations for current step
    print(outs_decoder_2_edge['pred_sigmoid'][:].shape) ## the predicted relations for the next step. 
    print(outs['pred_embedding'].shape)  ## the current latent graph node embedding
    print(outs['pred_edge_embed'].shape) ## the current latne graph edge embedding




if __name__ == '__main__':
    main()
