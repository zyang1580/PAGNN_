import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse
import numpy as np
import sys
sys.path.append("models/")
from mlp import MLP
import torch.utils


class adaptive_filed(nn.Module):
    def __init__(self, input_dim, number_layer, hidden_dim, max_hop=6):
        '''
        :param input_dim:
        :param number_layer: mlp layer
        :param hidden_dim:
        :param max_hop: the  maxmimum hop nodes can own
        '''
        super(adaptive_filed, self).__init__()
        self.mlp_layer = MLP(number_layer,input_dim,hidden_dim, 1)   #used for compute threshold which used for comput mask
        self.z = nn.Parameter(torch.FloatTensor([1]))
        self.idx = torch.tensor(np.arange(max_hop),dtype=torch.float32)
        self.R = 10
        self.alpha = 1
        self.max_hop = max_hop
    def forward(self,x):
        thresholds = self.mlp_layer(x)
        thresholds = F.sigmoid(thresholds) * self.max_hop
        np.save("threshold.npy",thresholds.data.numpy())
        np.save("input_x.npy",x.data.numpy())
        return self.compute_mask2(thresholds)

    def compute_mask(self,thresholds):
        N=thresholds.shape[0]
        idxs = self.idx.repeat(N,1)
        mask_init = (self.R + thresholds-idxs)/self.R
        zeros_ = torch.zeros_like(mask_init)
        iner_ = torch.where(mask_init>0,mask_init,zeros_)
        ones_ = torch.ones_like(iner_)
        outer_ =torch.where(iner_<1, iner_,ones_)
        return outer_

    def compute_mask2(self,thresholds):
        N = thresholds.shape[0]
        idxs = self.idx.repeat(N,1)
        idxs_trans = idxs -thresholds
        idxs_pow = torch.pow(idxs_trans, 2)
        mask_init = torch.exp( -idxs_pow/self.alpha)
        return mask_init
        torch.utils.d
class Att_mlp_sigmod(nn.Module):   #interaction attention
    def __init__(self, dims, sum_flag=1, multi_head=3, concat=False, inter=0,dire_sigmod=0):
        '''
        :param dims: input dims
        :param sum_flag: if use sum after attention
        :param multi_head: if sigmod attention it not be useful
        :param concat: not use
        :param inter: if use iteraction!!
        '''
        super(Att_mlp_sigmod, self).__init__()
        self.mutl_number = multi_head
        self.concat = concat
        self.sum_flag = sum_flag
        self.sigmod = True
        self.interaction = inter
        if inter == 0:
            print("not use interaction in attention layer!!!")
        else:
            print("use interaction!!")
            self.P = nn.Parameter(torch.FloatTensor(1,dims))
            nn.init.xavier_normal_(self.P.data, gain=nn.init.calculate_gain('relu'))
        self.mlp_layer = MLP(2,dims,64,1)

    def weight_forwaed(self,h):
        if self.interaction != 0:  #do interaction
            inter = torch.mul(h,self.P)
        else:                     # do interaction
            inter = h
        m = self.mlp_layer(inter)
        m = torch.sigmoid(m)
        #print(m.shape)
        return m


    def forward(self, graph_info, h):
        N = graph_info.shape[0]
        I = torch.eye(N).to(h.device)
        graph_info_1 = torch.spmm(graph_info.transpose(-1, -2), I).transpose(-1, -2)
        #zero_vec = -9e15 * torch.ones(graph_info.shape).cuda()
        h0 = h
        zero_vec = torch.zeros(graph_info.shape).to(h.device)
        e = self.weight_forwaed(h0)
        e_out = e
        e = e.transpose(-1, -2)
        e = e.repeat(N, 1)
        att = torch.where(graph_info_1 > 0, e, zero_vec)
        # att = F.softmax(att, dim=-1)
        h = torch.spmm(att, h0)
        h_out = h
        # if self.sum_flag == 1:
        #     num = torch.sum(graph_info_1, dim=-1)
        #     num = torch.diagflat(num)
        #     h = torch.matmul(num, h)
        # h_out = h
        return h_out,e_out

class Att_mlp_softmax(nn.Module):   #interaction attention
    def __init__(self, dims, sum_flag=1, multi_head=3, concat=False, inter=0,dire_sigmod=0):
        '''
        :param dims: input dims
        :param sum_flag: if use sum after attention
        :param multi_head: if sigmod attention it not be useful
        :param concat: not use
        :param inter: if use iteraction!!
        '''
        super(Att_mlp_softmax, self).__init__()
        self.mutl_number = multi_head
        self.concat = concat
        self.sum_flag = sum_flag
        self.sigmod = True
        self.interaction = inter
        if inter == 0:
            print("not use interaction in attention layer!!!")
        else:
            print("use interaction!!")
            self.P = nn.Parameter(torch.FloatTensor(1,dims))
            nn.init.xavier_normal_(self.P.data, gain=nn.init.calculate_gain('relu'))
        self.mlp_layer = MLP(2,dims,64,1)

    def weight_forwaed(self,h):
        if self.interaction != 0:  #do interaction
            inter = torch.mul(h,self.P)
        else:                     # do interaction
            inter = h
        m = self.mlp_layer(inter)
        #m = torch.sigmoid(m)
        #print(m.shape)
        return m


    def forward(self, graph_info, h):
        N = graph_info.shape[0]
        I = torch.eye(N).to(h.device)
        graph_info_1 = torch.spmm(graph_info.transpose(-1, -2), I).transpose(-1, -2)

        zero_vec = -9e15 * torch.ones(graph_info.shape).to(h.device)
        h0 = h
        #zero_vec = torch.zeros(graph_info.shape).cuda()
        e = self.weight_forwaed(h0)
        e = e.transpose(-1, -2)
        e = e.repeat(N, 1)
        att = torch.where(graph_info_1 > 0, e, zero_vec)
        att = F.softmax(att, dim=-1)
        h = torch.spmm(att, h0)

        if self.sum_flag == 1:
            num = torch.sum(graph_info_1, dim=-1)
            num = torch.diagflat(num)
            h = torch.matmul(num, h)
        h_out = h

        return h_out


class Att_dire(nn.Module):
    def __init__(self,dims,sum_flag=1,multi_head=3,concat=False,inter=0,dire_sigmod=0):
        super(Att_dire,self).__init__()
        self.mutl_number = multi_head
        self.concat = concat
        self.dire_sigmod = dire_sigmod
        self.sum_flag = sum_flag
        self.P = nn.Parameter(torch.FloatTensor(dims,multi_head))
        nn.init.xavier_normal_(self.P.data,gain=nn.init.calculate_gain('relu'))
        self.sum = True
    def forward(self,graph_info,h):
        N = graph_info.shape[0]
        I = torch.eye(N).to(h.device)
        graph_info_1 = torch.spmm(graph_info.transpose(-1, -2), I).transpose(-1, -2)
        zero_vec = -9e15 * torch.ones(graph_info.shape).to(h.device)
        #zero_vec = torch.zeros(graph_info.shape).cuda()
        h0 = h
        h_out = []
        for PN in range(self.mutl_number):
            P = self.P[:,PN]
            P = P.unsqueeze(-1)
            e = torch.matmul(h0,P)
            
            if self.dire_sigmod==1:  #采用sigmod 方式,don't do softmax
                zero_vec = torch.zeros(graph_info.shape).to(h.device)
                e = torch.sigmoid(e)
            else:                 # do softmax
                e = F.relu(e)
            e = e.transpose(-1, -2)
            e = e.repeat(N, 1)
            att = torch.where(graph_info_1 > 0, e, zero_vec)
            if self.dire_sigmod ==0:     #do softmax
                att = F.softmax(att,dim=-1)
            h = torch.spmm(att, h0)
            if self.sum_flag == 1:
                num = torch.sum(graph_info_1, dim=-1)
                num = torch.diagflat(num)
                h = torch.matmul(num, h)
            h_out.append(h)
        if self.mutl_number == 1:
            return h
        if not self.concat:
            h_out = torch.cat(h_out,dim=-1)
            h_out = h_out.view(N,self.mutl_number,-1)
            h_out = torch.sum(h_out,dim=-2)
        else:
            h_out = torch.cat(h_out, dim=-1)
        return h_out

class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device,attention=False,multi_head=3,sum_flag=1,inter=0,attention_type='mlp-sigmod',dire_sigmod=0):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()
        
        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))
        self.attention = attention
        self.mask = adaptive_filed(input_dim, 2, 54, max_hop=num_layers-1)  # used for compute mask for differet layer
        if attention:
            self.attention_layer = nn.ModuleList()
            if attention_type == 'mlp-sigmod':
                print("Atttention type: use sigmod attention!!!")
                AttModul = Att_mlp_sigmod
            elif attention_type == 'dire': # contain sum or not use sum_flag to flag
                AttModul = Att_dire
                print("Atttention type: use dire attention!!!")
            elif attention_type == 'mlp-softmax':                     #softmax
                AttModul = Att_mlp_softmax
                print("Atttention type: use softmax attention!!!")
            else:
                print("Don't have attention Model tyep:",attention_type)

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        if self.mask is not None:
            self.linears_prediction = nn.Linear(hidden_dim * (num_layers-1) , output_dim)
        else:
            self.linears_prediction = torch.nn.ModuleList()
            for layer in range(num_layers):
                if layer == 0:
                    self.linears_prediction.append(nn.Linear(input_dim, output_dim))
                    if attention:
                        self.attention_layer.append(AttModul(input_dim,multi_head=multi_head,sum_flag=sum_flag,inter=inter,dire_sigmod=dire_sigmod))
                else:
                    self.linears_prediction.append( nn.Linear(hidden_dim, output_dim) )
                    if attention:
                        self.attention_layer.append(AttModul(hidden_dim,multi_head=multi_head,sum_flag=sum_flag,inter=inter,dire_sigmod=dire_sigmod))




    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        '''
        to decide  wethear to reflect the depth of neighborhod
        '''
        #pooled = 2 * pooled + h   #change by @zhangyang
        pooled_rep = self.mlps[layer](pooled)      

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    def dotmask(self, inputs):
        pass



    def forward(self, batch_graph):
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]   # here is changed by @zhangyang

        if self.mask is not None:
            hidden_rep = []
        h = X_concat


        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)
        score_over_layer = 0
        if self.mask is not None:
            N = h.shape[0]
            hidden = torch.cat(hidden_rep,dim=1)
            hidden = hidden.reshape(N, self.num_layers-1,-1)# self.number_layers= max_hop
            field_mask = self.mask(X_concat).unsqueeze(-1)
            out_put = torch.mul(field_mask, hidden).reshape(N, -1)
            graph_representation = torch.matmul(graph_pool, out_put)
            pre = self.linears_prediction(graph_representation)
            return pre


        #perform pooling over all nodes in each graph in every layer
        for layer, h in enumerate(hidden_rep):
            if self.attention:
                pooled_h,e_out = self.attention_layer[layer](graph_pool,h)
            else:
                pooled_h = torch.spmm(graph_pool, h)
            score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)

        # if self.attention:
        #     adj_use = Adj_block
        #     D_adj =torch.sum(adj_use.to_dense(),dim=-1)
        #     #D_adj = D_adj.to_dense()
        #     N = D_adj.shape[0]
        #     NN = list(np.arange(N))
        #     pos = torch.LongTensor(NN).reshape(1,N)
        #     pos = pos.repeat(2,1).cuda()
        #     v = D_adj
        #     s_n  = torch.sum(D_adj, dim=-1)
        #
        #     D_adj = torch.sparse.FloatTensor(pos, v, torch.Size([N,N]))
        #
        #     #D_adj = torch.sum(adj_use,dim=-1)
        #     L_adj = D_adj - adj_use
        #     b_std = torch.spmm(L_adj,e_out)
        #     loss_att = torch.norm(b_std,2) / s_n
        #     if self.training:
        #         return score_over_layer,loss_att

        return score_over_layer