import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGPooling, global_mean_pool , global_max_pool 
import torch_scatter


class AttentionBlock(nn.Module):
    def __init__(self,time_step,dim):
        super(AttentionBlock, self).__init__()
        self.attention_matrix = nn.Linear(time_step, time_step)

    def forward(self, inputs):
        inputs_t = torch.transpose(inputs,2,1) # (batch_size, input_dim, time_step)
        attention_weight = self.attention_matrix(inputs_t)
        attention_probs = F.softmax(attention_weight,dim=-1)
        attention_probs = torch.transpose(attention_probs,2,1)
        attention_vec = torch.mul(attention_probs, inputs)
        attention_vec = torch.sum(attention_vec,dim=1)
        return attention_vec, attention_probs       # (batch_size, input_dim), (batch_size, time_step, input_dim)

class SequenceEncoder(nn.Module):
    def __init__(self,input_dim,time_step,hidden_dim):
        super(SequenceEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=1,batch_first=True)
        self.attention_block = AttentionBlock(time_step,hidden_dim) 
        self.dropout = nn.Dropout(0.2)
        self.dim = hidden_dim
    
    def forward(self,seq):
        '''
        inp : torch.tensor (batch,time_step,input_dim)
        '''
        seq_vector,_ = self.encoder(seq)
        seq_vector = self.dropout(seq_vector)
        attention_vec, _ = self.attention_block(seq_vector)     # (batch, input_dim)
        attention_vec = attention_vec.view(-1,1,self.dim) # prepare for concat
        return attention_vec        # (batch, 1, input_dim)


class GraphEncoder(nn.Module):
    def __init__(self,input_dim,time_step,hidden_dim):
        super(GraphEncoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=1,batch_first=True)
        self.gat = GATConv(hidden_dim,hidden_dim)
        self.dim = hidden_dim
    
    def forward(self,seq,edge_index):
        '''
        inp : torch.tensor (batch,time_step,input_dim)
        '''
        seq_vector,_ = self.encoder(seq)
        seq_vector = seq_vector[:,-1,:]
        attention_vec = self.gat(seq_vector,edge_index)
        return seq_vector, attention_vec


class CategoricalGraph(nn.Module):
    def __init__(self,input_dim,time_step,hidden_dim,inner_edge,outer_edge,input_num,device):
        super(CategoricalGraph, self).__init__()

        # basic parameters
        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.device = device

        # hidden layers
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim,time_step,hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim,hidden_dim)
        self.weekly_attention = AttentionBlock(input_num,hidden_dim)
        self.fusion = nn.Linear(hidden_dim*2,hidden_dim)

        # output layer 
        self.reg_layer = nn.Linear(hidden_dim,1)
        self.cls_layer = nn.Linear(hidden_dim,1)

    def forward(self,weekly_batch):
        # x has shape (category_num, stocks_num, time_step, dim)
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1,self.time_step,self.input_dim)) # (100,1,dim)

        # calculate embeddings for the rest of weeks
        for week_idx in range(1,self.input_num):
            weekly_inp = weekly_batch[week_idx] # (category_num, stocks_num, time_step, dim)
            weekly_inp = weekly_inp.view(-1,self.time_step,self.input_dim) # reshape for faster training 
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp) # (100,1,dim)
            weekly_embedding = torch.cat((weekly_embedding,week_stock_embedding),dim=1)

        # merge weeks 
        weekly_att_vector,_ = self.weekly_attention(weekly_embedding) # (100,dim)
        # weekly_att_vector = weekly_att_vector.view(5,20,-1)
        # category_vectors,_ = torch.max(weekly_att_vector,dim=1)

        # use category graph attention 
        category_vectors = self.cat_gat(weekly_att_vector,self.inner_edge) # (5,dim)
        # category_vectors = category_vectors.unsqueeze(1).expand(-1,20,-1)

        # fusion 
        fusion_vec = torch.cat((weekly_att_vector,category_vectors),dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        # output
        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self,test_data,device,top_k=5):
        y_pred_all = []
        test_w2,test_w3,test_w4 = test_data
        for idx,_ in enumerate(test_w2):
            batch_x2,batch_x3,batch_x4 = test_w2[idx].to(self.device),\
                                        test_w3[idx].to(self.device),\
                                        test_w4[idx].to(self.device)
            batch_weekly = [batch_x2,batch_x3,batch_x4]
            pred = self.forward(batch_weekly)[0].cpu().detach().numpy()
            y_pred_all.extend(pred.tolist())
        return y_pred_all    



class CategoricalGraphAtt(nn.Module):
    def __init__(self,input_dim,time_step,hidden_dim,inner_edge,outer_edge,input_num,use_gru,n_category, device):
        super(CategoricalGraphAtt, self).__init__()

        # basic parameters
        self.dim = hidden_dim
        input_dim -= n_category     # remove the label one hot
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.use_gru = use_gru
        self.device = device
        self.n_category = n_category

        # hidden layers
        self.pool_attention = AttentionBlock(20,hidden_dim)
        if self.use_gru:
            self.weekly_encoder = nn.GRU(hidden_dim,hidden_dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim,time_step,hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim,hidden_dim)
        self.inner_gat = GATConv(hidden_dim,hidden_dim)
        self.weekly_attention = AttentionBlock(input_num,hidden_dim)
        self.fusion = nn.Linear(hidden_dim*3,hidden_dim)

        # output layer 
        self.reg_layer = nn.Linear(hidden_dim,1)
        self.cls_layer = nn.Linear(hidden_dim,1)
        
        #
        assert self.inner_edge is None and self.outer_edge is None

    def forward(self,weekly_batch):
        stocks_num, time_step, _ = weekly_batch[0].shape
        if self.inner_edge is None and self.outer_edge is None:
            # category
            one_hot_category = weekly_batch[0][..., 0, -self.n_category: ]         # (stocks_num, n_category)
            index_category = torch.argmax(one_hot_category, dim=-1) # (stocks_num)
            self.index_category = index_category        # record for max pooling
            
            # inner edge
            inner_edge = [[], []]
            for i in range(stocks_num):
                for j in range(i + 1, stocks_num):
                    if index_category[i] == index_category[j]:
                        inner_edge[0].append(i)
                        inner_edge[1].append(j)
            self.inner_edge = torch.tensor(inner_edge, device=weekly_batch[0].device).long()
            
            # outer edge
            outer_edge = [[], []]
            for i in range(self.n_category):
                for j in range(i + 1, self.n_category):
                    outer_edge[0].append(i)
                    outer_edge[1].append(j)
            self.outer_edge = torch.tensor(outer_edge, device=weekly_batch[0].device).long()
        
        for week in range(self.input_num):
            weekly_batch[week] = weekly_batch[week][..., :-self.n_category]
        
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1,self.time_step,self.input_dim)) # (100,1,dim)

        # calculate embeddings for the rest of weeks
        for week_idx in range(1,self.input_num):
            weekly_inp = weekly_batch[week_idx] # (stocks_num, time_step, dim)
            weekly_inp = weekly_inp.view(-1,self.time_step,self.input_dim) # reshape for faster training 
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp) # (stocks_num,1,dim)
            weekly_embedding = torch.cat((weekly_embedding,week_stock_embedding),dim=1)

        # merge weeks 
        if self.use_gru:
            weekly_embedding,_ = self.weekly_encoder(weekly_embedding)
        weekly_att_vector,_ = self.weekly_attention(weekly_embedding) # (stocks_num, dim)

        # inner graph interaction 
        inner_graph_embedding = self.inner_gat(weekly_att_vector,self.inner_edge)   # (stocks_num, dim)
        # inner_graph_embedding = inner_graph_embedding.view(5,20,-1)

        # pooling 
        # weekly_att_vector = weekly_att_vector.view(5,20,-1)
        # category_vectors,_ =  self.pool_attention(weekly_att_vector) #torch.max(weekly_att_vector,dim=1)
        # category_vectors = torch.scatter_max(dim=0, index=index_category, src=weekly_att_vector)
        category_vectors = torch.zeros((self.n_category, inner_graph_embedding.shape[-1]), device=self.device)
        torch_scatter.scatter_max(
            src=inner_graph_embedding, 
            index=self.index_category,
            dim=0,
            out=category_vectors)
        # category_vectors, _ = torch.scatter(X, dim=0, index=idx.unsqueeze(0).expand(N, -1).long())

        # use category graph attention
        # print(category_vectors.shape)
        category_vectors = self.cat_gat(category_vectors,self.outer_edge) # (5,dim)
        # category_vectors = category_vectors.unsqueeze(1).expand(-1,20,-1)

        # fusion
        expand_category_vectors = torch.gather(
            category_vectors, 
            dim=0, 
            index=self.index_category.unsqueeze(-1).expand(-1, category_vectors.shape[-1]))
        print(weekly_att_vector[:2, :2])
        print(inner_graph_embedding[:2, :2])
        print(expand_category_vectors[:2, :2])
        fusion_vec = torch.cat((weekly_att_vector,inner_graph_embedding, expand_category_vectors),dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        # output
        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)
        print(reg_output[:2])

        return reg_output, cls_output

    def predict_toprank(self,test_data,device,top_k=5):
        y_pred_all_reg, y_pred_all_cls = [], []
        test_w1,test_w2,test_w3,test_w4 = test_data
        for idx,_ in enumerate(test_w2):
            batch_x1,batch_x2,batch_x3,batch_x4 = test_w1[idx].to(self.device), \
                                        test_w2[idx].to(self.device),\
                                        test_w3[idx].to(self.device),\
                                        test_w4[idx].to(self.device)
            batch_weekly = [batch_x1,batch_x2,batch_x3,batch_x4][-self.input_num:]
            pred_reg, pred_cls = self.forward(batch_weekly)
            pred_reg, pred_cls = pred_reg.cpu().detach().numpy(), pred_cls.cpu().detach().numpy()
            y_pred_all_reg.extend(pred_reg.tolist())
            y_pred_all_cls.extend(pred_cls.tolist())
        return y_pred_all_reg, y_pred_all_cls



class CategoricalGraphPool(nn.Module):
    def __init__(self,input_dim,time_step,hidden_dim,inner_edge,inner20_edge,outer_edge,input_num,use_gru,device):
        super(CategoricalGraphPool, self).__init__()

        # basic parameters
        self.dim = hidden_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.inner_edge = inner_edge
        self.inner20_edge = inner20_edge
        self.outer_edge = outer_edge
        self.input_num = input_num
        self.use_gru = use_gru
        self.device = device

        # hidden layers
        self.pool_attention = AttentionBlock(20,hidden_dim)
        if self.use_gru:
           self.weekly_encoder = nn.GRU(hidden_dim,hidden_dim)
        self.encoder_list = nn.ModuleList([SequenceEncoder(input_dim,time_step,hidden_dim) for _ in range(input_num)])
        self.cat_gat = GATConv(hidden_dim*2,hidden_dim)
        self.inner_gat = GATConv(hidden_dim,hidden_dim)
        self.pooling_gcn = SAGPooling(hidden_dim,ratio=0.5)
        self.weekly_attention = AttentionBlock(input_num,hidden_dim)
        self.fusion = nn.Linear(hidden_dim*3,hidden_dim)

        # output layer 
        self.reg_layer = nn.Linear(hidden_dim,1)
        self.cls_layer = nn.Linear(hidden_dim,1)

    def forward(self,weekly_batch):
        # x has shape (category_num, stocks_num, time_step, dim)
        weekly_embedding = self.encoder_list[0](weekly_batch[0].view(-1,self.time_step,self.input_dim)) # (100,1,dim)

        # calculate embeddings for the rest of weeks
        for week_idx in range(1,self.input_num):
            weekly_inp = weekly_batch[week_idx] # (category_num, stocks_num, time_step, dim)
            weekly_inp = weekly_inp.view(-1,self.time_step,self.input_dim) # reshape for faster training 
            week_stock_embedding = self.encoder_list[week_idx](weekly_inp) # (100,1,dim)
            weekly_embedding = torch.cat((weekly_embedding,week_stock_embedding),dim=1)

        # merge weeks 
        if self.use_gru:
            weekly_embedding,_ = self.weekly_encoder(weekly_embedding)
        weekly_att_vector,_ = self.weekly_attention(weekly_embedding) # (100,dim)

        # inner graph interaction 
        inner_graph_embedding = self.inner_gat(weekly_att_vector,self.inner_edge)
        inner_graph_embedding = inner_graph_embedding.view(5,20,-1)

        # pooling 
        weekly_att_vector = weekly_att_vector.view(5,20,-1)
        cat_embdding, _, _, batch, _, _ = self.pooling_gcn(weekly_att_vector[0],self.inner20_edge)
        cat_embdding = torch.cat([global_max_pool(cat_embdding, batch), global_mean_pool(cat_embdding, batch)], dim=-1)
        cat_embdding = cat_embdding.view(1,-1)

        for cat_idx in range(1,5):
            topk_embedding, _, _, batch, _, _ = self.pooling_gcn(weekly_att_vector[cat_idx],self.inner20_edge)
            topk_embedding = torch.cat([global_max_pool(topk_embedding, batch), global_mean_pool(topk_embedding, batch)], dim=-1)
            topk_embedding = topk_embedding.view(1,-1)
            cat_embdding = torch.cat((cat_embdding,topk_embedding),dim=0)


        # use category graph attention 
        category_vectors = self.cat_gat(cat_embdding,self.outer_edge) # (5,dim)
        category_vectors = category_vectors.unsqueeze(1).expand(-1,20,-1)

        # fusion 
        fusion_vec = torch.cat((weekly_att_vector,category_vectors,inner_graph_embedding),dim=-1)
        fusion_vec = torch.relu(self.fusion(fusion_vec))

        # output
        reg_output = self.reg_layer(fusion_vec)
        reg_output = torch.flatten(reg_output)
        cls_output = torch.sigmoid(self.cls_layer(fusion_vec))
        cls_output = torch.flatten(cls_output)

        return reg_output, cls_output

    def predict_toprank(self,test_data,device,top_k=5):
        y_pred_all = []
        test_w1,test_w2,test_w3,test_w4 = test_data
        for idx,_ in enumerate(test_w2):
            batch_x1,batch_x2,batch_x3,batch_x4 = test_w1[idx].to(self.device), \
                                        test_w2[idx].to(self.device),\
                                        test_w3[idx].to(self.device),\
                                        test_w4[idx].to(self.device)
            batch_weekly = [batch_x1,batch_x2,batch_x3,batch_x4][-self.input_num:]
            pred = self.forward(batch_weekly)[0].cpu().detach().numpy()
            y_pred_all.extend(pred.tolist())
        return y_pred_all  
