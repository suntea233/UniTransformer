
import torch
import torch.nn as nn
from TorchCRF import CRF

import torch.nn.functional as F
from transformers import RobertaModel,ViTModel


def get_padding_mask(seq_q,seq_k):
    # print(seq_k.size())
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    padding_mask = seq_k.data.eq(1).unsqueeze(1)
    return padding_mask.expand(batch_size,len_q,len_k)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout_rate, device, mask=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_head = num_heads
        self.dropout_rate = dropout_rate
        self.mask = mask
        self.linearQ = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linearK = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linearV = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LayerNormalization = nn.LayerNorm(hidden_dim)
        self.Q = nn.Sequential(self.linearQ,self.relu)
        self.K = nn.Sequential(self.linearK,self.relu)
        self.V = nn.Sequential(self.linearV,self.relu)
        self.device = device

    def forward(self, queries, keys, values, attention_mask):
        '''
        :param queries: shape:(batch_size,input_seq_len,d_model)
        :param keys: shape:(batch_size,input_seq_len,d_model)
        :param values: shape:(batch_size,input_seq_len,d_model)
        :return: None
        '''
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        q_split, k_split, v_split = torch.chunk(q,self.num_head,dim=-1), torch.chunk(k,self.num_head,dim=-1), torch.chunk(v,self.num_head,dim=-1)
        q_, k_, v_ = torch.stack(q_split,dim=1), torch.stack(k_split,dim=1), torch.stack(v_split,dim=1)
        # shape : (batch_size, num_head, input_seq_len, depth = d_model/num_head)
        a = torch.matmul(q_,k_.permute(0,1,3,2)) # a = q * k^T(后两个维度)
        a = a / (k_.size()[-1] ** 0.5) # shape:(batch_size,num_head,seq_len,seq_len)
        if attention_mask != None:
            seq_len = attention_mask.shape[1]
            mask = attention_mask.unsqueeze(2).repeat(1, 1, seq_len).bool()
            mask = mask.unsqueeze(1).repeat(1, self.num_head, 1, 1)
            a = torch.masked_fill(a, mask=~mask, value=1e-9)

        a = F.softmax(a,dim=-1)

        a = torch.matmul(a,v_)
        a = torch.reshape(a.permute(0, 2, 1, 3), shape=(q.shape[0],q.shape[1],q.shape[2]))
        a += queries
        a = self.LayerNormalization(a)
        return a


class FC(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,device):
        super().__init__()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(self.input_dim,hidden_dim)
        self.layer2 = nn.Linear(hidden_dim,output_dim)
        self.relu = nn.ReLU()

        self.LayerNormalization = nn.LayerNorm(output_dim)
        self.device = device




    def forward(self,x):
        # print(x.shape)
        outputs = self.layer1(x)
        outputs = self.relu(outputs)
        outputs = self.layer2(outputs)
        outputs += x
        outputs = self.LayerNormalization(outputs)
        return outputs


class EncoderLayer(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim,dropout_rate,num_heads,device):
        super().__init__()
        self.device = device
        self.self_attention = MultiHeadAttention(hidden_dim=hidden_dim,num_heads=num_heads,dropout_rate=dropout_rate,device=self.device)
        self.fc =FC(input_dim=input_dim,hidden_dim=hidden_dim,output_dim=output_dim,device=self.device)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self,x,attention_mask):

        attention_score = self.self_attention(x,x,x,attention_mask)
        outputs = self.fc(attention_score) # shape = (batch_size,seq_len,d_model)
        return outputs


class Encoder(nn.Module):
    def __init__(self,num_layers,input_dim,hidden_dim,output_dim,dropout_rate,num_heads,device):
        super().__init__()

        self.layers = nn.ModuleList([EncoderLayer(input_dim,hidden_dim,output_dim,dropout_rate,num_heads,device) for _ in range(num_layers)])


    def forward(self,x,attention_mask=None):

        for layer in self.layers:
            x = layer(x,attention_mask)
        return x


class UniTransformer(nn.Module):
    def __init__(self,args):
        super(UniTransformer, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_embedding = RobertaModel.from_pretrained(args.text_pretrained_model).embeddings
        self.image_embedding = ViTModel.from_pretrained(args.image_pretrained_model).embeddings
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.num_layers = args.num_layers
        self.dropout_rate = args.dropout_rate
        self.num_heads = args.num_heads
        self.max_len = args.max_len
        self.num_labels = args.num_labels

        self.image_fc = FC(self.input_dim,self.hidden_dim,self.output_dim,self.device)
        self.text_fc = FC(self.input_dim,self.hidden_dim,self.output_dim,self.device)
        self.multi_fc = FC(self.input_dim,self.hidden_dim,self.output_dim,self.device)

        self.logits = nn.Linear(self.output_dim,self.num_labels)

        self.encoder = Encoder(self.num_layers,self.input_dim,self.hidden_dim,self.output_dim,self.dropout_rate,self.num_heads,self.device)
        self.momentum_encoder = Encoder(self.num_layers,self.input_dim,self.hidden_dim,self.output_dim,self.dropout_rate,self.num_heads,self.device)


        self.crf = CRF(self.num_labels, batch_first=True)
        self.queue_size = 16

        self.model_pairs = [
            [self.encoder, self.momentum_encoder]
        ]

        self.temp = nn.Parameter(torch.ones([]) * 0.07)
        self.register_buffer("image_queue", torch.tensor([]))
        self.register_buffer("text_queue", torch.tensor([]))
        self.register_buffer("attention_mask_queue",torch.tensor([]))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.momentum = 0.999
        self.two_stage = False

#when second_step, not frozen
    def forward(self,text,image,attention_mask,labels=None):

        image_embedding_outputs = self.image_embedding(image)
        text_embedding_outputs = self.text_embedding(text)

        if self.two_stage == False:
            image_hidden_state = self.encoder(image_embedding_outputs)
            text_hidden_state = self.encoder(text_embedding_outputs, attention_mask)

            image_last_hidden_state = self.image_fc(image_hidden_state)
            text_last_hidden_state = self.text_fc(text_hidden_state)

            loss = self.MOCO(text,image,text_last_hidden_state,image_last_hidden_state,attention_mask)
            return loss

        else:
            image_hidden_state = self.encoder(image_embedding_outputs)
            text_hidden_state = self.encoder(text_embedding_outputs, attention_mask)

            image_last_hidden_state = self.image_fc(image_hidden_state)
            text_last_hidden_state = self.text_fc(text_hidden_state)

            multi_hidden_state = torch.cat([text_last_hidden_state,image_last_hidden_state],dim=1)

            batch = image.shape[0]

            multi_attention_mask = torch.cat([attention_mask,torch.ones(size=(batch,self.max_len),device=self.device)],dim=1)

            multi_hidden_state = self.encoder(multi_hidden_state,attention_mask=multi_attention_mask)

            multi_last_hidden_state = self.multi_fc(multi_hidden_state)
            logits = self.logits(multi_last_hidden_state)

            if labels is not None:
                crf_mask = attention_mask.eq(1)
                crf_loss = -self.crf(logits, labels, crf_mask)
                return crf_loss
            else:
                return logits



    def MOCO(self, text, image, text_hidden_state, image_hidden_state, attention_mask):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        image_feat = image_hidden_state
        text_feat = text_hidden_state

        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            image_embedding_outputs = self.image_embedding(image)
            text_embedding_outputs = self.text_embedding(text)


            image_feat_m = self.momentum_encoder(image_embedding_outputs)

            image_feat_all = torch.cat([image_feat_m, self.image_queue.clone().detach()], dim=0)

            text_feat_m = self.momentum_encoder(text_embedding_outputs, attention_mask=attention_mask)
            text_feat_all = torch.cat([text_feat_m, self.text_queue.clone().detach()], dim=0)


        t_attention_mask = torch.cat([attention_mask, self.attention_mask_queue.clone().detach()])

        i2t_loss1, i2t_loss2 = self.token_with_patch_contrastive_loss(text_feat_all, image_feat, t_attention_mask,
                                                                      dim=0)
        t2i_loss1, t2i_loss2 = self.token_with_patch_contrastive_loss(text_feat, image_feat_all, attention_mask, dim=-1)

        loss_i2t = i2t_loss1 + i2t_loss2
        loss_t2i = t2i_loss1 + t2i_loss2
        loss_itc = (loss_i2t + loss_t2i) / 2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, attention_mask)

        return loss_itc


    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, attention_mask):
        batch_size = image_feat.shape[0]
        ptr = int(self.queue_ptr)
        if self.text_queue.shape[0] >= self.queue_size:

            if ptr + batch_size > self.text_queue.shape[0]:
                t = self.text_queue.shape[0] - ptr
                self.image_queue[ptr:, :, :] = image_feat[:t, :, :]
                self.text_queue[ptr:, :, :] = text_feat[:t, :, :]
                self.attention_mask_queue[ptr:, :] = attention_mask[:t, :]

                self.image_queue[:batch_size - t, :, :] = image_feat[t:batch_size, :, :]
                self.text_queue[:batch_size - t, :, :] = text_feat[t:batch_size, :, :]
                self.attention_mask_queue[:batch_size - t, :] = attention_mask[t:batch_size, :]
                ptr = batch_size - t
            # # replace the keys at ptr (dequeue and enqueue)
            else:
                self.image_queue[ptr:ptr + batch_size, :, :] = image_feat
                self.text_queue[ptr:ptr + batch_size, :, :] = text_feat
                self.attention_mask_queue[ptr:ptr + batch_size, :] = attention_mask
                ptr = (ptr + batch_size) % self.text_queue.shape[0]  # move pointer

        else:
            self.image_queue = torch.cat([self.image_queue, image_feat])
            self.text_queue = torch.cat([self.text_queue, text_feat])
            self.attention_mask_queue = torch.cat([self.attention_mask_queue, attention_mask])

            ptr = (ptr + batch_size) % self.text_queue.shape[0]  # move pointer
        self.queue_ptr[0] = ptr


    def token_with_patch_contrastive_loss(self, text_hidden_state, image_hidden_state, attention_mask, dim):
        t2i_sim = []
        i2t_sim = []
        for i in range(len(text_hidden_state)):
            temp1 = []
            temp2 = []
            for j in range(len(image_hidden_state)):
                sim1 = self.t2i_compute_similarity(text_hidden_state[i], image_hidden_state[j], attention_mask[i])
                temp1.append(sim1.item())
                sim2 = self.i2t_compute_similarity(text_hidden_state[i], image_hidden_state[j], attention_mask[i])
                temp2.append(sim2.item())
            t2i_sim.append(temp1)
            i2t_sim.append(temp2)

        text2img_sim = torch.tensor(t2i_sim, device=self.device) / self.temp
        img2text_sim = torch.tensor(i2t_sim, device=self.device) / self.temp

        y_true = torch.zeros(text2img_sim.size(),device=self.device)
        y_true.fill_diagonal_(1)
        y_true = y_true.long()
        t2i_loss = -torch.sum(F.log_softmax(text2img_sim, dim=dim) * y_true, dim=1).mean()
        i2t_loss = -torch.sum(F.log_softmax(img2text_sim, dim=dim) * y_true, dim=1).mean()
        return t2i_loss, i2t_loss


    def t2i_compute_similarity(self, text_hidden_state, image_hidden_state, attention_mask):
        multi_hidden_state = torch.matmul(text_hidden_state, image_hidden_state.transpose(0, 1))
        text2img_max_score = multi_hidden_state.max(-1)[0].float()
        padding_length = sum(attention_mask)
        text2img_mean_score = text2img_max_score * attention_mask
        text2img_mean_score = text2img_mean_score.sum(dim=-1)
        text2img_similarity = text2img_mean_score / padding_length
        return text2img_similarity


    def i2t_compute_similarity(self, text_hidden_state, image_hidden_state, attention_mask):
        multi_hidden_state = torch.matmul(image_hidden_state, text_hidden_state.transpose(0, 1))
        img2text_max_score = multi_hidden_state * attention_mask
        img2text_mean_score = img2text_max_score.max(-1)[0].float()
        img2text_mean_score = img2text_mean_score.mean(dim=-1)
        return img2text_mean_score

