import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self,d_emb,n_position):
        super(PositionalEncoding,self).__init__()

        self.register_buffer('pos_table',self._sinusoid_encoding_table(d_emb,n_position))
        
    ''' Sinusoid position encoding table '''
    def _sinusoid_encoding_table(self,d_emb,n_position):
        def pos_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_emb) for hid_j in range(d_emb)]

        sinusoid_table = np.array([pos_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:,0::2] = np.sin(sinusoid_table[:,0::2]) ## dim 2i
        sinusoid_table[:,1::2] = np.sin(sinusoid_table[:,1::2]) ## dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)
    
    def forward(self,x):
        return x+self.pos_table[:,:x.size(1)].clone().detach()

class ScaledDotProductAttention(nn.Module):
    def __init__(self,temperature,attn_dropout=0.1):
        super().__init__()
        self.temp = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self,q,k,v,mask=None):
        QK = torch.matmul(q/self.temp, k.transpose(2,3))

        if mask is not None:
            QK = QK.masked_fill(mask==0,-1e9)
        attn = self.dropout(F.softmax(QK,dim=-1))
        output = torch.matmul(attn,v)

        return output,attn
        
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_heads,dropout):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads

        self.w_q = nn.Linear(d_model,n_heads*self.d_k,bias=False)
        self.w_k = nn.Linear(d_model,n_heads*self.d_k,bias=False)
        self.w_v = nn.Linear(d_model,n_heads*self.d_v,bias=False)
        self.fc = nn.Linear(n_heads*self.d_v,d_model,bias=False) 
        
        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self,q,k,v,mask=None):
        batch_size,len_q,len_k,len_v = q.size(0),q.size(1),k.size(1),v.size(1)
        residual = q

        q = self.w_q(q).view(batch_size,len_q,self.n_heads,self.d_k)
        k = self.w_q(k).view(batch_size,len_k,self.n_heads,self.d_k)
        v = self.w_q(v).view(batch_size,len_v,self.n_heads,self.d_v)
        q,k,v = q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q,attn = self.attention(q,k,v,mask=mask) ## Output:bs x len x n x d
        q = q.transpose(1,2).contiguous().view(batch_size,len_q,-1) ## Output:bs x len x d_model
        q = self.dropout(self.fc(q))
        q += residual          
        q = self.layer_norm(q)

        return q,attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self,d_in,d_hid,dropout):
        super().__init__()
        self.w1 = nn.Linear(d_in,d_hid)
        self.w2 = nn.Linear(d_hid,d_in)
        self.layer_norm = nn.LayerNorm(d_in,eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        residual = x

        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)

        return x

class EncoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,n_heads,dropout):
        super(EncoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model,d_ff,dropout)

    def forward(self,enc_in,self_attn_mask):
        enc_out,self_attn = self.self_attn(enc_in,enc_in,enc_in,self_attn_mask)
        enc_out = self.pos_ffn(enc_out)

        return enc_out,self_attn

class DecoderLayer(nn.Module):
    def __init__(self,d_model,d_ff,n_heads,dropout):
        super(DecoderLayer,self).__init__()
        self.self_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.enc_attn = MultiHeadAttention(d_model,n_heads,dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model,d_ff,dropout)

    def forward(self,dec_in,enc_out,self_attn_mask,dec_enc_attn_mask):
        dec_out,self_attn = self.self_attn(dec_in,dec_in,dec_in,self_attn_mask)
        dec_out,dec_enc_attn = self.enc_attn(dec_out,enc_out,enc_out,dec_enc_attn_mask)
        dec_out = self.pos_ffn(dec_out)

        return dec_out,self_attn,dec_enc_attn


class Encoder(nn.Module):
    def __init__(self,n_vocab,n_position,pad_idx,d_model,d_ff,n_heads,n_layers,
            dropout,scale_emb):
        super().__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model

        self.word_emb = nn.Embedding(n_vocab,d_model,padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model,n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)
        self.layer_stack = nn.ModuleList(
                [EncoderLayer(d_model,d_ff,n_heads,dropout=dropout) for _ in range(n_layers)])
        

    def forward(self,seq,mask):
        word_emb = self.word_emb(seq)
        if self.scale_emb:
            word_emb *= self.d_model ** 0.5
        word_pos_emb = self.layer_norm(self.dropout(self.pos_enc(word_emb)))

        enc_out = word_pos_emb
        for layer in self.layer_stack:
            enc_out,self_attn = layer(enc_out,mask)

        return enc_out


class Decoder(nn.Module):
    def __init__(self,n_vocab,n_position,pad_idx,d_model,d_ff,n_heads,n_layers,
            dropout,scale_emb):
        super().__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model

        self.word_emb = nn.Embedding(n_vocab,d_model,padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model,n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model,eps=1e-6)
        self.layer_stack = nn.ModuleList(
                [DecoderLayer(d_model,d_ff,n_heads,dropout=dropout) for _ in range(n_layers)])

    def forward(self,trg_seq,trg_mask,enc_out,src_mask):
        trg_word_emb = self.word_emb(trg_seq)
        if self.scale_emb:
            trg_word_emb *= self.d_model ** 0.5
        trg_word_pos_emb = self.layer_norm(self.dropout(self.pos_enc(trg_word_emb)))

        dec_out = trg_word_pos_emb
        for layer in self.layer_stack:
            dec_out,self_attn,dec_enc_attn = layer(dec_out,enc_out,trg_mask,src_mask)

        return dec_out

##
def get_pad_mask(seq,pad_idx):
    return (seq!=pad_idx).unsqueeze(-2)

def get_subseq_mask(seq):
    bs,len_s = seq.size()
    mask = (1-torch.triu(torch.ones((1,len_s,len_s),device=seq.device),diagonal=1)).bool()
    return mask

class Transformer(nn.Module):
    def __init__(self,n_src_vocab,n_trg_vocab,src_pad_idx,trg_pad_idx,
            d_model,d_ff,n_heads,n_layers,dropout,
            trg_emb_prj_weight_sharing, emb_src_trg_weight_sharing,scale_emb_or_prj,
            n_pos=200):

        super().__init__()
        self.d_model = d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
       
        self.encoder = Encoder(n_src_vocab,n_pos,src_pad_idx,
            d_model,d_ff,n_heads,n_layers,dropout,scale_emb)
        self.decoder = Decoder(n_trg_vocab,n_pos,trg_pad_idx,
            d_model,d_ff,n_heads,n_layers,dropout,scale_emb)
        self.trg_word_prj = nn.Linear(d_model,n_trg_vocab,bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.word_emb.weight

        if emb_src_trg_weight_sharing:
            self.encoder.word_emb.weight = self.decoder.word_emb.weight

        
                
    def forward(self,src_seq,trg_seq):
        src_mask = get_pad_mask(src_seq,self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq,self.trg_pad_idx) & get_subseq_mask(trg_seq)

        enc_output = self.encoder(src_seq,src_mask)
        dec_output = self.decoder(trg_seq,trg_mask,enc_output,src_mask)

        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model**(-0.5)

        return seq_logit.view(-1,seq_logit.size(2))

