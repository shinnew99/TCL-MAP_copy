import torch
from torch import nn  # nn이 Neural network의 약자였다니, PyTorch의 neural network 모듈, layers와 model을 정의한다.
from torch.nn import Parameter
import torch.nn.functional as F
import sys

# Code adatped from the fairseq repo.

class MultiheadAttention(nn.Module):  # inherits from nn.Module, 
    """Multi-headed attention.
    See "Attention is All you Need" for more details.    
    """
    # Attention Is All You Need 논문에서 제안된 multi-head attention 메커니즘을거의 그대로 구현한 코드

    def __init__(self, embed_dim,num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv =False, add_zero_attn=False):
        # embed_dim: input 임베딩값의 차원
        # add_bias_kv: whether to include learnable bias for the key and value
        # add_zero_attn: whether to add zero attention for padding tokens (maksing을 도와줌)

        # 파라미터초기화 및 검증
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim //num_heads  # num_heads가 embed_dim의 몫(약수)임을 확인
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5  # attention 점수를안정적으로 만들기 위해 head_dim의 제곱근을 곱한다, a scalingfactor (1/sqrt(head_dim)), which is used to scale the query vectors in attention calculation


        # 가중치 및 bias 초기화
        self.in_proj_weight = Parameter(torch.Tensor(3*embed_dim, embed_dim))  # in_proj_weight은 query, key, value에 대한 입력을 학습할가중치, 3개의 attention matrix(q, k, v)를 처리하기 위해 3배의 embed_dim크기로 설정
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3*embed_dim))  # bias를 적용할지 여부에 따른 초기화, 근데 왜 3을 곱할까..?
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # attention 후, 출력값을 재구성하는 linear layer

        # 추가적인 bias 파라미터 - add_bias_kv가 조건을 만족할 때, 추가적인 bias를적용할지 여부를 결정, 그렇지 않면 None으로 설정한다.
        if add_bias_kv:  
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        
        # zero-attention 관련 설정: zero-attention을 추가할지 여부를설정하고 가중치 초기화를 위해 reset_parameters() 함수를 호출한다. 
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()

    # 파라미터 초기화 함수
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)  # Xavier 초기화 방식으로 가중치들을 초기화한다. 이는 신경망에서 가중치가 너무 크거나 작지 않게 유지되도록 도와준다.
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.consatant_(self.out_proj.weight, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_normal(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    # forward 함수: 보내주는 역할, 입력 데이터를받아서 어텐션 메커닞므을 적용하여 출력을 계산한다, 주석에서 보듯이 query, key, value는 모두 (Time x Batch x Channel)의 형식을 가진다.    
    def forward(self, query, key, value, attn_mask = None):
        """ Input shae: Time x Batch X channel
        Self-attention can be implemented by passing inthe same arguments for query, key and value.
        Timesteps can be masked by supplying a TxT mask in the 'attn_mask'argument. Padding elementscanbe excluded from the ky by passing a binary ByteTensor('key_padding_mask') with shape:
        batch x src_len, where padding elemetnsare indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()  # self-attention인지 확인하는 부분. query, key, value가 같은 데이터를가리킬 경우 self-attention임을 의미
        kv_same = key.data_ptr() == value.data_ptr()  # encoder-decoder attention인지 확인

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v =self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value) 
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)


        q = q.contiguous().view(tgt_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz*self.num_heads, self.head_dim).transpose(0,1)
        if v is not None:
            v = v.contiguous().view(-1, bsz* self.num_heads, self.head_dim).transpose(0,1)
        
        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz*self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
        
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz*self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1)/self.num_heads
        return attn, attn_weights
    
    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self.in_proj(key, start=self.embed_dim).chunk(2, dim=-1)
    
    def in_proj_q(self, query, **kwargs):
        return self.in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self.in_proj(key, start=self.embed_dim, end=2*self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2*self.embed_dim)
    
    def _in_proc(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_projc_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)



