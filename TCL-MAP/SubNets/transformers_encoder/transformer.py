import torch
from torch import nn
import torch.nn.functional as F
from .position_embedding import SinusoidalPositionalEmbedding
from .multihead_attention import MultiheadAttention
import math


class TransformerEncoder(nn.Module):
      """
      Transformerencoder cosisting of*args.encoder_layers* layers. 
      Each layer is a :class: 'TransformerEncoderLayer'.
      Args:
        embed_tokens (torch.nn.Embedding): input enbedding
        num_heads (int): number of heads
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
      """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                    embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.dropout = embed_dropout  # Embedding drouput
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)

        self.attn_mask = attn_mask
        self.layers =nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads = num_heads,
                                                attn_dropout = attn_dropout,
                                                relu_dropout = relu_dropout,
                                                res_dropout = res_dropout,
                                                attn_mask = attn_mask)
            self.layers.append(new_layer)
        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
              self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v=None):
        """
        x_in (FloatTensor): embed input of shape '(src_len, batch, embed_dim)'
        x_in_k (FloatTensor): embedded input of shape '(src_len, batch, embed_dim)'
        x_in_v (FloatTensor): embedded input of shape '(src_len, batch, embed_dim)'
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of shape '(src_len, batch, embed_dim)'
                - **encoder_padding_mask** (ByteTensor): the positions of padding elements of shape '(batch, src_len)'
        """
    
        # embed tokens and positions
        x = self.embed_scale * x_in
        if self.embed_position is not None:
           x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0].transpose(0,1))  # Add positional embedding
        x = F.dropout(x, p=self.dropout, training=self.training)
      
        if x_in_k is not None and x_in_v is not None:
           # embed tokens and positions
           x_k = self.embed_scale * x_in_k
           x_v = self.embed_scale * x_in_v
           if self.embed_positions is not None:
                x_k += self.embed_positions(x_in_k.transpose(0,1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
                x_v += self.embed_positions(x_in_v.transpose(0,1)[:, :, 0]).transpose(0, 1)  # Add positional embedding
           x_k = F.dropout(x_k, p=self.dropout, traiing=self.training)
           x_v = F.dropout(x_v, p=self.dropout, training=self.training)


        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x  = layer(x, x_k, x_v)
            else:
                 x = layer(x)
            intermediates.append(x)
        
        if self.normalize:
             x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
             return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerEncoderLayer(nn.Module):
    """ Encoder layer block.
    In the original paper each operation (multi-head attention or FFN) is postprocessedwith: 'dropout -> add residual -> layernorm'.
    In the tensor2tensor code they suggest that learning is more robust when preprocessing each layer with layernorms and postprocessing with: 'dropout -> add residual'.
    We default to the approach in the paper, but the tensor2tensor approach can be enabledbythe setting *args.encoder_normalize_before* to ''True''.
    Args:
       embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
          super().__init__()
          self.embed_dim = embed_dim
          self.num_heads = num_heads

          self.self_attn = MultiheadAttention(  # 모듈을 사용해 다중 헤드 주의(attention)를 구현하고, LayerNorm 및 Linear 레이어를 정의
               embed_dim = self.embed_dim,
               num_heads = self.num_heads,
               attn_dropout = attn_dropout
          )
          self.attn_mask =attn_mask

          self.relu_dropout = relu_dropout
          self.res_dropout = res_dropout
          self.normalize_before = True

          self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)  # The "Add *Norm" part in the paper
          self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
          self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])


    def forward(self, x, x_k = None, x_v = None):
        """ 
        Args:
            x_in (FloatTensor): embedded input of shape '(src_len, batch, embed_dim)'
            x_in_k (FloatTensor): embedded input of shape '(src_len, batch, embed_dim)'
            x_in_v (FloatTensor): embedded input of shape '(src_len, batch, embed_dim)'
        
        Returns:
            dict:
                - **encoder_out** (Tensor):the last encoder layer's output of shape '(src_len, batch, embed_dim)'
                - **encoder_padding_mask** (ByteTensor): the positions of padding elements of shape '(batch, src_len)'
        """

      
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x
    
    def maybe_layer_norm(self, i, x, before=False, after=False):
         assert before ^ after
         if after ^ self.normalize_before:
              return self.layer_norms[i](x)
         else:
              return x


# 보조 함수
def fill_with_neg_inf(t):
        """ FP16-compatible function that fills a tensor with -inf."""
        return t.float().fill_(float('-inf')).type_as(t)  # -inf로 채우는 self.attention에서 future masking을 만들 때 사용하는 함수
    

def buffered_future_mask(tensor, tensor2=None):  # 특정 시점 이후의 값에 마스킹을 적용해 미래 정보를 차단하는 역할을 한다, 시퀀스 데이터에서 인과적 (causal) attention을 구현할 때 사용된다.
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(tensor.device)
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):  # nn.Linear를 초기화하는 함수로, Xavier 초기화 방식으로 가중치를 설정하고 편향이 있을 경우 이를 0으로 초기화한다.
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xabier_uniform_(m.weight)
    if bias:
        nn.ini.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):  # 임베딩차원을 입력받아 nn.LayerNorm 모듈을 생성하는 함수
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == "__main__":
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 20, 300))
    print(encoder(x).shape)


""" Transformer 코드와 유사하지만 일부 차이가 있음, 둘 다 Transformer 모델의 인코더 부분을 정의하고 있지만, 몇 가지 다른 점들이 존재함. 
예를 들어:
1. 가져오는 모듈: transformer.py에서 SinusoidalPositionalEmbedding, MultiheadAttention을 가져오고 있음. 이 부분이 직접적으로 사용되는지 여부를 코드에서 확인할 필요가 있음.
2. Embedding 관련 차이점: TransformerEncoder 클래스에서는 self.embed_scale, self.embed_position 등의 임베딩 관련 변수를 사용하는데, 임베딩에 대한 자세한 정의는 
두 코드 모두에서 완전하지 않으므로 일부 세부 구현에서 차이가 있을 수 있음.
3. 레이어 정의 및 초기화: 두 코드 모두 다수의 Transformer 인코더 레이어를 사용하는데, 레이어를 구성하는 방식에서 Linear, LayerNorm 초기화와 관련된 세부 구현이 다른 점이 있음.
4. Multihead Attention 사용: 두 코드 모두 Multihead Attention을 사용하지만, 구현 방식에서의 차이가 있을 수 있음.

요약하자면, 두 코드가 매우 유사한 Transformer 인코더 구조를 따르고 있지만 세부적인 초기화나 구현 방식에서 약간의 차이점이 있음.
"""