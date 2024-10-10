import math

import torch
import torch.nn as nn

# Code adapted from the fairseq repo.


""" Transformer 구조에서 매우 중요한 부분중 하나로 , 입력 시퀀스에서 단어의 순서와 같은 위치정보를 효율적으로 반영해주는 역할을 한다. """
def make_positions(tensor, padding_idx, left_pad):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1.
    Padding symbols are ignored, but it is necessary to specify whether padding is added on the lefy side (left_pad=True) or right side (left_pad=False).
    """

    max_pos = padding_idx + 1 + tensor.size(1)
    device = tensor.get_device()
    buf_name = f'range_buf_{device}'
    if not hasattr(make_positions, buf_name):
        setattr(make_positions, buf_name, tensor.new())
    setattr(make_positions, buf_name, getattr(make_positions, buf_name).type_as(tensor))
    if getattr(make_positions,buf_name).numel() < max_pos:   # numel()이라는 메서드도 있군
        torch.arange(padding_idx+1, max_pos, out=getattr(make_positions, buf_name))
    mask = tensor.ne(padding_idx)
    positions = getattr(make_positions, buf_name)[:tensor.size(1)].expand_as(tensor)
    if left_pad:
        positions = positions - mask.size(1) + mask.long().sum(dim=1).unsqueeze(1)
    new_tensor = tensor.clone()
    return new_tensor.masked_scatter_(mask, positions[mask]).long()


class SinusoidalPositionalEmbedding(nn.Module):
    """ This module produces sinusoidal positional embeddings of any length.
    Padding symbols are ignored, but it is necessary to specify whether padding is 
    added on the left side(left_pad=True) or right side(left_pad=False).
    """
    
    def __init__(self, embedding_dim, padding_idx=0, left_pad=0, init_size=128):
        super().__init__()
        self.embdding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.left_pad = left_pad
        self.weights = dict()  # device --> actual weight; due to nn.DataParallel :-(, weights는 각장치에서 사용할 수 있는 임베딩을 저장하는 딕셔너리로, DataParallel 모델을 사용할 떄 각 장치에서 임베딩을 유지하기 위한 구조이다.
        self.register_buffer('_float_tensor', torch.FloatTensor(1))  # register_buffer는 PyTorch에서 모델의 상태를 저장하는데 사용되지만, 학습 가능한 파라미터로 취급되지 않는다.

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """ Build sinusoidal embeddings.
        This matches the implementation in tensor2tensor, 
        but differs slightly from the description in Section 3.5 of "Attention Is All You Need".
        """

        half_dim = embedding_dim //2
        emb = math.log(10000) / (half_dim-1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float)* -emb)
        emb = torch.arange(num_embeddings, dtpye=torch.float).unsqueeze(1)*emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)  # sin과 cos 함수 사용
        if embedding_dim %2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    
    def forward(self, input):
        """ Input is expected to be of size [bsz x seqlen]. """  # bsz는 batch_size의 약자, seqlen은 sequence_length의 약자
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        device = input.get_device()
        if device not in self.weights or max_pos > self.weights[device].size(0):
            # recompute/expand embeddings if needed
            self.weights[device] = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights[device] = self.weights[device].type_as(self.float_tensor).to(input.device)
        positions = make_positions(input, self.padding_idx, self.left_pad)
        return self.weights[device].index_select(0, positions.reshape(-1)).reshape(bsz,seq_len, -1)
    
    def max_positions(self):  # 지원가능한 최대 위치를 반환하여, 이는 기본적으로 매우 큰 임의의 값(1e5, 100,000)으로 설정해놓음
        """ Maximum number of supported postions. """
        return int(1e5)  # an arbitrary large number