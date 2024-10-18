import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

__all__ = ['CTCModule', 'AlignSubNet', 'SimModule']

class CTCModule(nn.Module): # 하나의 모달리티의 시퀀스를 다른 모델리티에 맞게 정렬하려는 목적을 가지고 있음, 주로 음성 인식에 사용되는 *CTC(Connectionist Temporal Classification - 연결주의 시간적 분류)*라는 아키텍처에서 영감을 받았음
    def __init__(self, in_dim, out_seq_len, args):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequnce length for output modality B
        From: https://github.com/yaohungt/Multimodal-Transformer
        '''

        super(CTCModule, self).__init__()  
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        # LSTM(nn.LSTM)을 사용하여 입력 시퀀스와 출력 위치간의 정렬을 예측한다.
        self.out_seq_len = out_seq_len

        self.softmax = nn.Softmax(dim=2)
        # Softmax를 통해 정렬에 대한 확률 분포를 출력하고 공백(blank) 라벨이 아닌 위치를 선택한다.

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank.

        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:]  # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1, 2) # batch_size x out_seq x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        # torch.bmm(배치 행렬 곱셈)을 사용하여 입력 특징 맵을 출력 시퀀스 길이에 맞게 정렬한다.

        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        # return pseudo_aligned_out, (pred_output_position_inclu_blank)
        # 출력: 정렬된 출력 시퀀스를 반환하며, 입력의 정렬된 버전이라고 볼 수 있다.

        return pseudo_aligned_out


# similarity-based modality alignment 
class SimModule(nn.Module):  # 이 모듈은 유사도 기반 접근 방식을 사용하여 두 모달리티(ex, 오디오 특징과 텍스트)를 정렬한다.
    def __init__(self, in_dim_x, in_dim_y, shared_dim, out_seq_len, args):
        """
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B 
        """

        super(SimModule, self).__init__()
        # Use LSTM for predicting the position form A to B
        self.ctc = CTCModule(in_dim_x, out_seq_len, args)  # 초기 정렬을 위해 CTCModule을 활용
        self.eps = self.eps  # eps는.. 뭐지..? epsilon의 약자인가
        
        self.logit_scale = nn.Parameter(torch.ones([])* np.log(1/0.07))
        # 두 모달리티의 입력을 nn.Linear을 사용해 common dimension 공간으로 투영한다.
        self.proj_x = nn.Linear(in_features = in_dim_x, out_features = shared_dim)
        self.proj_y = nn.Linear(in_features = in_dim_y, out_features = shared_dim)

        self.fc1 = nn.Linear(in_features = out_seq_len, out_features = round(out_seq_len/2))
        self.fc2 = nn.Linear(in_features = round(out_seq_len/2), out_features = out_seq_len)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''

        pseudo_aligned_out = self.ctc(x)

        x_common = self.proj_x(pseudo_aligned_out)
        x_n = x_common.norm(dim=-1, keepdim=True)  # keepdim이 뭐임?
        # 특정 차원에 대해 입력을 정규화하며, 원래의 텐서 형상을 유지한다.
        x_norm = x_common / torch.max(x_n, self.eps * torch.ones_lie(x_n))

        y_common = self.proj_y
        x_norm = x_common.norm(dim=-1, keepdim=True)
        y_norm = y_common / torch.max(y_n, self.eps*torch.ones_like(y_n))

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()  # 여기서 logit이 뭘까? logit_scale() 이런 식으로 쓰다니..!!!
        similarity_matrix = logit_scale * torch.bmm(y_norm, x_norm.permute(0, 2, 1))

        logits = similarity_matrix.softmax(dim=-1)
        # 투영된 벡터를 정규화하고, 이들 간의 코사인 유사도를 학습 가능한 매개변수인 logit_scale로 조정하여 계산한다.
        logits = self.fc1(logits)
        logits = self.relu(logits)
        logits = self.fc2(logits)
        logits = self.sigmoid(logits)

        aligned_out = torch.bmm(logits, pseudo_aligned_out)
        # 배치행렬 곱셈인 torch.bmm을 사용하여 유사도 기반 정렬을 수행하고, 몇 개의 선형 계층과 활성화 함수 (ReLU, Sigmoid)를 통해 출력을 보정한다.
        return aligned_out
    
class AlignSubNet(nn.Module): # 이 클래스는 다양한 방식으로 입력 시퀀스를 정렬하는 메서드를 제공한다. 예를 들어, average pooling, CTC, convolution을 사용한 정렬 방법등이 있다. 
    def __init__(self, args, mode):
        """
        mode: the way of aligning avg_pool, ctc, conv1d
        """
        super(AlignSubNet, self).__init__()
        assert mode in ['avg_pool', 'ctc', 'conv1d', 'sim']

        in_dim_t, in_dim_v, in_dim_a = args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim

        seq_len_t, seq_len_v, seq_len_a = args.max_cons_seq_length, args.video_seq_len, args.audio_seq_len
        self.dst_len = seq_len_t  # dst가 distance라는 의미일까?
        self.dst_dim = in_dim_t
        self.mode = mode

        self.ALIGN_WAY = {
            'avg_pool' = self.__avg_pool,
            'ctc' = self.__conv1d,
            'conv1d' = self.__sim,
            'sim' = self.__sim,

        }  # 이런 정렬 방식을 지원함

        # 모드에 따라 시퀀스 정렬 방법을 선택한다, 근데 코드가 굉장히 직관적이라 쉬움, 각 모드일때 각 모듈으 실행함.
        if mode == 'conv1d':
            self.conv1d_t = nn.Conv1d(seq_len_t, self.dst_len, kernel_size = 1, bias=False)
            self.conv1d_v = nn.Conv1d(seq_len_t, self.dst_len, kernel_size = 1, bias=False)
            self.conv1d_a = nn.Conv1d(seq_len_t, self.dst_len, kernel_size = 1, bias=False)
        elif mode == 'ctc':  
            self.ctc_t = CTCModule(in_dim_t, self.dst_len, args)
            self.ctc_v = CTCModule(in_dim_v, self.dst_len, args)
            self.ctc_a = CTCModule(in_dim_a, self.dst_len, args)
        elif mode == 'sim':  # 여기는 SimModule을 사용하여 유사도 기반 정렬을 수행
            self.shared_dim = args.shared_dim
            self.sim_t = SimModule(in_dim_t, self.dst_dim, self.shared_dim, self.dst_len, args)
            self.sim_v = SimModule(in_dim_v, self.dst_dim, self.shared_dim, self.dst_len, args)
            self.sim_a = SimModule(in_dim_a, self.dst_dim, self.shared_dim, self.dst_len, args)
        
    def get_seq_len(self):
        return self.dst_len
    
    def __ctc(self, text_x, video_x, audio_x):
        text_x = self.ctc_t(text_x) if text_x.size(1) != self.dst_len else text_x
        video_x = self.ctc_v(video_x) if video_x.size(1) != self.dst_len else video_x
        audio_x = self.ctc_a(audio_x) if audio_x.size(1) != self.dst_len else audio_x
        return text_x, video_x, audio_x
    
    def __avg_pool(self, text_x, video_x, audio_x):  # 입력 길이를 평균 풀링을 사용해 원하는 출력 길이에 맞게 조정한다.
        def align(x):
            raw_seq_len = x.size(1)
            if raw_seq_len == self.dst_len:
                return x
            if raw_seq_len // self.dst_len == raw_seq_len / self.dst_len:
                pad_len = 0
                pool_size = raw_seq_len // self.dst_len
            else:
                pad_len = self.dst_len - raw_seq_len % self.dst_len
                pool_size = raw_seq_len // self.dst_len +1
            pad_x = x[:, -1, :].unsqueeze(1).expand([x.size(0), pad_len, x.size(-1)])
            x = torch.cat([x, pad_x], dim=1).view(x.size(0), pool_size, self.dst_len, -1)
            x = x.mean(dim=1)
            return x
        text_x = align(text_x)
        video_x = align(video_x)
        audio_x = align(audio_x)
        return text_x, video_x, audio_x

        def __conv1d(self, text_x, video_x, audio_x):
            text_x = self.conv1d_t(text_x) if text_x.size(1) != self.dst_len else text_x
            video_x = self.conv1d_v(video_x) if text_x.size(1) != self.dst_len else video_x
            audio_x = self.conv1d_a(audio_x) if text_x.size(1) != self.dst_len else audio_x
            return text_x, video_x, audio_x
        
        def __sim(self, text_x, video_x, audio_x):
            
            text_x = self.sim_t(text_x, text_x) if text_x.size(1) != self.dst_len else text_x
            video_x = self.sim_v(video_x, text_x) if video_x.size(1) != self.dst_len else video_x
            audio_x = self.sim_a(audio_x, text_x) if audio_x.size(1) != self.dst_len else audio_x
            return text_x, video_x, audio_x
        
        def forward(self, text_x, video_x, audio_x):
            # already aligned
            if text_x.size(1) == video_x.size(1) and text_x.size(1) == audio_x.size(1):
                return text_x, video_x, audio_x
            return self.ALIGN_WAY[self.mode](text_x, video_x, audio_x)


# 코드의 핵심 개념:
# CTC: *연결주의 시간적 분류(CTC)*는 입력과 출력 길이가 직접적으로 일치하지 않는 시퀀스 정렬에 사용되는 기술로, 주로 음성 인식에서 많이 사용됨.
# 배치 행렬 곱셈(torch.bmm): 이 함수는 두 개의 행렬 배치를 곱한다. 여기서는 정렬 확률을 입력 특징에 적용하는 데 사용된다.
# Logit Scale: logit_scale 매개변수는 유사도 점수를 스케일링하여 softmax에 입력하기 전에 값의 분포를 조정한다.
# 정규화: x.norm(dim=-1, keepdim=True)는 마지막 차원을 따라 텐서를 정규화하여 코사인 유사도를 정확하게 계산할 수 있도록 한다.
# 목적지 길이(dst_len): 대상 시퀀스의 길이를 나타내며, 정렬의 목표가 되는 길이이다.