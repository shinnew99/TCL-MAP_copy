from torch import nn
import torch
import torch.nn.functional as F

class SupConLoss(nn.Module):
    # SupConLoss는 지도형 대조 학습을 위한 손실 함수를 정의하고, SimCLR와 같은 비지도 대조 학습에도 사용할 수 있는 구조
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf. 
    It also supports the unsupervised contrastive loss in SimCLR"""
    # 1. Supervised Contrastive Learning 손실을 구현하며, 데이터 샘플 간의 유사성과 차이를 학습하도록 돕는다, 
    # 주어진 레이블 또는 마스크를 이용해 같은 클래스의 샘플들이 서로 가까워지고, 다른 클래스의 샘플들은 멀어지도록 학습을 유도한다.
    # 레이블 또는 마스크가 주어지지 않을때는 SimCLR 처럼 비지도 대조 손실로 동작할 수 있다. 
    def __init__(self, temperature = 0.07, contrast_mode = 'all'):
        # parameter: 손실 계산 시 온도 매개변수를 사용하여 학습의 안정성을 높인다. 낮은 온도일 수록 샘플 간의 차이가 강조된다.
        # contrast_mode: 'one' 또는 'all' 중 선택하여 특정 앵커(anchor)기준으로 댖고할지 ('one') 모든 샘플을 기준으로 대조할지 ('all')을 결정한다.
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        

    def forward(self, features, labels=None, mask=None):
        # 2. forward 메서드: 입력 매개변수가 features, labels, mask 
        # features: 모델의 출력 벡터로, 각 샘플에 대한 숨겨진 벡터를 나타낸다. 형상은 [batch_size, n_views, ...]이다.
        # mask: 각 샘플 간의 관계를 나태나는 마스크 행렬. mask_{i, j}=1 일 때, 샘플 i와 j가 같은 클래스에 속한다.
        """ Compute loss for model. If both 'labels' and 'mask' are None, 
        it degenerates to SimCLR unsupervised loss: ttps://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """



        device = (torch.device('cuda')
                if features.is_cuda
                else torch.device('cpu'))
        
        if len(features.shape) < 3:
            raise ValueError(' `features` needs tobe [bsz, n_views], ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
           features = features.view(features.shape[0], features.shape[1], -2)

        # 입력처리:
        # - 레이블과 마스크가 둘 다 없는 경우 SimCLR와 유사한 비지도 대조 손실로 처리한다. 
        # - 레이블이 주어질 경우, 동일한 레이블을 가진 샘플 간의 대조관계를 설정하는 마스크를 만든다.
        # - features를 정규화 (F.normalize)하여 벡터 길이를 1로 만들어 코사인 유사도 계싼시 사용한다.

        features = F.normalize(features, dim=2)
        batch_size = features.shape[0]



        # 마스크 생성 및 사용
        # - labels를 통해 대조 마스크를 생성하거나 mask가 제공되면 그대로 사용한다.
        # 마스크를이용해 같은 클래스에 속하는 샘플 간에만 대조 학습이 이루어지도록 제한한다.
        if labels is not None and mask is not None:
            raise ValueError('Cannot defineboth `labels` and `mask` ')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does notmatch numb of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # 3. 대조 학습 계산 - 변수들의 역할들 알아두는게 좋을듯
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature 
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
         # - 유사도 계산: 앵커와 대조 샘플간의 코사인 유사도를 torch.matmul을 사용해 계산하고 self.temperature로 나누어 조정한다, 안정성을 위해 logits에서 최댓값을 뺌으로써 계산의 안정성을 높인다.
        # comptue logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  

        # 자기 대조 방지: logits_mask를 사용해 자신과의 대조는 고려하지 않도록 조정한다.
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1, # 여기서 1과 0이 뭐지..?
            torch.arange(batch_size * anchor_count).view(-1,1).to(device),
            0
        )

        mask = mask*logits_mask

        # 로그 확률 계산: torch.exp()를 사용해 지수값을 계산하고 log_prob를 구해 각 샘플의 로그 확률을 얻는다, 양성 샘플(같은 클래스)의 로그 확률 평균을 계산해 학습에 사용할 손실값을 얻는다.
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask*log_prob).sum(1) / mask.sum(1)

        # 손실계산: 손실(loss)은 양성 샘플에 대한 평균 로그 확률을 부호 반전시켜 계산되며, 이는 높은 유사성을 가진 샘플들 간의 거리를 최소화하도록 한다.
        # loss
        loss = -mean_log_prob_pos   # 뭐 이렇게 생긴 변수가 다 있지?
        loss = loss.view(anchor_count, batch_size).mean()

        # 결과적으로 같은 클래스 샘플은 더 가까워지고 다른 클래스 샘플은 멀어지도록 학습한다.
        return loss
    

# 코드의 핵심 개념:
# 1. Contrastive Learning: 주어진 샘플을 기준으로 같은 클래스의 샘플은 가까워지고 다른 클래스의 샘플은 멀어지도록 하는 학습 방법이다. 이를 통해 모델이 샘플 간의 의미적 관계를 더 잘 학습할 수 있다.
# 2. 온도 조절(temperature): 온도 매개변수를 사용하여 학습 시 코사인 유사도 값의 스케일을 조정해 학습의 안정성을 유지한다.
# 3. 대조 마스크(mask): 동일한 클래스 샘플 간의 관계를 정의하는 데 사용되며, 이를 통해 손실 계산 시 필요한 샘플 쌍을 선택한다.
# 4. 비지도 학습과의 차이: 레이블이 없는 경우 비지도 학습처럼 동작하지만, 레이블이 있는 경우 지도 학습을 수행하여 더 효과적으로 샘플을 구분한다.


# 전체적인 기능:
# 이 클래스는 Supervised Contrastive Learning과 SimCLR 비지도 대조 학습을 모두 지원하며, 
# # 데이터의 숨겨진 벡터 표현 간의 관계를 학습하여 모델의 분류 성능을 향상시킬 수 있다. 특히, 같은 클래스의 샘플을 더 잘 그룹화하고 다른 클래스와 구별되도록 학습하는 데 매우 유용하다.