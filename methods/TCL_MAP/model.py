import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from .SubNets.transformers_encoder.transformer import TransformerEncoder
from .AlignNets import AlignSubNet

class MAG(nn.Module):  # MAG를 여기서 쓰는구나, 다중 모달 데이터를 처리하는 네트워크를 정의. 
    def __init__(self, config, args):
        super(MAG, self).__init__()
        self.args = args

        if self.args.need_aligned:  # self.args.need_aligned가 참이면 텍스트, 비디오, 오디오 데이터를 정렬하는 AlignSubNet을 초기화한다, 이는 데이터의 시간적 정렬 또는 차우너 정렬을 위해 사용될 수 있다.
            self.alignNet = AlignSubNet(args, args.mag_aligned_method)  

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim

        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)  # 비디오와 텍스트를 결합
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)  # 오디오와 텍스트를 결합

        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)  # 비디오 특징을 텍스트 차원으로 매핑하는 선형변환
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)  # 오디오 특징을 텍스트 차원으로 매핑하는 선형변환

        self.beta_shift = args.beta_shift  # 조정 파라미터로 텍스트와 다중 모달 특징 사이의 가중치를 조정하는데 사용됨.

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6  # 작은 값을 설정해 나눗셈에서 0으로 나누는 것을 방지한다.

        if self.args.need_aligned:  # 이게참이면, 텍스트, 비주얼, 오디오 데이터를 정렬한다.
           text_embedding, visual, acoustic = self.alignNet(text_embedding, visual, acoustic)

        weight_v = F.relu(self.W_hw(torch.cat((visual, text_embedding), dim=-1)))  # 텍스트와 비주얼 데이터, 텍스트와 오디오 데이터를 결합해 선형 변환을 수행한 후, ReLU 활성화 함수를 적용하여 가중치 (weight_v, weight_a)를 계산한다.
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)  # 비주얼과 오디오 특징을 텍스트 차원으로 매핑한 결과를 가중치와 곱해 합친 다중 모달 특징이다.
        
        em_norm = text_embedding.norm(2, dim=-1)  # text_embedding과 다중 모달(h_m)의 L2 노름을 계산해 특징의 크기를 측정 
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device) 
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm/(hm_norm + eps)) * self.beta_shift  # 텍스트 임베딩과 다중 모달 특징의 크기 비율과 self.beta_shift를 사용해 임계값을 계산한다. 이는 텍스트와 다중 모달 특징 간의 상호작용을 조절하는 역할을 한다.

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)  

        alpha = torch.min(thresh_hold, ones)  # 임계값과 1 사이의 최솟값을 선택해 다중 모달 특징의 가중치를 제한한다.
        alpha = alpha.unsqueeze(dim=-1)  

        acoustic_vis_embedding = alpha * h_m  # h_m에 alpha를 곱한 결과로 다중 모달 특징을 텍스트 임베딩에 추가하기 전의 결과물이다.

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )  # 다중 모달 특징과 텍스트임베딩을 더한 후 LayerNorm과 dropout을 적용해 최종 출력 임베딩을 생성한다.


        return embedding_output
# 이 MAG 클래스는 텍스트, 오디오, 비디오 데이터를 조합해 하나의 임베딩으로 통합하는 역할을 한다, 
# 이 과정에서 각 모달리티의 특징을 텍스트 차원으로 변환하고, 텍스트 임베딩과의 상호작용을 통해 다중 모달 정보를 합성한다. 
# beta_shift를 통해 텍스트와 다른 모달리티 간의 균형을 조정하며 alignNet을 사용해 필요한 경우 데이터 정렬을 수행한다.

# 이 클래스는 특히 다중 모달 학습에서 중요한 역할을 하는데, 
# 텍스트와 오디오, 비주얼 데이터를 효율적으로 통합하여 모델이 더 풍부한 정보를 학습할 수 있도록 돕는다.


class MAP(BertPreTrainedModel):
    # Transformers 라이브러리의 BertPreTrainedModel을 확장한 모델로, 텍스트, 오디오, 비디오 등여러 모달리티를 통합하여 사용할 수 있게해주는 구성 요소를 추가한 것
    # 이 모델은 projection layer, Attention Mechanism, Prompt기반 접근 방식을 결합하여 다양한 데이터를 통합 처리할 수 있게 한다.
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config
        # embeddings, encoder, pooler는 BERT 모델의 기본 구성 요소를 초기화한다, 
        # 임베딩은 단어 수준의 특징을 캡처하고, 인코더는 시퀀스를처리하며 폴더는 분류 작업을 위한 요약된 출력을 생성한다 
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # fusion layer
        self.MAG = MAG(  # Modality Aware Generation 레이어
            config, args  # 텍스트, 오디오, 비디오 등의 멀티모달 입력을 결합하여 통합 임베딩을 생성하는 층, 
        ) # 이 레이어는 서로 다른 입력을 하나의 임베딩으로 결합한다는데 이게 결합하는 건가..?
        self.args = args

        # MAP module
        self.alignNet = AlignSubNet(args, args.aligned_method)  
        # self.alignNet은 서로 다른 모달리티를 정렬하는 모듈이다. 이 모듈은 설정된 정렬방법 (args.aligned_method)에 따라 오디오, 비디오, 텍스트의 특징들을 조정한다.
        self.embed_dim = args.text_feat_dim
        self.num_heads = args.nheads
        self.layers = args.n_levels
        self.attn_dropout = args.attn_dropout
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask

        # 각각 오디오, 비디오, 텍스트 입력을 공통 임베딩 공간(self.embed_dim)으로 변환하는 nn.Sequential 모델이다.
        # 이 레이어들은 각 모달리티의 차원을 정렬하고 LayerNorm을 통해 정규화한다.
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(args.audio_feat_dim),
            nn.Linear(args.audio_feat_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.video_proj = nn.Sequential(
            nn.LayerNorm(args.video_feat_dim),
            nn.Linear(args.video_feat_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(args.text_feat_dim),
            nn.Linear(args.text_feat_dim, self.embed_dim),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, args.text_feat_dim)
        )
        
        # 시각 및 음성 데이터를 처리하여 context별 임베딩을생성하는 TransformerEncoder이다. 
        self.trans_a_with_l = TransformerEncoder(embed_dim = self.embed_dim,
                                                 num_heads = self.num_heads,
                                                 layers = self.layers,
                                                 attn_dropout = self.attn_dropout,
                                                 relu_dropout = self.relu_dropout,
                                                 embed_dropout = self.embed_dropout,
                                                 attn_mask = self.attn_mask)
        # 이 모듈은 모달리티 간 상호작용을 가능하게 하여, 입력 모달리티 간의 공통 정보를 기반으로 임베딩을세밀하게 조정한다.
        
        self.gamma = nn.Parameter(torch.ones(args.text_feat_dim)*1e-4)
        # 원래의 컨텍스트 임베딩과 trans_a_with_l 레이어에서 생성된 임베딩 사이의 가중치를 조정하는 학습 가능한 파라미터이다.

        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)


    def forward( 
            self, 
            input_ids,  # 텍스트 토큰
            visual, # 시각
            acoustic,  # 오디오 특징
            condition_idx,
            ctx,  # ctx는 뭐임?
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,  # BERT 관련 입력을 처리한다.
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            output_attentions = None,  # 이 코드가 왜 안쓰이지?
            output_hidden_states = None,
    ):
        r"""
    Return:
        :obj:'tuple(torch.FloatTensor)' comprising various elements dpending on the configuration (:class: '~transformers.BertConfig')
        last_hidden_state (:obj: 'torch.FloatTensor' ofshpae : obj:'(batch_size, sequence_length, hidden_size)'):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:'torch.FloatTensor': of shape :obj:'(batch_size, sequence_length, hidden_size)'):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tahn activation function. The Linear layer weights are trained from the next sentence prediction (classification) objective during pre_training.

            This output is usually *not* a good summary of the semantic content of the input, you're often better with averaging or pooling the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:'tuple(torch.FloatTensor)', 'optional', returned when "output_hidden_states=True" is passed or when "config.output_hidden_states=True"):
            Tuple of :obj:'torch.FloatTensor' (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:'(batch_size, sequence_length, hidden_size)'.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:'tuple(torch.FloatTensor)', 'optional', returned when "output_attentions=True" is passed or when "config.output_attentions=True''):
            Tuple of :obj:'torch.FloatTensor' (one for each layer) of shape
            :obj:'(batch_size, num_heads, sequence_length, sequence_length)'.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.  
        """

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need tomake it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # We need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask =self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        # get embeddings of normal samples
        embedding_output = self.embeddigns(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            inputs_embeds = inputs_embeds,
        )

        # generate and employ modality-aware prompt - 모달리티aware 프롬프트 생성
        batch_ctx = ctx.unsqueeze(0).repeat(acoustic.shape[0], 1, 1)
        # batch_ctx는 입력 크기에 맞게 컨텍스트 임베딩을 반복 생성하며, 이는 정렬된 시각 및 음성 데이터와 함께 사용된다.
        _, aligned_visual, aligned_acoustic = self.alignNet(batch_ctx, visual, acoustic)
        aligned_acoustic = self.audio_proj(aligned_acoustic) 
        aligned_visual = self.video_proj(aligned_visual)
        batch_ctx = self.text_proj(batch_ctx)
        generated_ctx = self.trans_a_with_l(batch_ctx.permute(1, 0, 2), aligned_visual.permute(1, 0, 2), aligned_acoustic.permute(1, 0, 2)).permute(1, 0, 2)
        # 시각 및 음성 입력을 처리하여 generated_ctx라는새로운 컨텍스트를 생성하며, 이는 self-attention을사용해 생성된 것이다.
        generated_ctx = batch_ctx + self.out_projc(generated_ctx) * self.gamma
        # self.gamma를 이용해 원래 텍스트 임베딩과 합쳐지며, 지정된 위치(condition_idx)의 토큰 시퀀스에 적용된다. 
        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i]- self.args.prompt_len: condition_idx[i], :] = generated_ctx[i]

        # Early fusion with MAG - MAG와의 결합
        fused_embedding = self.MAG(embedding_output, visual, acoustic)
        # 통합된 임베딩(텍스트, 시각 및 오디오 정보를 포함)은 MAG 레이어를 통해 더욱 정제된 표현으로 변환된다.

        # refine tokens of normal smaples
        encoder_outputs = self.encoder( # self.encoder로 전달되어 더 싶은 표현 학습을 수행한다.
            fused_embedding, 
            attention_mask = extended_attention_mask,
            head_mask = head_mask,
            encoder_hidden_states = encoder_hidden_states,
            encoder_attention_mask = encoder_extended_attention_mask,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        sequence_output = encoder_outputs[0]  # sequence output, 각 토큰의 숨겨진 상태 와 pooled_output(시퀀스 요약)이 생성된다.
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ] # add hidden_states and attentions if they are here
        # sequence_output, pooled_output, (hidden_states), (attentions)

        # 이 메서드는시퀀스 및 풀링된 출력, 숨겨진 상태, 어텐션 가중치 등을 반환하여 후속 작업에 필요한 다양한 표현을 제공한다.

        return outputs, generated_ctx
# Context (ctx): ctx는 프롬프트를 생성할 때 사용되는 초기 컨텍스트 임베딩을 의미한다. 입력 모달리티에 따라 프롬프트를조정한느 데 핵심적인 역할을 한다.
# Attention-Mask: 시퀀스의 패딩 토큰을 처리하는데 사용, 또한 모델이 sequence-to-sequence방식 (self.config.is_decoder)으로 사용될 경우 크로스 어텐션을 처리할 수 있도록 설계되었다.
# 이 모델은 텍스트, 시각, 음성 데이터를 결합하여 BERT의 언어 모델링 기능과 다른 모달리티의 특징을 통합 처리하는 멀티모달 작업에 최적화되어 있다.

class MAP_Model(BertPreTrainedModel):  # 멀티모달 입력을 처리하기 위한 BERT 기반의 모델로, 텍스트, 시각, 음향 데이터를 통합하여 분류 또는 회귀 작업을 수행할 수 있도록 설계되었다. 
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = args.num_labels
        self.label_len = args.label_len

        self.bert = MAP(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)  # 드롭아웃 레이어와 분류를 위한 self.classifier를 추가하여 모델의 출력을 조정한다.

        self.init_weights()

    def forward(  # 모델의 주요 처리 루틴을 정의한다. 
        self,
        text,
        visual,
        acoustic,
        condition_idx,  # 각 샘플에서 특정 조건에 대한 임베딩을 생성한다. 
        ctx,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        labels = None,  # 손실 함수를 정의하여 회귀 또는 분류 문제에 따라 적절한 손실을 계산한다.
        output_attentions = None,
        output_hidden_states = None,
    ):
        r"""
        labels (:obj:'torch.LongTensor' of shape :obj:'(batch_size,)', 'optional', defaults to :obj:'None'):
            Labels for computing the sequence classification/regression loss.
            Indicies should be in :obj:'[0, ..., config.num_labels -1]'.
            If :obj:'config.num_labels == 1' a regression loss is computed (Mean-Square loss),
            If :obj:'config.num_labels > 1' a classification loss is computed (Cross-Entropy).
        
    Returns:
        :obj:'tuple(torch.FloatTensor)' comprising various elements depending on the configuration(:class'~transformers.BertConfig') and inputs:
        loss (:obj:'torch.FloatTensor' of shape :obj'(1,)' 'optional', returned when :obj:'label' is provided):
            Classification (or regression if config.num_lavels==1) loss.
        logits (:obj:'torch.FloatTensor' of shape :obj:'(batch_size, config.num_labels)'):
            Classification (or regression if config.num_labels ==1) scores (before SoftMax).
        hidden_states (:obj:'tuple(torch.FloatTensor)', 'optional', returned when ''output_hidden_states=True'' is passed or when ''config.output_hidden_states=True''):
            Tuple of :obj:'torch.FloatTensor' (one forthe output of the embeddings + one for the output of each layer)
            of shape :obj:'(batch_size, sequence_length, hidden_size)'.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:'tuple(torch.FloatTensor)', 'optional', returned when ''output_attentions=True'' is passed or when ''config.output_attentions=True''):
            Tuple of :obj:'torch.FloatTensor' (one for each layer) of shape
            :obj:'(batch_size, num_heads, sequence_length, sequence_length)'.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        """

        input_ids, attention_mask, token_type_ids = text[:, 0], text[:, 1], text[:, 2]

        outputs, generated_ctx = self.bert(
            input_ids,
            visual,
            acoustic,
            condition_idx,
            ctx,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids, 
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
        )

        sequence_output = outputs[0]
        condition_tuple = tuple(sequence_output[torch.arange(sequence_output.shape[0]), condition_idx.view(-1) +i, :].unsqueeze(1) for i in range(self.label_len))
        # condition_idx로 생성된 조건임베딩이, 여기서 condition_tuple을 통해 이뤄진다, 텍스트의 특정 위치에서 정보가 추출된다.
        # 이 임베딩은 후에 계속 작업된다.
        condition = torch.cat(condition_tuple, dim=1)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[
            2:
        ] # add hidden states and attention if they are here

        if labels is not None:  # 손실 함수를 정의하여 회귀 또는 분류 문제에 따라 적절한 손실을 계산한다.
            if self.num_labels == 1:  
                # we are doing regression
                loss_fct = MSELoss()  # 회귀인 경우 MSE(Mean Squared Error)손실을,
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()  # 분류인 경우 Cross-Entropy 손실을 사용한다.
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1)
                )
            outputs = (loss,) + outputs
        # 모델의 출력은 logit, 풀링된 출력, 조건 임베딩 및 생성된 컨텍스트 generated_ctx로 구성된다, 
        # 추가로 숨겨진 상태나 attention 정보도 포함 될 수 있다. 
        return outputs, pooled_output, condition, generated_ctx
# condition_idx가 킥이거든요~, 특정 조건에 대한 임베딩을 효율적으로 추출할 수 있고, 이를 통해 모델의 표현력을 높일 수 있다.
# 여기까지 보면서 알 수 있는 점이, 멀티모달 입력을 사용하는 자연어처리를 수행하는 거 같다. 
# 예를 들어, 영상과 오디오에서 관련된 정보를 텍스트와 결합하여 감정 분석이나 분류 작업을 수행 할 수 있다.

class Cons_Model(BertPreTrainedModel):  # BERT 기반의 모델로, 인코더와 디코더 역할을모두 수행할 수 있도록 설계되어 있음
    # BertPreTrainedModel을 상속받아 초기화한다.
    """
    The model can behave as an encoder (with only self_attention) as well as a decoder, in which case a layer of cross-attention is added
    between the self-attention layers, following the architecture described in [Attention is all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the 'is_decoder' argument of the configuration set to 'True'.
    To be used in a Seq2seq model, the model needs to be initialized with both 'is_decoder' argument and 'add_cross_attention' set to 'True';
    an 'encoder_hidden_states' is then expected as an input to the forward pass.
    """ 

    def __init__(self, config, args, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.emebeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        # BERT임베딩과 인코더를 포함하여 모델 구조를 설정한다.

        
        self.pooler = BertPooler(config) if add_pooling_layer else None  # 선택적으로 pooling layer(BertPooler)를 추가할수 있다. 이를 통해 최종 출력의차원을 줄일 수 있다.
        self.args = args  # args 매개변수를 사용하여 추가적인 파라미터를 설정한다.
        # Initialize weights and apply final processing
        self.post_init()

    # 입력 임베딩 처리
    # get_input_embeddings() 및 set_input_embeddings(value) 메서드를 통해 모델의 입력 임베딩을 가져오거나 설정 할 수 있다.
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self,value):
        self.embeddings.word_embeddings = value

    # head pruning - 메서드는 특정 레이어에서 불필요한 어텐션헤드를 제거하여 모델의크기를 줄이고 성능을 개선하는데 도움을 준다.
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of{layer_num: list of heads to prune in this layer} 
        See base class PreTrainedModel        
        """

        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 이 메서드는 모델의 주요 처리로직을 정의한다, 여러 입력을 처리하며, 인코더와 디코더의 역할을 구분하여 동작할 수 있다.
    def forward(
        self,
        condition_idx,
        ctx,
        # 다양한 입력 인자들을
        input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        position_ids = None,
        head_mask = None,
        inputs_embeds = None,
        encoder_hidden_states = None,
        #을 받아 처리한다.

        encoder_attention_mask = None,
        past_key_values = None,
        use_cache = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ):
        
        r"""
        encoder_hidden_states ('torch.FloatTensor' of shape '(batch_size, sequence_length, hidden_size)', *optional*):
            Sequence of hidden_states at the output of the last layer of the encoder. Used in the cross-attention if the model is configured as a decoder.
        encoder_attention_mask ('torch.FloatTensor' of shape '(batch_size, sequence_length)', *optional*):
            Mask to avoid performing attention on the padding token indicies of the encoder input. This mask is used inthe cross-attention if the model is configured as a decoder.
            Mask values selected in '[0, 1]':
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **maksed**.
            past_key_values('tuple(tuple(torch.FloatTensor))' of length 'config.n_layers' with each tuple having 4 tensors of shape '(batch_size, num_heads, sequence_length -1, embed_size_per_head)'):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

                If 'past_key_values' are used, the user can optionally input only the last 'decoder_input_ids' (those that don't have their past key value states given to this model) of shape
                '(batch_size, 1)' instead of all 'decoder_input_ids' of shape ' (batch_size, sequence_length)'.
            use_cache ('bool', *optional*):
                If set to 'True', 'past_key_values' by value states are returned and can be usedto speed up decoding (see 'past_key_values').             
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length+past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        # get_extended_attention_mask() 메서드를 사용하여 어텐션 마스크를 확장하고 필요에 따라 크로스 어텐션 마스크를 처리한다, 
        # 크로스 어텐션은 모델이 인코더의 출력을 참조할 수 있도록 하여 디코더가 필요한 정보를 효율적으로 가져올 수 있도록 한다.

        # If a 2D or 3D attention mask is provided forthe cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shpae bsz x n_heads x N x N
        # input head_mask has shpae [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # get embeddings of augmented samples
        embedding_output = self.embeddings(
            input_ids = input_ids,
            position_ids = position_ids,
            token_type_ids = token_type_ids,
            inputs_embeds = inputs_embeds,
            past_key_values_length = past_key_values_length,
        )

        # employ modality-aware prompt
        for i in range(embedding_output.shape[0]):
            embedding_output[i, condition_idx[i] - self.args.prompt_len: condition_idx[i], :] = ctx[i]

        # refine tokens with BERT encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask = extended_attention_mask,
            head_mask = head_mask,
            encoder_hidden_states = encoder_hidden_states,
            past_key_values = past_key_values,
            use_cache = use_cache, 
            output_attentions = output_attentions,
            output_attentions = output_hidden_states,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        # pooler 메서드를 쓰다니, BertPooler에서 처음 쓰임

        if not return_dict:  # 선택적으로 return_dict 매개변수에 따라 출력 형식을 조정할 수 있다.
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        
        return BaseModelOutputWithPoolingAndCrossAttentions( # 최종 출력은 이 형태로 반환된다. 이는 인코더의 마지막 숨겨진 상태, 풀링된 출력, 과거 key-value 쌍, 숨겨진 상태 및 attention 정보 등을 포함한다.
            last_hidden_state = sequence_output,
            pooled_output = pooled_output,
            past_key_values = encoder_outputs.past_key_values,
            hidden_states = encoder_outputs.hidden_states,
            attentions = encoder_outputs.attentions,
            cross_attentions = encoder_outputs.cross_attentions,
        )
# Cons_Model은 자연어 처리(NLP) 분야에서 Seq2Seq 작업(ex, 기계 번역, 텍스트 요약 등)에 적합함. 여기서는 텍스트 요약으로 보여짐. 
# # 인코더와 디코더 역할을 동시에 수행할 수 있음, BERT기반 모델
    
class TCL_MAP(nn.Module):  # 여기서 MAP_model과 Cons_Model을 결합하여 멀티모달 데이터를 처리하는 모델임.
    def __init__(self, args):

        super(TCL_MAP, self).__init__()
        # MAP_Model과 Cons_Model의 인스턴스를 초기화. 각각의 모델은 사전 훈련된 버전을 로드하며, 캐시 경로와 인자(args)를 받는다.
        self.model = MAP_Model.from_pretrained(args.text_backbone, cache_dir = args.cache_path, args=args)
        self.cons_model = Cons_Model.from_pretrained(args.text_backbone, cache_dir = args.cache_paths, args = args)
        # 컨텍스트 벡터를 초기화하는 _init_ctx 메서드를 호출하여 임베딩의 초기값을 설정한다.
        self.ctx_vectors = self._init_ctx(args)
        
        # MAP_Model: 일반 샘플을 처리할 때는 이 모델을 통해 텍스트, 비디오, 오디오 피처를 통합하여 출력한다.
        # Cons_Model: 증강 샘플의 경우, 이 모델을 사용하여 보조 텍스트 피처를처리하고, 마지막 숨겨진 상태 (last_hidden_state)를 가져온다.
        # 조건 인덱스를 사용하여 cons_condition을 생성하고, 이를 통해 최종출력을 형성한다.

    def _init_ctx(self, args):  # context 초기화
        ctx = torch.empty(args.prompt_len, args.text_feat_dim, dtype=torch.float)  # 주어진 args의 파라미터를 바탕으로 임베딩 텐서를 초기화한다. 
        nn.init.trunc_normal_(ctx)  # trunc_normal_을 사용하여 정규 분포를 기반으로 값을 설정한다.
        return ctx
    
    def forward(self, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):  # 전달하는 forward 메서드
        video_feats = video_feats.float()
        audio_feats = audio_feats.float()

        # process normal sample - 여러 입력을 처리
        outputs, pooled_output, condition, generated_ctx = self.model(
            test = text_feats,
            visual = video_feats,
            audio = audio_feats,
            condition_idx = condition_idx,
            ctx = self.ctx
        )

        # process augmented sample
        cons_input_ids, cons_input_mask, cons_segment_ids = cons_text_feats[:, 0], cons_text_feats[:,1], cons_text_feats[:, 2]
        cons_outputs = self.cons_model(
            input_ids = cons_input_ids,
            condition_idx = condition_idx,
            ctx = generated_ctx,
            token_type_ids = cons_segment_ids,
            attention_mask = cons_input_mask
        )
        last_hidden_state = cons_outputs.last_hidden_state
        cons_condition_tuple = tuple(last_hidden_state[torch.arrange(last_hidden_state.shape[0]), condition_idx.view(-1) + i, :].unsqueeze(1))
        cons_condition = torch.cat(cons_condition_tuple, dim =1)

        # return classification feature and Label/[MASK] token representation

        # 밑에 4가지 형태로 여러 결과를 반환한다. 이는 각각의 모델에서 생성된 결과로, 텍스트와 멑티모달 피처의 통합된 정보를 제공한다. 
        return outputs[0], pooled_output, condition.mean(dim=1), cons_condition.mean(dim=1)
# TCL_MAP 클래스는 텍스트, 비디오, 오디오 데이터가 결합된 멀티모달 학습에 적합하다. 예를 들어 영상분석, 감정인식 또는 텍스트-비주얼 관련 작업에서 사용할수 있다.


# 입력 데이터는 모두 float 형식으로 변환되어야 하며, 특정 인덱스에 따라 조건을 설정할 수 있다. 
# 조건 인덱스와 컨텍스트 벡터가 중요한 역할을 하므로, 올바르게 설정하는 것이 필요하다.
# 이 클래스는 멀티모달 데이터의 상호작용을 극대화하며, 특정 조건에 맞게 피처를 생성하는데 유용하다. 다양한 모달리티에서 정보를 통합하여 보다 풍부한 표현을 가능하게 하다.