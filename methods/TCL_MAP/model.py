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

class MAG(nn.Module):
    def __init__(self, config, args):
        super(MAG, self).__init__()
        self.args = args

        if self.args.need_aligned:
            self.alignNet = AlignSubNet(args, args.mag_aligned_method)

        text_feat_dim, audio_feat_dim, video_feat_dim = args.text_feat_dim, args.audio_feat_dim, args.video_feat_dim

        self.W_hv = nn.Linear(video_feat_dim + text_feat_dim, text_feat_dim)
        self.W_ha = nn.Linear(audio_feat_dim + text_feat_dim, text_feat_dim)

        self.W_v = nn.Linear(video_feat_dim, text_feat_dim)
        self.W_a = nn.Linear(audio_feat_dim, text_feat_dim)

        self.beta_shift = args.beta_shift

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6

        if self.args.need_aligned:
           text_embedding, visual, acoustic = self.alignNet(text_embedding, visual, acoustic)

        weight_v = F.relu(self.W_hw(torch.cat((visual, text_embedding), dim=-1)))
        weight_a = F.relu(self.W_ha(torch.cat((acoustic, text_embedding), dim=-1)))

        h_m = weight_v * self.W_v(visual) + weight_a * self.W_a(acoustic)
        
        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm/(hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(text_embedding.device)

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )


        return embedding_output
    

class MAP(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        # fusion layer
        self.MAG = MAG(
            config, args
        )
        self.args = args

        # MAP module
        AlignSubNet(args, args.aligned_method)