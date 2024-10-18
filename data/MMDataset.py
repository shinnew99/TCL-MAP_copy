from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ["MMDataset"]

class MMDataset(Dataset):
    def __init__(self, label_ids, text_feats, video_feats, audio_feats, cons_text_feats, condition_idx):

        self.label_ids = label_ids
        self.text_feats = text_feats
        self.cons_text_feats = cons_text_feats
        self.condition_idx = condition_idx
        self.video_feats = video_feats