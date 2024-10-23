import os
import csv
import sys
import pickle
from transformers import BertTokenizer
import numpy as np

def get_t_data(args, data_args):
    t_data, cons_text_feats, condition_idx = get_data(args, data_args)
    return t_data, cons_text_feats, condition_idx


def get_data(args, data_args):

    processor = DatasetProcessor(args)
    



def get_backbone_feats(args, data_args, examples):



class InputExample(object):


class InputFeatures(objects):

class DatasetProcessor(DataProcessor):
    def __init__(self, args):


    def get_examples(self, data_dir, mode):


    
