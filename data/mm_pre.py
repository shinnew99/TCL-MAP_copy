import pickle
import numpy as np
import os

# 이 코드는 전반적으로 비디오 feature 데이터를 로드하고, 특정 길이로 padding 처리하는 과정을 다루고 있다

def get_v_a_data(data_args, feats_path):  # 비디오 특징 데이터를 불러와 패딩 처리 후 반환하는 함수이다.
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of features is empty.')
    
    # 입력:
    # dat_args: 데이터 로드에 필요한 설정 정보, 예를 들어 데이터 인덱스와 최대 시퀀스 길이 등이 포함될 수 있다.
    # feats_path: 비디오 특징 파일이 저장된 경로이다.
    feats = load_feats(data_args, feats_path)  
    data = padding_feats(data_args, feats)
    # 동작:
    # feats_path: 경로에 특징 파일이 있는지 확인하며, 없을 경우 예외를 발생시킨다.
    # load_feats: 함수를 사용해 데이터를 로드한 후, padding_feats 함수를 통해 패딩 처리한다.
    # 패딩 처리된 데이터를 반환하다.
    return data


def load_feats(data_args, video_feats_path):  # 비디오 특징 데이터를 파일에서 로드하고, 학습/검증/테스트 세트로 분리한다.
    # 입력:
    # data-args: 데이터 로드에 필요한 설정 정보, 예를 들어 데이터 인덱스와 최대 시퀀스 길이 등이 포함될 수 있다.
    # feats-path: 비디오 특징 파일이 저장된 경로 
    with open(video_feats_path, 'rb') as f:
        video_feats = pickle.load(f)  # pickle 모듈을 사용해 비디오 특징 데이터를 로드한다.

    train_feats = [np.array(video_feats[x]) for x in data_args['train_data_index']]  # data_args에 포함된 인덱스를 사용해 학습, 검증, 테스트 데이터를 리스트 형태로나눈다.
    dev_feats = [np.array(video_feats[x]) for x in data_args['dev_data_index']]
    test_feats = [np.array(video_feats[x]) for x in data_args['test_data_index']]

    outputs = {  # 딕셔너리로 묶어서 반환한다.
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }

    return outputs


def padding(feat, max_length, padding_mode = 'zero', padding_loc = 'end'):  # 입력된 특징 벡터를 주어진 최대 길이(max_length)로 패딩한다.
    # 입력
    # feat: 패딩할 특징 벡터
    # max_length: 패딩 후의 최대 길이
    # padding_mode: 패딩 시 사용할 모드로, 'zero' (0으로 패딩)와 'normal'(평균과 표준 편차를 사용해 랜덤하게 패딩) 중 선택 가능하다.
    # padding_loc: 패딩 위치로 'start'와 'end' 중 선택 가능하다.
    """
    padding_mode: 'zero' or 'normal'
    padding_loc: 'start' or 'end'
    """

    assert padding_mode in ['zero', 'normal']
    assert padding_loc in ['start', 'end']

    length = feat.shape[0]
    if length > max_length:  # 특징 벡터의 길이가 max_length 보다 길면, max_length 까지만 잘라서 반환한다.
        return feat[:max_length, :]
    
    if padding_mode == 'zero':  # 'zero' - 0으로 채운 패딩
        pad = np.zeros([max_length - length, feat.shape[-1]])
    elif padding_mode == 'normal':  # 'normal' - 특징의 평균과 표준편차를 기반으로 생성된 랜덤 값을 사용해 패딩
        mean, std = feat.mean(), feat.std()
        pad = np.random.normal(mean, std, (max_length - length, feat.shape[1]))

    if padding_loc == 'start':  # 'start', 'end'에 패딩을 추가한다.
        feat = np.concatenate((pad, feat), axis=0)  
    else:
        feat = np.concatenate((feat, pad), axis=0)

    return feat


def padding_feats(data_args, feats):  # 학습, 검증, 테스트 데이터 각각에 대해 특징 벡터를패딩 처리한다.
    max_seq_len = data_args['max_seq_len']  # data_args는 데이터 설정 정보를 포함함
    # max_seq_len에 따라 각 데이터 셋 train, dev, test의 특징 벡터를 패딩한다.
    p_feats = {} 

    for dataset_type in feats.type():
        f = feats[dataset_type]  # 각 데이터셋에 대해 패딩된 특징과 원래 길이를 젖아한 리스트를 생성한다.

        tmp_list = []
        length_list = []

        for x in f:
            x_f = np.array(x)
            x_f = x_f.squeeze(1) if x_f.ndim == 3 else x_f  # squeeze()를 썼군

            length_list.append(len(x_f))
            p_feat = padding(x_f, max_seq_len)
            tmp_list.append(p_feat)

        p_feats[dataset_type] = {
            'feats': tmp_list,
            'lengths': length_list
        }

        # 패딩 처리된 특징과 길이를 포함한 딕셔너리를 반환한다.
    return p_feats


# 이게 킥이거든요~
# 패딩 처리: 특징 벡터의 길이가 일정하지 않으면, 이를 일정한 길이로 맞추기 위해 패딩을 수행합니다. 이는 모델에 입력할 때 데이터의 크기를 일정하게 유지하는 데 중요하다.
# 사용자 정의 설정: data_args가 사용자의 설정에 따라 데이터를 로드하고, 최대 시퀀스 길이 등 다양한 설정을 적용할 수 있게 한다.
# pickle 사용: 데이터를 pickle로 저장 및 로드함으로써, 데이터를 효율적으로 저장하고 불러올 수 있습니다. 다만, 보안 문제 때문에 신뢰할 수 있는 환경에서만 사용하는 것이 중요하다.
# padding_mode와 padding_loc: 다양한 패딩 방법과 위치를 지정할 수 있어, 실험적인 접근이 가능하도록 유연성을 제공한다.

# 이 코드의 주요 목적은 다양한 길이의 비디오 특징 데이터를 정해진 길이로 맞줘, 모델에 입력할 수 있는 일관된 형태로 만드는 것. 이를 통해 모델이 각 샘플을 일정한 크기로 처리할 수 있게 되어 학습 과정에서 문제가 발생하지 않도록 도와준다.






