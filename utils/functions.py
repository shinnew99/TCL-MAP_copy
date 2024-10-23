import os
import torch
import numpy as np
import pandas as pd
import random
import logging   # logging이라는 default 함수도 있군
import copy
from .metrics import Metrics

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. """
    def __init__(self, args, delta = 1e-6):
        '''
        Args:
            patience(int): How long to wait after last time validation loss improved.
            delta(float): Minimum change in the monitored quantity to qulify as an improvement.
        '''
        self.patience = args.wait_patience  # 학습이 멈추기 전 기다리는 시간을 설정한다.
        self.logger = logging.getLogger(args.logger_name)  # 
        self.monitor = args.eval_monitor  # eval_monitor  
        self.counter = 0
        self.best_socre = 1e8 if self.monitor == 'loss' else 1e-6  # 개선 여부를 판단하기 위한 기준점이다, 'loss'의 경우 낮을 수록 좋기 때문에 초기 값은 매우 큰 수로 설정되며, 'accuracy'와 같은 경우 초기 값은 매우 작은 수로 설정된다.
        self.early_stop = False
        self.delta = delta  # EarlyStopping에 delta가 있었나..?, 개선 여부를 판단할 최소 변화량이다.
        self.best_model = None

    def __call__(self, score, model):

        better_flag = score <= (self.best_score - self.delta) if self.monitor == 'loss' else score >= (self.best_score +self.delta)
        # 성능이 개선되었는지를 확인하고, 개선되지 않았을 때 counter를 증가시킨다. counter가 patience보다 커지면 조기 종료를 활성화한다.
        # 모델이 개선되었을 때 best_model에 현재 모델 상태를 저장한다.

        if better_flag:  # better_flag를 통해 성능이 개선되었는지를 확인하고, 개선되지 않았을 때 counter를 증가시킨다. counter가 patience보다 커지면 조기 종료를 활성화한다.
            self.counter = 0
            self.best_model = copy.deepcopy(model)
            self.best_scores =score

        else:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: (self.counter) out of (self.patience)')

            if self.counter >= self.patience:
                self.early_stop = True
    
    def set_torch_seed(seed):  # random, numpy, torch의 시드를 고정하여 코드 실행 결과의 일관성을 보장한다.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        # random, numpy, torch의 시드를 고정하여 코드 실행 결과의 일관성을 보장한다.
        # CUDA와 관련된 설정들도 포함되어 있어, GPU 환경에서도 동일한 결과를 재현할 수 있게 한다.

    def set_output_path(args, save_model_name):  # 모델의 결과를 저장할 디렉토리를 생성한다, 모델파일을 저장할 model_path
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

        pred_output_path = os.path.join(args.output_path, save_model_name)
        if not os.path.exists(pred_output_path):
            os.makedirs(pred_output_path)
        
        model_path = os.path.join(pred_output_path, args.model_path)
        if not os.pth.exists(model_path):
            os.makedirs(model_path)
        
        return pred_output_path, model_path
    
    def save_npy(npy_file, path, file_name):  # npy 파일을 save하고 load 한다는건.. npy로 이루어진 데이터셋이 뭐가있지..? 오디오인가..?
        npy_path = os.path.join(path, file_name)
        np.save(npy_path, npy_file)

    def load_npy(path, file_name):
        npy_path = os.path.join(path, file_name)
        npy_file = np.load(npy_path)
        return npy_file
    
    def save_model(model, model_dir):  # 모델의 가중치(state_dict)를 저장하고 불러온다. 
        save_model = model.module if hasattr(model, 'module') else model  # model.module 속성을 사용해 병렬 학습 시에도 정상적으로 모델을저장할 수 있도록 처리한다.
        model_file = os.path.join(model_dir, 'pytorch_model.bin')
        
        torch.save(save_model.state_dict(), model_file)

    def restore_model(model, model_dir, device):
        output_model_file = os.path.join(model_dir, 'pytorch_model.bin')
        m = torch.load(output_model_file, map_location = device)
        model.load_state_dict(m)
        return model
    
    def save_results(args, test_results, debug_args=None):  # 결과 저장 및 평가
        save_keys = ['y_pred', 'y_true', 'features', 'scores']
        for s_k in save_keys:
            if s_k in test_results.keys():
                save_path = os.path.join(args.output_path, s_k + '.npy')  # 테스트 결과를 .npy 파일로 저장함, 여기서 .npy 포맷은 오디오 값인듯?
                np.save(save_path, test_results[s_k]) # Metrics 클래스를 통해 다양한 평가 지표를 계산하여 results딕셔너리에 저장한다.

        results = {}
        metrics = Metrics(args)

        for key in metrics.eval_metrics:
            if key in test_results.keys():
                results[key] = round(test_results[key]*100, 2)
        
        if 'best_eval_score' in test_results:
            eval_key = 'eval_' + args.eval_monitor 
            results.update({eval_key: test_results['best_eval_score']})

        _vars = [args.dataset, args.method, args.text_backbone, args.seed, args.log_id]
        _names = ['dataset', 'method', 'text_backbone', 'seed', 'log_id']

        if debug_args is not None:
            _vars.extend([args[key] for key in debug_args.keys()])
            _names.extend(debug_args.keys())

        vars_dict = {k:v for k,v in zip(_names, _vars)}
        results = dict(results, **vars_dict)

        keys = list(results.keys())
        values = list(results.values())

        if not os.path.exists(args.results_path):
            os.makedirs(args.results_path)

        results_path = os.path.join(args.results_path, args.results_file_name)

        if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:  # 결과 파일이 이미 존재하면, 새로운 결과를 추가로 덧붙여 csv 형식으로 저장한다.
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)  # pd.DataFrame을 사용해 csv 파일에 결과를 저장하며, 만약 파일이 비어 있으면 새로운 파일을 생성해 기록한다.
            df1.to_csv(results_path, index=False)

        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)



# 주요 부분:
# self.delta: 학습이 개선되었는지를 판단할 때 사용하는 작은 값으로, 변화량이 이 값 이상일 때만 개선으로 간주한다.
# 로그 출력: logging 모듈을 사용해 조기 종료 시 로그를 남긴다.
# 모델 관리 및 재현성: 시드를 고정하거나 모델 상태를 복원하는 방식은 모델의 성능 재현성을 확보하는 데 중요한다.

# 의문점 및 추가 설명
# delta의 역할: 아주 작은 개선도 고려할지, 아니면 어느 정도의 차이가 있어야만 개선으로 볼지를 결정하는 변수이다.
# 로그 및 디렉터리 생성 부분: 모델 결과의 저장 경로와 로깅 설정이 사용자가 실험을 관리하는 데 유용하게 작동한다.
# 이 코드는 학습 중 모델 성능 개선 여부를 모니터링하고, 최적의 모델 상태를 저장하여 학습이 과도하게 오래 걸리거나 과적합되는 것을 방지하는 데 유용한 구조를 갖추고 있다.