from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, auc, precision_recall_curve, roc_curve

from scipy.optimize import brentq
from scipy.interpolate import interp1d

import logging
import numpy as np

class AverageMeter(object): # 학습 중 성능 지표의 평균 값을 계산하고 저장하는 유틸리티 클래스
    # 이 클래스는 일반적으로 학습 중의 손실(loss)나 정확도(acc)를 추적하는 데 사용된다.
    """ Computes and stores the average and current value """
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):  # 새로운 값을 추가하고, 합계(sum)와 횟수(count)를 업데이트하여 평균(avg)을 다시 계산한다.
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = float(self.sum)/self.count

class Metrics(object):  # 모델의 성능을 평가하는 메트릭들을 계산하고 필요에 따라 결과를 로그로 출력할 수 있도록 지원하는 클래스이다.
    """ column of confusion matrix: predicted index row of confusion matrix: target index """
    def __init__(self, args): # args인자로 로거(logger)를 설정하고 평가할 메트릭들의 목록을 정의한다.
        self.logger = logging.getLogger(args.logger_name)
        self.eval_metrics = ['acc', 'f1', 'prec', 'rec', 'weighted_f1', 'weighted_prec', 'weighted_rec' ]

    def __call__(self, y_true, y_pred, show_results = False):  # 모델 예측값(y_pred)과 실제값(y_true)을 입력받아, 여러 메트릭을 계산하여 결과를 딕셔너리 형태로 반환한다.
        acc_score = self._acc_score(y_true, y_pred)
        macro_f1, weighted_f1 = self._precision_score(y_true, y_pred)
        macro_prec, weighted_prec = self._precision_score(y_true, y_pred)
        macro_rec, weighted_rec = self._recall_score(y_true, y_pred)
        
        # 전통적인 효과지표들을 썼군요
        eval_results = {
            'acc': acc_score,
            'f1': macro_f1,
            'weighted_f1': weighted_f1,
            'prec': macro_prec,
            'weighted_prec': weighted_prec,
            'rec': macro_rec,
            'weighted_rec': weighted_rec
        }


        if show_results:
            self._show_confusion_matrix(y_true, y_pred)  # true인 경우 이를 출력

            self.logger.info("***** In-domain Evaluation results *****")
            for key in sorted(eval_results.keys()):
                self.logger.info(" %s=%s", key, str(round(eval_results[key], 4)))

        return eval_results


    # 각각 정확도, f-1스코어, 정밀도, 재현율을 계산한다.
    def _acc_score(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def _f1_score(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average='macro'), f1_score(y_true, y_pred, average='weighted')
    
    def _precision_score(self,y_true, y_pred):
        return precision_score(y_true, y_pred, average='macro'), precision_score(y_true, y_pred, average='weighted')

    def _recall_score(self, y_true, y_pred):
        return recall_score(y_true, y_pred, average='macro'), recall_score(y_true, y_pred, average='weighted')
    
    def _show_consufion_matrix(self, y_true, y_pred):
        # y_true와 y_pred를 기반으로 혼동 행렬을 계산하고 이를 로그에 출력한다.
        cm = confusion_matrix(y_true, y_pred)  # confusion_matrix은 모델이 각 클래스에서 예측한 결과와 실제 결과의 상관관계를 시각적으로 확인할 수 있는 도구이다. 
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))

# accuracy_score: 전체 샘플 중 올바르게 분류된 샘플의 비율을 계산한다.
# f1_score: 정밀도와 재현율의 조화 평균을 사용해, 클래스 간 불균형 문제를 보완한다.
# precision_score: 모델이 '양성'으로 예측한 샘플 중 실제로 양성인 비율을 계산한다.
# recall_score: 실제 '양성' 샘플 중 모델이 정확하게 예측한 비율을 계산한다.
# confusion_matrix: 예측 클래스와 실제 클래스의 빈도수를 나타내는 행렬이다.

# 주요 부분:
# 로그 출력: 로거를 사용해 메트릭 결과를 기록하고, 학습 과정 중 중요 정보를 저장한다. 이는 모델의 성능을 추적하고 문제를 파악하는 데 유용하다.
# 가중치 평균과 매크로 평균의 차이: macro와 weighted 평균을 사용해 다양한 방식으로 모델의 성능을 평가한다. macro는 각 클래스에 동일한 중요도를 부여하고, weighted는 샘플 수가 많은 클래스에 더 높은 중요도를 부여한다.
# 혼동 행렬의 시각화: _show_confusion_matrix 메서드는 모델의 예측 성능을 클래스별로 분석할 수 있는 좋은 방법이다. 특히, 잘못된 분류 패턴을 파악하는 데 유용하다.

# 의문점 및 추가 설명
# brentq와 interp1d의 역할: 이 코드에서는 사용되지 않았지만, 일반적으로 ROC 커브에서 임계값을 조정해 정확한 EER(Equal Error Rate)을 계산하는 데 사용된다.
# eval_results 딕셔너리의 구조: 여러 메트릭 값을 한 곳에 저장해 반환함으로써, 모델의 성능을 비교적 간단하게 분석할 수 있다.
# 스코어별 역할: 다양한 평가 지표를 사용함으로써 단순한 정확도 이상의 정밀한 모델 성능 평가가 가능하다.
# 이 코드의 목적은 모델이 예측한 결과에 대해 다양한 평가 지표를 계산하고, 이를 바탕으로 모델의 성능을 종합적으로 평가하는 것이다. 특히, 다중 클래스 분류 문제에서 각 클래스별로 성능을 분석할 때 유용하게 사용할 수 있다.