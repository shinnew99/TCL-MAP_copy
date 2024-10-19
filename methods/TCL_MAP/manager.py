import torch
import torch.nn.functional as F
import logging  # 로그 관리
from torch import nn
from utils.functions import restore_model, save_model, EarlyStopping
from tqdm import trange,tqdm
from data.utils import get_dataloader
from utils.metrics import AverageMeter, Metrics
from transformers import AdamW, get_linear_schedule_with_warmup
from .model import TCL_MAP
from .loss import SupConLoss  # 대조 손실 계산을 위한 함수
import numpy as np

__all__ = ['TCL_MAP_manager']

class TCL_MAP_manager:  # TCL_MAP 모델의 학습 및 평가 과정 전체를 관리한다.
    def __init__(self, args, data):
        self.logger = logging.getLogger(args.logger_name)  # 로깅을 설정하고 GPU 또는 CPU 사용 여부를 결정한다. 
        self.device = torch.device('cuda:0' if torch.cuada.is_available() else 'cpu')
        self.device = self.device
        self.model = TCL_MAP(args)  # 모델 로드
        self.model.to(self.device)
        self.optimizer, self.scheduler = self._set_optimizer(args, self.model)  # 옵티마이저와 학습률 스케쥴러를 결정한다.

        mm_dataloader = get_dataloader(args, data.mm_data)  # 학습, 검증, 테스트 데이터를 위한 데이터 로더를 준비한다. 
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = mm_dataloader['train'], mm_dataloader['dev'], mm_dataloader['test']
        self.cons_criterion = SupConLoss(temperature = args.temperature)  # 객체 초기화
        self.metrics = Metrics(args)  # 여기도 객체 초기화

        if args.train:
            self.best_eval_score = 0
        else:
            self.model = restore_model(self.model, args.model_output_path, self.device)


    def _set_optimizer(self, args, model): 
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_paramters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
        ]

        optimizer = AdamW(optimizer_grouped_paramters, lr = args.lr, correct_bias=False)   # 모델 매개변수를 업데이트할 AdamW 옵티마이저를 설명하며, bias 및 LayerNorm 같은 일부 파라미터에는 가중치 감소를 적용하지 않는다.

        num_train_optimization_steps = int(args.num_train_examples/args.train_batch_size) * args.num_train_epochs
        num_warmup_steps = int(args.num_train_examples * args.num_train_epochs * args.warmup_proportion.args.train_batch_size)

        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps =num_warmup_steps,
                                                    num_training_steps = num_train_optimization_steps)
        # 학습률을 일정한 비율로 조절하며, 초기에는 warmup을 적용한다.
        return optimizer, scheduler
    
    def _train(self, args):  
        early_stopping = EarlyStopping(args)

        for epoch in trange(int(args.num_train_epochs), desc = "Epoch"):
            self.model.train()
            loss_record = AverageMeter()  # AberageMeter이 뭐지?
            cons_loss_record = AverageMeter()  # cons_loss는 대조학습에서 계산된 손실
            cls_loss_record = AverageMeter()  # cls_loss는 분류 손실

            

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
                text_feats = batch['text_feats'].to(self.device)
                cons_text_feats = batch['cons_text_feats'].to(self.device)
                condition_idx = batch['condition_idx'].to(self.device)
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(True):
                    logits, _, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)

                    cons_feature = torch.cat((condition.unsqueeze(1), cons_condition.unsqueeze(1)), dim=1)
                    cons_loss = self.cons_criterion(cons_feature)
                    cls_loss = self.criterion(logits. label_ids)
                    loss = cls_loss + cons_loss
                    self.optimizer.zero_grad()

                    loss.backward()
                    loss_record.update(loss.item(), label_ids.size(0))
                    cons_loss_record.update(cons_loss.item(), label_ids.size(0))
                    cls_loss_record.update(cls_loss.item(), label_ids.size(0))

                    if args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires.grad], args.grad_clip)
                        # 필요시 큰 기울기를 방지하기 위해 기울기 클리핑을 수행한다.      

                    self.optimizer.step()
                    self.scheduler.step()

            outputs = self._get_outputs(args, self.eval_dataloader)
            eval_score = outputs[args.eval_monitor]

            eval_results = {
                'train_loss': round(loss_record.avg, 4),
                'cons_loss': round(cons_loss_record.avg, 4),
                'cls_loss': round(cls_loss_record.avg, 4),
                'eval_score': round(eval_score, 4),
                'best_eval_score': round(early_stopping.best_score, 4),
            }

            self.logger.info("**** Epoch: %s: Eval results****", str(epoch+1))
            for key in eval_results.keys():
                self.logger.info(" %s = %s", key, str(eval_results[key]))

            early_stopping(eval_score, self.model)

            if args.save_model:
                self.logger.info("Trained models are saved in %s", args.model_output_path)
                save_model(self.model, args.model_output_path)
                # 각 에포크 이후 검증 데이터를사용해 모델을 평가하고, 평가 결과를 로깅한다.
                # 모델 성능이 개선될 경우, 해당 모델을 저장.


        def _get_outputs(self, args, dataloader, show_results=False):
            # 검증 또는 테스트 데이터셋에서 모델을 평가하며, 이 과저에서 기울기 업데이트를 수행하지 않는다.
            # 모델의 예측값, 로짓, 특징등을 저장한다.

            
            self.model.eval()

            total_labels = torch.empty(0, dtype=torch.long).to(self.device)
            total_preds = torch.empty(0, dtype=torch.long).to(self.device)
            total_logits = torch.empty((0, args.num_labels)).to(self.device)
            total_features = torch.empty((0, args.feat_size)).to(self.device)

            for batch in tqdm(dataloader, desc="Iteration"):
                text_feats = batch['text_feats'].to(self.device)
                cons_text_feats = batch['cons_text_feats'].to(self.device)
                condition_idx = batch['condition_idx'].to(self.device)  # condition은 뭘 뜻하는 걸까..?
                video_feats = batch['video_feats'].to(self.device)
                audio_feats = batch['audio_feats'].to(self.device)
                label_ids = batch['label_ids'].to(self.device)

                with torch.set_grad_enabled(False):
                    logits, features, condition, cons_condition = self.model(text_feats, video_feats, audio_feats, cons_text_feats, condition_idx)
                    total_logits = torch.cat((total_logits, logits))
                    total_labels = torch.cat((total_labels, label_ids))
                    total_features = torch.cat((total_features, features))

            total_probs = F.softmax(total_logits.detach(), dim=1)  # softmax로 확률을 계산하고 예측 클래스를 결정
            total_maxprobs, total_preds = total_probs.max(dim=1)

            y_logit = total_logits.cpu().numpy()
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            y_prob = total_maxprobs.cpu().numpy()
            y_feat = total_features.cpu().numpy()

            outputs = self.metrics(y_true, y_pred, show_results=show_results)  # 객체를 사용해 평가 지표를 계산하며, 필요시 실제값과 예측값을 저장한다.

            if args.save_pred and show_results:
                np.save('y_true_' + str(args.seed) + '.npy', y_true)
                np.save('y_pred' + str(args.seed)+ '.npy', y_pred)
            
            outputs.update(
                {
                    'y_prob': y_prob,
                    'y_logit': y_logit,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_feat': y_feat,
                }
            )

            return outputs
        

        def _test(self, args):  
            test_results = {}

            ind_outputs = self._get_outputs(args, self.test_dataloader, show_results=True)  # _get_outputs를 호출하여 모델을 평가하고 결과를 반환한다. 
            if args.train:   
                ind_outputs['best_eval_score'] = round(self.best_eval_score, 4)
            
            test_results.updater(ind_outputs)
            # 학습을 진행한 경우, 최고의평가 점수를 포함하여 결과를 반환함.
            return test_results 
        


# 코드의 주목할 점
# 대조 학습: SupConLoss를 사용한 대조 학습은 모델이 특징 공간에서 유사한 샘플은 가깝게, 다른 샘플은 멀게 학습하도록 도와줍니다.
# 조기 종료: 검증 성능을 모니터링하여 성능 향상이 없을 때 학습을 조기에 종료해 과적합을 방지합니다.
# 장치 관리: GPU 사용 가능 여부에 따라 GPU를 사용하여 대규모 모델 학습을 효율적으로 수행합니다.
# 맞춤형 학습률 스케줄: warmup과 선형 감소를 통해 학습률을 조절해 더 나은 수렴을 돕습니다.


# TCL_MAP_manager 클래스는 텍스트, 비디오, 오디오 특징을 통합한 멀티모달 AI 모델을 학습하고 평가하는 데 필요한 모든 기능을 갖춘 파이프라인. 
# 대조 학습, 조기 종료, 그리고 성능 모니터링을 통해 학습을 효율적이고 효과적으로 관리한다.