## ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
  
박장원 님께서 공개하신 Pretrained KoELECTRA 모델의 Fine-tuning 을 통해 Classification 을 수행합니다  
Huggingface 와 Pytorch base 코드를 사용합니다  
[박장원 님의 KOELECTRA Repository](https://github.com/monologg/KoELECTRA)

### Dataset
- NSMC (nsmc) : 영화리뷰가 긍정인지 부정인지 분류합니다
- KLUE-Topic Classification (ynat) : KLUE benchmark의 일부로 뉴스 타이틀을 보고 주제를 분류합니다
- korean_unsmile_dataset : 문장이 어떤 혐오표현인지 분류합니다

### Benchmark
- NSMC : KoELECTRA 공식 github에 나오는 dev set 성능은 90.63% 입니다. 현재 레포의 ELECTRA classifier 의 성능은 90.55% 이고 하이퍼파라미터 튜닝 및 몇 가지 tweak을 적용하여 개선의 여지가 있습니다
- KLUE-Topic Classification : klue leaderboard 를 보면 test set에서 최고 성능이 Macro F1 기준 86.05% 입니다. 현재 레포의 ELECTRA classifier는 dev set 기준으로 85.34% 가 나오고 있습니다

### Cookbook Level별 지원 기능
- Beginner
    - Basic fit, evaluate, predict 기능

- Intermediate
    - Beginner 기능
    - save checkpoint
    - scheduler : earlystopping 추가
    - metrics : 하나 혹은 둘 이상 metric 지원
    - test_during_evaluation : validation 수행 여부 지정

- Expert
    - Beginner + Intermediate 기능
    - scheduler : get_linear_schedule_with_warmup 추가
    - gradient_accumulation
    - clip_grad_norm

### [How to Run]
- 가상환경 세팅
```
conda env create --file environment.yml # 가상환경 생성 
```

- binary-class와 mutl-class 데이터 셋으로 분류기 학습하기
```
python train_beginner.py --task [nsmc | ynat]
```
```
python train_intermediate.py --task [nsmc | ynat]
```
```
python train_expert.py --task [nsmc | ynat]
```
  
- multi-label 데이터 셋으로 분류기 학습하기
```
python train_multi_label.py
```

### [UPDATE]  
(2022.04) ckpt 파일명에 시간연월일 추가 : 덮어쓰기 방지  
(2022.04) 성능 결과 저장 파일명에 시간연월일 추가 : 덮어쓰기 방지  
(2022.04) config 파일 내 earlystopping argument 추가 (es_metric, es_patience, es_min_delta)  
(2022.04) Level별 Cookbook 을 위한 Refactoring 수행 (PR #7)  
(2022.05) type hint, docstring 추가  


### [TO-DO]
- multi label 분류 코드 개발  
- level 별 기능 추가  
    - Intermediate  
        - Tokenizer : \[unused\] token 에 vocab 추가
        - model : freeze layers  
    - Expert
        - wandb integration
        - Loss : Dice Loss or Focal Loss (binary task)
