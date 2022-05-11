## BERT(**B**idirectional **E**ncoder **R**epresentations from **T**ransformers)
SKT에서 제공하는 Pretrained KoBERT 모델을 활용하여 분류 태스크를 수행합니다.  
BERT의 [CLS] 토큰을 분류 layer에 입력해 문장을 분류합니다.

### How to Run
- 가상환경 세팅
```
conda env create --file environment.yml # 가상환경 생성 
```

- binary-class와 mutl-class 데이터 셋으로 분류기 학습하기
```
python train.py
```

- multi-label 데이터 셋으로 분류기 학습하기
```
python train_multi_label.py
```

- API Server 띄우기
```
python api_server.py
```

### Dataset
- NSMC : 영화리뷰가 긍정인지 부정인지 분류합니다.
- KLUE-tc : KLUE benchmark의 일부로 뉴스 타이틀을 보고 주제를 분류합니다.
- korean_unsmile_dataset : 문장이 어떤 혐오표현인지 분류합니다.

### Benchmark
- NSMC : kobart 공식 github에 나오는 성능은 90.2%입니다. 현재 dev set 성능은 oo.oo%이고 하이퍼파라미터 튜닝 및 몇 가지 tweak을 적용하여 개선의 여지가 있습니다.
- KLUE-tc : klue leaderboard를 보면 test set에서 최고 성능이 Macro F1 기준 oo.oo%입니다. 이 리포의 bart classifier는 dev set 기준으로 oo.oo%가 나오고 있습니다.



