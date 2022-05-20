## ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)
  
박장원 님께서 공개하신 Pretrained KoELECTRA 모델과 Huggingface 와 Pytorch 코드를 사용하여 분류 태스크를 수행합니다.  
  
[박장원 님의 KOELECTRA Repo](https://github.com/monologg/KoELECTRA)



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

- API Server 띄우기
```
python api_server.py
```

### Dataset
- NSMC (nsmc) : 영화리뷰가 긍정인지 부정인지 분류합니다.
- KLUE-Topic Classification (ynat) : KLUE benchmark의 일부로 뉴스 타이틀을 보고 주제를 분류합니다.
- korean_unsmile_dataset : 문장이 어떤 혐오표현인지 분류합니다.

### Benchmark
- NSMC : KoELECTRA 공식 github에 나오는 dev set 성능은 90.63% 입니다. 현재 레포의 ELECTRA classifier 의 성능은 90.55% 이고 하이퍼파라미터 튜닝 및 몇 가지 tweak을 적용하여 개선의 여지가 있습니다.
- KLUE-Topic Classification : klue leaderboard 를 보면 test set에서 최고 성능이 Macro F1 기준 86.05% 입니다. 현재 레포의 ELECTRA classifier는 dev set 기준으로 85.34% 가 나오고 있습니다.


[TO-DO]
- api.py 서빙코드 준비
- multiclass 코드 준비
- 난이도별 기능 추가
- typing
- 주석

[UPDATE]
- README.md (v)
- environment.yaml (v)
- ckpt 파일명에 시간연월일 추가 - 덮어쓰기 방지 (v)
- 성능 결과 저장 파일명에 시간연월일 추가 - 덮어쓰기 방지 (V)
- config 파일 내 earlystopping argument 추가 - es_metric, es_patience, es_min_delta 조정 가능 (v)

