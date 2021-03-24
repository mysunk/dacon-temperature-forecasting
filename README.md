공공 데이터 활용 온도 추정 AI 경진대회
=======================================

기상청 공공데이터를 활용하여 온도를 예측하는 대회입니다.   
대회 링크:
https://www.dacon.io/competitions/official/235584/overview/description#desc-info

Dataset
==================
이 저장소에 데이터셋은 제외되어 있습니다.  
데이터셋 출처: 
https://www.dacon.io/competitions/official/235584/data/

Structure
==================
```setup
.
└── main.py
└── pred_residual.py
└── util.py
```
* main.py: feature extraction부터 modeling까지의 main문
* pred_residual.py: 1차 예측 후, 잔차를 예측하는 파일
* util.py: custom 함수가 정의된 파일

Results
==================
* 평가지표: 실제값과 1 미만의 차이에 대해서는 패널티를 주지 않는 MSE
* MSE 결과: 4.6
* private rank: 11/378