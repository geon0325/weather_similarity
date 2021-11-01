# 영상 기반 유사사례 검색: 샘플 코드 및 데이터

## 1. 개요
### 목표
- 심층학습 기반 영상 패턴 추출 기술 개발
- K-최근접 이웃그래프 기반 검색 기술 개발
- 심층학습 기반 초단기 영상 예측 기술 개발
- 비지도 학습 기반 이상 탐지 기술 개발

### 필요성
- 유사 벡터 탐색의 높은 정확도 (유사도 지표 기준 0.80 이상)
- 실시간 검색 기능

### 의의
- 방대한 양의 누적된 데이터에서 실시간으로 유사사례 검색 가능
- 유사 사례 참고를 통한 예보 품질 향상
- 실시간 영상 패턴 인식 및 위험 기상 사전 탐지 가능

## 2. 데이터셋 (TODO)

- COMS 위성 이미지 예시:
<p align="center">
  <img src="raw_images/201103030645_image/coms_mi_le1b_ir01_ea040ps_201103030645.png" width=15% height=15%>
  <img src="raw_images/201103030645_image/coms_mi_le1b_ir02_ea040ps_201103030645.png" width=15% height=15%>
  <img src="raw_images/201103030645_image/coms_mi_le1b_swir_ea040ps_201103030645.png" width=15% height=15%>
  <img src="raw_images/201103030645_image/coms_mi_le1b_wv_ea040ps_201103030645.png" width=15% height=15%>
</p>

- 데이터셋 설명 표:

- 데이터셋 링크: NAS 링크

- [데이터셋 분석](Analyze_COMS.ipynb)
    1. 유효한 이미지 분포 파악
    2. 관측 주기에 관한 분포 분석
    3. 6시간 길이의 비디오를 구성하기에 적절한 이미지 개수 분석
    
<p align="center">
  <img src="images/관측주기_분포.PNG" width=50%>
</p>

## 3. 유사도 지표
### 영상 유사도 지표: 격자 기반 정렬
1. 이미지를 N x N 격자로 분할.
2. 각 격자 내의 픽셀 값을 B개의 구간으로 나누어진 분포료 표현.
3. 격자 간의 (1 - D-statistic) 계산하여 유사도 측정.
4. 격자 간의 유사도 평균으로 이미지 간의 유사도 측정.
5. 이미지 간의 유사도 평균으로 영상 간의 유사도 측정.

<p align="center">
  <img src="images/ground_truth.PNG" width=50% height=50%>
</p>

### 예시
- (**왼쪽**) 2012년 10월 27일 14시 15분 / (**가운데**) 2012년 10월 27일 16시 45분 / (**오른쪽**) 2012년 12월 22일 01시 00분
- **왼쪽**과 **가운데**의 유사도 = **0.807**
- **왼쪽**과 **오른쪽**의 유사도 = **0.354**

<p align="center">
  <img src="images/gt_sample_2.png" width=32% height=32%>
  <img src="images/gt_sample_1.png" width=32% height=32%>
  <img src="images/gt_sample_3.png" width=32% height=32%>
</p>

### 실행 방법
- [ground_truth](https://github.com/geonlee0325/weather_similarity/tree/main/ground_truth) 폴더에 있는 3개의 예시 영상 [video_1](https://github.com/geonlee0325/weather_similarity/tree/main/ground_truth/video_1), [video_2](https://github.com/geonlee0325/weather_similarity/tree/main/ground_truth/video_2), [video_3](https://github.com/geonlee0325/weather_similarity/tree/main/ground_truth/video_3) 참고
- 격자 개수(**N**)과 픽셀 분포의 구간 개수(**B**)를 다음과 같이 설정하여, 다음과 같이 [main.cpp](https://github.com/geonlee0325/weather_similarity/tree/main/ground_truth/main.cpp) C++ 코드를 컴파일 및 실행:
```
컴파일: g++ -o run main.cpp 
실행  : ./run [Video1의 경로] [Video2의 경로] [격자개수 N] [분포구간개수 B]
```
- 실행 예시:
```
입력: ./run video_1 video_2 24 20
출력: 0.807409

입력: ./run video_1 video_3 24 20
출력: 0.354187
```

## 4. 전처리 (TODO)
- [이미지 전처리](preprocess_image.ipynb)
- [비디오 전처리](preprocess_video.ipynb)
- 실행 방법 설명

## 5. 모델 학습 (TODO)
### 심층 메트릭 학습을 통한 이미지 유사도 학습
- 이미지 간의 유사도가 다차원 영상 벡터 간의 유사도에서 유지되도록 학습

<p align="center">
  <img src="images/model_1.PNG" width=50% height=50%>
</p>

- 손실함수로는 [Log-Ratio Loss](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Metric_Learning_Beyond_Binary_Supervision_CVPR_2019_paper.pdf) 활용

<p align="center">
  <img src="images/model_2.PNG" width=40% height=40%>
</p>

### 학습 / 검증 / 평가 데이터
- **2013년 1월 1일 00시 00분**부터 **2013년 1월 7일 23시 45분**까지의 567개의 영상 
- 이미지를 [구글 드라이브](https://drive.google.com/file/d/1ct4O3C5-gJUBp7j2AxSdM3-2KQIuUXNy/view?usp=sharing)에서 다운로드하여 [sample_code](https://github.com/geonlee0325/weather_similarity/blob/main/sample_code/)에 옮긴 후 압축해제:
```
unzip images.zip
```
- i-번째 이미지는 ```images/image_[i]```에 폴더에 저장됨.
- 각 영상을 구성하는 20장의 이미지 배열을 [video2image.txt](https://github.com/geonlee0325/weather_similarity/blob/main/sample_code/video2image.txt)에서 확인:
```
1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
...
566,567,568,569,570,571,573,574,575,576,577,578,579,580,581,582,583,584,585,586
567,568,569,570,571,572,573,574,575,576,577,578,579,580,582,583,584,585,586,587
```
- 567개의 영상을 **20:20:60**의 비율로 학습 데이터 ([train.txt](https://github.com/geonlee0325/weather_similarity/blob/main/sample_code/train.txt)), 검증 데이터 ([valid.txt](https://github.com/geonlee0325/weather_similarity/blob/main/sample_code/valid.txt)), 평가 데이터 ([test.txt](https://github.com/geonlee0325/weather_similarity/blob/main/sample_code/test.txt)) 
- 각 데이터는 ```[영상 1]<tab>[영상 2]<tab>[영상 3]<tab>[영상 1과 영상 2의 유사도]<tab>[영상 1과 영상 3의 유사도]``` 형태로 표현됨:
```
384	380	60	0.89016	0.481642
63	59	443	0.864789	0.437944
39	47	302	0.81809	0.485667
...
```

### 모델 실행 방법 설명
- 여기 설명

### 실행 결과
- 실행 결과 (correlation)
