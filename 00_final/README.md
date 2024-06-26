### 위성영상 기반의 식생지수와 기상 정보를 사용한 머신러닝 기반 논 농작물 생산량 예측

#### 1. 프로젝트 개요
 농작물 생산량 예측은 수급 조절, 가격 예측, 관련 정책 수립과 더불어 비료, 관개 등의 
 자원 사용 조절을 통한 지속가능한 농업을 위해 중요한 의사결정 자료 중 하나이다. 
 농작물 생산량 예측을 위해서는 비료 사용, 토양 조성, 관개 방식, 병충해, 기상 등의 정보를 반영할 필요가 있다. 
 개별 농가의 이러한 정보를 얻기는 힘들다는 문제가 있다. 
 하지만 위성영상 또는 무인비행체 영상을 통해 실시간으로 넓은 지역의 식생을 파악할 수 있다. 
 또한 기상청을 통해 기상 정보를 얻을 수 있다. 따라서 농작물 생산을 모니터링할 수 있는 식생 정보와 
 농작물 생산에 영향을 미치는 요소 중 하나인 기상 정보를 바탕으로 농작물 생산량을 예측해보고자 한다. 
 Google Earth Engine 등을 바탕으로 
 ‘위성영상 기반의 식생지수와 기상 정보를 사용한 머신러닝 기반 논 작물 생산량 예측’을 진행하였다.
 

#### 2. 데이터 수집


- 대상 지역: 전라남도 남해군, 전북특별자치도 김제시, 전북특별자치도 김제시
- 대상 기간: 2014년 ~ 2021년
- 대상 작물: 맥류, 두류, 잡곡류, 미곡
- 대상 식생지수: NDVI, GNDVI, NDRE, RVI, CVI

1) 토지피복도 데이터
- 환경공간정보서비스에서 제공하는 2023년 중분류 토지피복도 데이터를 수집하였다.
  - 토지피복도 병합: [01_shp_files.py](./01_shp_files.py)

2) 위성영상 기반 식생지수 데이터
- Google Earth Engine에서 Sentinel-2, Landsat-8 위성영상에서 식생지수 데이터를 수집하였다.
  - 위성영상 기반 식생지수 데이터 수집: [02_save_satellite.py](02_save_satellite.py)
  - 논 영역 필터링 및 통계값 데이터 수집: [03_masking_field.py](03_masking_field.py)

3) 기상 데이터
- 기상청에서 제공하는 기상 데이터를 수집하였다.
- 월별 평균 및 총합 값을 얻었다.

4) 농작물 생산량 데이터
- 통계청에서 제공하는 지자체통계-농작물 생산량 데이터를 수집하였다.

5) EDA
- 위성영상 구름 마스킹 여부에 따른 차이를 확인하였다.
  -  위성 영상과 무인비행체 영상 처리: [EDA/01_preprocess_drone.py](EDA/01_preprocess_drone.py), [EDA/02_preprocess_gee.py](EDA/02_preprocess_gee.py)
- 연도별 월별 기상 정보 변화를 확인하였다.
- 농작물 생산량의 변화를 확인하였다.
- 그래프: [EDA/02_draw_results.py](EDA/02_draw_results.py)


6) 위성영상 데이터와 기상 데이터를 결합하여 머신러닝 모델을 학습하고, 생산량을 예측하였다.
- 전체 데이터 병합: [04_preprocess_datas.py](04_preprocess_datas.py)
- 머신러닝 모델 학습 및 예측: [05_models.py](05_models.py)


