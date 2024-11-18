# Brake Detection Project

## 프로젝트 개요
- 차량의 브레이크 여부를 이미지 분석을 통해 예측하는 프로젝트입니다.

## 모델 훈련
- `train_model.py`를 통해 모델을 훈련할 수 있습니다.
- 데이터는 `brake_data/` 폴더에 있어야 하며, `yes/`, `no/` 서브 폴더로 구분되어 있어야 합니다.

## 예측
- `predict_brake.py`를 사용하여 이미지를 입력받아 브레이크 여부를 예측합니다.

## 설치 방법
```bash
pip install -r requirements.txt

## OpenCV 
pip install opencv-python

##