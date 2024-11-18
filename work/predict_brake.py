import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    return img_input

# 예측 함수
def predict_brake(image_path, model_path="brake_detection_model.h5", threshold=0.5):
    # 모델 로드
    model = load_model(model_path)
    
    # 이미지 전처리
    image = preprocess_image(image_path)
    
    # 예측 수행
    prediction = model.predict(image)
    
    # 예측 값 확인
    predicted_value = prediction[0][0]
    print(f"예측 값: {predicted_value}")
    
    # Threshold 적용
    if predicted_value > threshold:
        print(f"브레이크를 밟고 있습니다. (Threshold: {threshold})")
    else:
        print(f"브레이크를 밟고 있지 않습니다. (Threshold: {threshold})")

# 실행 코드
if __name__ == "__main__":
    image_path = "input_image.jpg"
    threshold = 0.7  # 임계값 설정
    predict_brake(image_path, threshold=threshold)
