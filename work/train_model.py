import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 사전 학습된 MobileNetV2 로드
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 데이터 로더 준비
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)  # 10%만 검증 데이터로 사용

train_generator = train_datagen.flow_from_directory(
    'brake_data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)
validation_generator = train_datagen.flow_from_directory(
    'brake_data/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)
# 검증 데이터를 제외하고 훈련할 경우
model.fit(train_generator, epochs=5)

# 모델 훈련
#model.fit(train_generator, validation_data=validation_generator, epochs=5)
# 모델 저장
model.save("brake_detection_model.h5")
print("모델이 저장되었습니다: brake_detection_model.h5")
