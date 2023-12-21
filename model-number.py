import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import warnings
warnings. filterwarnings('ignore')
# تعیین دیتافریم با نام‌های مناسب
labels_df = pd.read_csv('image-data/label-image.csv')
print(type(labels_df))
# تعریف ژنریتور داده با تنظیمات مختلف
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# تولید داده با استفاده از flow_from_dataframe
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    x_col='image',
    y_col='label',
    target_size=(28, 28),  # ابعاد تصویر ورودی
    batch_size=32,
    class_mode='categorical',  # برای دسته‌بندی چند دسته‌ای
    subset='training'  # استفاده از زیرمجموعه آموزشی
)

# ساخت مدل
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# آموزش مدل
model.fit(train_generator, epochs=5)
