import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# خواندن فایل CSV
labels_df = pd.read_csv('image-data/label-image.csv')

# تبدیل label ها به one-hot encoded
labels_df['label'] = pd.Categorical(labels_df['label'])
labels_df['label'] = labels_df['label'].cat.codes
labels_onehot = pd.get_dummies(labels_df['label'])

# افزودن ستون‌های one-hot encoded به DataFrame
labels_df = pd.concat([labels_df, labels_onehot], axis=1)

# حذف ستون اصلی label
labels_df = labels_df.drop('label', axis=1)

# تعریف مدل
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

# تنظیمات ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# تولید داده‌ها
train_generator = train_datagen.flow_from_dataframe(
    dataframe=labels_df,
    x_col='image',
    y_col=list(labels_onehot.columns),
    target_size=(28, 28),
    batch_size=32,
    class_mode='raw'
)
print(labels_df.head())
print(labels_onehot.head())

"""
# آموزش مدل
model.fit(train_generator, epochs=5)

# ذخیره مدل
model.save("model/my_model.h5")
"""