#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('tensorflow_version', '1.x')
import tensorflow as tf
print(tf.__version__)


get_ipython().system('pip install -q keras')
get_ipython().system('pip install scikit-image')


import warnings
warnings.filterwarnings("ignore")

get_ipython().system('pip install -U git+https://github.com/qubvel/efficientnet')



from keras.layers import Input, Lambda, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras.models import Model
import efficientnet.keras as enet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


IMAGE_SIZE = [224, 224]
efficient_net = enet.EfficientNetB0(include_top=False, input_shape=(224,224,3), weights='imagenet')
  
# our layers - you can add more if you want
no_of_classes = #your no_of_classes 
x = efficient_net.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(no_of_classes, activation='softmax')(x)
model = Model(inputs=efficient_net.input, outputs=predictions)

for layer in efficient_net.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

model.summary()



model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set_path',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test_set_path',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')



r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


model.save('EN-B5(vehicles).h5')


model.load_weights('/content/drive/My Drive/Colab Notebooks/EN-B5(vehicles).h5')

testing_folder = "test_set_path"

img_size = 224

batch_size = 1

val_datagen = ImageDataGenerator(
    rescale=1. / 255)
validation_generator = val_datagen.flow_from_directory(
    testing_folder,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')


filenames = validation_generator.filenames
nb_samples = len(filenames)
predictions = model.predict_generator(validation_generator, steps=nb_samples)


from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_fscore_support


val_preds = np.argmax(predictions, axis=-1)
val_trues = validation_generator.classes
cm = metrics.confusion_matrix(val_trues, val_preds)

precisions, recall, f1_score, _ = metrics.precision_recall_fscore_support(val_trues, val_preds)




print(cm)

print(precisions)
print(recall)
print(f1_score)
print(_)






