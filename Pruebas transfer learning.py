#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import seaborn as sn 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random
import numpy as np
import glob as glob
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.utils import class_weight
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from keras.applications.vgg19 import VGG19
from keras.applications import Xception
from keras.applications import ResNet50V2
from keras.applications import ResNet101
from keras.applications import ResNet152
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import MobileNet
from keras.applications import DenseNet121
from keras.applications import DenseNet201
from keras.applications import EfficientNetB0


# In[ ]:


color = 'rgb' #'grayscale' o 'rgb'

escala = 752

preprocesamiento = 'si'


# In[ ]:


def cargar_datos(path1, path2, escala = escala, color = color):
    
    EMD = os.listdir(path1)
    NO_EMD = os.listdir(path2)
    
    data = []
    labels = []

    for i in EMD:   
        image=tf.keras.preprocessing.image.load_img(path1+'/'+i, color_mode= color, 
        target_size= (escala, escala))
        image=np.array(image)
        data.append(image)
        labels.append(1)
    for i in NO_EMD:   
        image=tf.keras.preprocessing.image.load_img(path2+'/'+i, color_mode= color, 
        target_size= (escala, escala))
        image=np.array(image)
        data.append(image)
        labels.append(0)
        
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels


# In[ ]:


dataset_iphone_no = cargar_datos('Datos INP EMD/iPhone/EMD', 'Datos INP EMD/iPhone/NO EMD')

dataset_samsung_no = cargar_datos('Datos INP EMD/Samsung/EMD', 'Datos INP EMD/Samsung/NO EMD')

dataset_no = cargar_datos('Datos INP EMD/OCT/EMD', 'Datos INP EMD/OCT/NO EMD')

dataset_messidor_no = cargar_datos('Datos INP EMD/MESSIDOR/EMD', 'Datos INP EMD/MESSIDOR/NO EMD')


# In[ ]:


dataset_iphone_si = cargar_datos('Datos preprocesados INP EMD/iPhone/EMD', 'Datos preprocesados INP EMD/iPhone/NO EMD')

dataset_samsung_si = cargar_datos('Datos preprocesados INP EMD/Samsung/EMD', 'Datos preprocesados INP EMD/Samsung/NO EMD')

dataset_si = cargar_datos('Datos preprocesados INP EMD/OCT/EMD', 'Datos preprocesados INP EMD/OCT/NO EMD')

dataset_messidor_si = cargar_datos('Datos preprocesados INP EMD/MESSIDOR/EMD', 'Datos preprocesados INP EMD/MESSIDOR/NO EMD')


# In[ ]:


def transferLearning(test, red, preprocesamiento):
    
    #Definir conjuntos de datos train y test
    
    if test=='iphone' and preprocesamiento='si':
        
        train_ds = np.concatenate((dataset_si[0],dataset_messidor_si[0]))
        train_labels = np.concatenate((dataset_si[1],dataset_messidor_si[1]))
        
        test_ds = dataset_iphone_si[0]
        test_labels = dataset_iphone_si[1]
        
    elif test=='samsung' and preprocesamiento='si':
        
        train_ds = np.concatenate((dataset_si[0],dataset_messidor_si[0]))
        train_labels = np.concatenate((dataset_si[1],dataset_messidor_si[1]))
        
        test_ds = dataset_samsung_si[0]
        test_labels = dataset_samsung_si[1]
        
    elif test=='iphone' and preprocesamiento='no':
        
        train_ds = np.concatenate((dataset_no[0],dataset_messidor_no[0]))
        train_labels = np.concatenate((dataset_no[1],dataset_messidor_no[1]))
        
        test_ds = dataset_iphone_no[0]
        test_labels = dataset_iphone_no[1]

    elif test=='samsung' and preprocesamiento='no':
        
        train_ds = np.concatenate((dataset_no[0],dataset_messidor_no[0]))
        train_labels = np.concatenate((dataset_no[1],dataset_messidor_no[1]))
        
        test_ds = dataset_samsung_no[0]
        test_labels = dataset_samsung_no[1]
    
    train_labels_categorical = to_categorical(train_labels, num_classes=2)
    test_labels_categorical = to_categorical(test_labels, num_classes=2)
    
    #Definir modelo de transfer learning
    base_model = red(weights=None, include_top=False, input_shape=train_ds[0].shape)
    base_model.load_weights('Pesos/' + str(red).split(' ')[1] + '.h5')
    base_model.trainable = False ## Not trainable weights

    #Preprocessing input
    train_ds = preprocess_input(train_ds) 
    test_ds = preprocess_input(test_ds)
    
    #Definir fine tunning
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(50, activation='relu')
    dense_layer_2 = layers.Dense(20, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    #Definir compensador de pesos
    classes = np.unique(train_labels)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_labels)
    dic_class_weights = {0:class_weights[0], 1:class_weights[1]}
    
    #Entrenar el modelo
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20,  restore_best_weights=True)

    history = model.fit(train_ds, train_labels_categorical, epochs=200, validation_split=0.2, batch_size=64, callbacks=[es], class_weight=dic_class_weights)
    
    #Métricas de evaluación
    print("Estoy evaluando el modelo"+red+"utilizando datos de test de"+test)
    
    score_test = model.evaluate(x=test_ds, y=test_labels_categorical, verbose = 0)
    print("Test loss:", score_test[0])
    print("Test accuracy:", score_test[1])
    
    predictions = model.predict(test_ds)
    roc_score = roc_auc_score(test_labels_categorical, predictions, multi_class='ovr')
    print("AUC score:", roc_score)
    
    labels_predict = []
    for e in predictions:
        if e[0]>e[1]:
            labels_predict.append(0)
        else:
            labels_predict.append(1)
        
    f1 = f1_score(test_labels, labels_predict, average='weighted')
    print("f1-score", f1)
    
    matriz = confusion_matrix(test_labels, labels_predict)
    print("Matriz de confusión", matriz)
    
    return history


# In[ ]:


history_vgg16 = transferLearning('iphone', VGG16, preprocesamiento)


# In[ ]:


history_vgg16_bis = transferLearning('samsung', VGG16, preprocesamiento)


# In[ ]:


history_vgg19 = transferLearning('iphone', VGG19, preprocesamiento)


# In[ ]:


history_vgg19_bis = transferLearning('samsung', VGG19, preprocesamiento)


# In[ ]:


history_xception = transferLearning('iphone', Xception, preprocesamiento)


# In[ ]:


history_xception_bis = transferLearning('samsung', Xception, preprocesamiento)


# In[ ]:


history_resnet50v2 = transferLearning('iphone', ResNet50V2, preprocesamiento)


# In[ ]:


history_resnet50v2_bis = transferLearning('samsung', ResNet50V2, preprocesamiento)


# In[ ]:


history_resnet101 = transferLearning('iphone', ResNet101, preprocesamiento)


# In[ ]:


history_resnet101_bis = transferLearning('samsung', ResNet101, preprocesamiento)


# In[ ]:


history_resnet152 = transferLearning('iphone', ResNet152, preprocesamiento)


# In[ ]:


history_resnet152_bis = transferLearning('samsung', ResNet152, preprocesamiento)


# In[ ]:


history_inceptionv3 = transferLearning('iphone', InceptionV3, preprocesamiento)


# In[ ]:


history_inceptionv3_bis = transferLearning('samsung', InceptionV3, preprocesamiento)


# In[ ]:


history_InceptionResNetV2 = transferLearning('iphone', InceptionResNetV2, preprocesamiento)


# In[ ]:


history_InceptionResNetV2_bis = transferLearning('samsung', InceptionResNetV2, preprocesamiento)


# In[ ]:


history_MobileNet = transferLearning('iphone', MobileNet, preprocesamiento)


# In[ ]:


history_MobileNet_bis = transferLearning('samsung', MobileNet, preprocesamiento)


# In[ ]:


history_DenseNet121 = transferLearning('iphone', DenseNet121, preprocesamiento)


# In[ ]:


history_DenseNet121_bis = transferLearning('samsung', DenseNet121, preprocesamiento)


# In[ ]:


history_DenseNet201 = transferLearning('iphone', DenseNet201, preprocesamiento)


# In[ ]:


history_DenseNet201_bis = transferLearning('samsung', DenseNet201, preprocesamiento)


# In[ ]:


history_EfficientNetB0 = transferLearning('iphone', EfficientNetB0, preprocesamiento)


# In[ ]:


history_EfficientNetB0_bis = transferLearning('samsung', EfficientNetB0, preprocesamiento)

