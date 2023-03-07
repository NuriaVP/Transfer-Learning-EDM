#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import math
import csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report  
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_VGG16

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_VGG19

from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_Xception

from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_ResNet50V2

from tensorflow.keras.applications import ResNet101
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_ResNet101_152

from tensorflow.keras.applications import ResNet152

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_InceptionV3

from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_InceptionResNetV2

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_MobileNet

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_DenseNet121

from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_DenseNet201

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as preprocess_EfficientNetB0


# In[ ]:


color = 'rgb' 
batch = 16
dic_escala = {VGG16:224,VGG19:224,Xception:299,ResNet50V2:224,ResNet101:224,ResNet152:224,InceptionResNetV2:299,MobileNet:224,DenseNet121:224,DenseNet201:224,EfficientNetB0:224,InceptionV3:299}
dic_preprocesado = {VGG16:preprocess_VGG16,
                VGG19:preprocess_VGG19,
                Xception:preprocess_Xception,
                ResNet50V2:preprocess_ResNet50V2,
                ResNet101:preprocess_ResNet101_152,
                ResNet152:preprocess_ResNet101_152,
                InceptionResNetV2:preprocess_InceptionResNetV2,
                MobileNet:preprocess_MobileNet,
                DenseNet121:preprocess_DenseNet121,
                DenseNet201:preprocess_DenseNet201,
                EfficientNetB0:preprocess_EfficientNetB0}


# In[ ]:


def cargar_datos(path1, path2, escala, color = color):
    
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


def transferLearning_classweight_holdout(inp,proc,red):
    
    #CARGA DE DATOS
    
    escala = dic_escala[red]
    
    if inp==False:
        if proc==False:
            ruta_general='Datos EMD'
        else:
            ruta_general='Datos preprocesados EMD'
    else:
        if proc==False:
            ruta_general='Datos INP EMD'
        else:
            ruta_general='Datos preprocesados INP EMD'
    
    dataset_iphone = cargar_datos(ruta_general+'/iPhone/EMD', ruta_general+'/iPhone/NO EMD', escala)
    dataset_samsung = cargar_datos(ruta_general+'/Samsung/EMD', ruta_general+'/Samsung/NO EMD', escala)
    dataset_oct = cargar_datos(ruta_general+'/OCT/EMD', ruta_general+'/OCT/NO EMD', escala)
    
    print('___________________________________________________________________________________')
    print('TEST: iPHONE')
    print('___________________________________________________________________________________')
    print(f'La red empleada es {red} por lo que las imágenes sean reescalado a {escala}x{escala}')
    print(f'Las imágenes están es {color}, están inpaintadas {inp}, están preprocesadas {inp}')
    print('___________________________________________________________________________________')

    
    train_ds = np.concatenate((dataset_oct[0],dataset_samsung[0]))
    train_labels = np.concatenate((dataset_oct[1],dataset_samsung[1]))
 
    test_ds = dataset_iphone[0]
    test_labels = dataset_iphone[1]
    
    #PREPROCESADO DE LOS DATOS
    
    
    train_ds = dic_preprocesado[red](train_ds) 
    test_ds = dic_preprocesado[red](test_ds)
    
    train_labels_categorical = to_categorical(train_labels, num_classes=2)
    test_labels_categorical = to_categorical(test_labels, num_classes=2)
    
    #IMPORTACIÓN MODELO TRANSFER LEARNING
    
    base_model = red(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    base_model.trainable = False ## Not trainable weights
    
    #FINE TUNNING
    
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(50, activation='relu')
    dense_layer_2 = layers.Dense(20, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax') #1 / sigmoid


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    #COMPENSADOR DE PESOS
    
    classes = np.unique(train_labels)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_labels)
    dic_class_weights = {0:class_weights[0], 1:class_weights[1]}
    
    #ENTRENAMIENTO
    
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', #binary_crossentropy
    metrics=['accuracy'],)

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20,  restore_best_weights=True)

    history = model.fit(train_ds, train_labels_categorical, epochs=200, validation_split=0.2, batch_size=32, callbacks=[es], class_weight=dic_class_weights)
    
    #MÉTRICAS DE EVALUACIÓN
    
    print('Las métricas de evaluación obtenidas para iPhone son:')
    
    score_test = model.evaluate(x=test_ds, y=test_labels_categorical, verbose = 0)
    print("Test loss:", score_test[0])
    print("Test accuracy:", score_test[1])
    
    predictions = model.predict(test_ds)
    pred = list(map(lambda x: list(x).index(max(x)),predictions))
    
    matrix = confusion_matrix(test_labels, pred)
    print(f"Matriz de confusión en test con iPhone:\n\n{matrix}\n")
    
    f_score = f1_score(test_labels, pred, average = 'weighted')
    print(f"Valor de 'F1 score' en test con iPhone: {f_score}\n")
    
    auc_roc = roc_auc_score(test_labels, pred, multi_class = 'ovo')
    print(f"Valor de 'AUC' en test con iPhone: {auc_roc}\n")
    
    print('___________________________________________________________________________________')
    print('TEST: Samsung')
    print('___________________________________________________________________________________')
    print(f'La red empleada es {red} por lo que las imágenes sean reescalado a {escala}x{escala}')
    print(f'Las imágenes están es {color}, están inpaintadas {inp}, están preprocesadas {inp}')
    print('___________________________________________________________________________________')
    
    train_ds = np.concatenate((dataset_oct[0],dataset_iphone[0]))
    train_labels = np.concatenate((dataset_oct[1],dataset_iphone[1]))
 
    test_ds = dataset_samsung[0]
    test_labels = dataset_samsung[1]
    
    #PREPROCESADO DE LOS DATOS
    
    train_ds = dic_preprocesado[red](train_ds) 
    test_ds = dic_preprocesado[red](test_ds)
    
    train_labels_categorical = to_categorical(train_labels, num_classes=2)
    test_labels_categorical = to_categorical(test_labels, num_classes=2)
    
    #IMPORTACIÓN MODELO TRANSFER LEARNING
    
    base_model = red(weights="imagenet", include_top=False, input_shape=train_ds[0].shape)
    base_model.trainable = False ## Not trainable weights
    
    #FINE TUNNING
    
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(50, activation='relu')
    dense_layer_2 = layers.Dense(20, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax') #1 / sigmoid


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    #COMPENSADOR DE PESOS
    
    classes = np.unique(train_labels)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_labels)
    dic_class_weights = {0:class_weights[0], 1:class_weights[1]}
    
    #ENTRENAMIENTO
    
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', #binary_crossentropy
    metrics=['accuracy'],)

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=20,  restore_best_weights=True)

    history = model.fit(train_ds, train_labels_categorical, epochs=200, validation_split=0.2, batch_size=32, callbacks=[es], class_weight=dic_class_weights)
    
    #MÉTRICAS DE EVALUACIÓN
    
    print('Las métricas de evaluación obtenidas para Samsung son:')
    
    score_test = model.evaluate(x=test_ds, y=test_labels_categorical, verbose = 0)
    print("Test loss:", score_test[0])
    print("Test accuracy:", score_test[1])
    
    predictions = model.predict(test_ds)
    pred = list(map(lambda x: list(x).index(max(x)),predictions))
    
    matrix = confusion_matrix(test_labels, pred)
    print(f"Matriz de confusión en test con Samsung:\n\n{matrix}\n")
    
    f_score = f1_score(test_labels, pred, average = 'weighted')
    print(f"Valor de 'F1 score' en test con Samsung: {f_score}\n")
    
    auc_roc = roc_auc_score(test_labels, pred, multi_class = 'ovo')
    print(f"Valor de 'AUC' en test con Samsung: {auc_roc}\n")
    
    return history


# In[ ]:


print('___________________________________________________________________________________')
print('SIN INPAINT SIN PREPROCESAMIENTO')
print('___________________________________________________________________________________')


# In[ ]:


for e in [VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0]:
    transferLearning_classweight_holdout(False,False,e)


# In[ ]:


print('___________________________________________________________________________________')
print('SIN INPAINT CON PREPROCESAMIENTO')
print('___________________________________________________________________________________')


# In[ ]:


for e in [VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0]:
    transferLearning_classweight_holdout(False,True,e)


# In[ ]:


print('___________________________________________________________________________________')
print('CON INPAINT CON PREPROCESAMIENTO')
print('___________________________________________________________________________________')


# In[ ]:


for e in [VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0]:
    transferLearning_classweight_holdout(True,False,e)


# In[ ]:


print('___________________________________________________________________________________')
print('CON INPAINT CON PREPROCESAMIENTO')
print('___________________________________________________________________________________')


# In[ ]:


for e in [VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0]:
    transferLearning_classweight_holdout(True,True,e)

