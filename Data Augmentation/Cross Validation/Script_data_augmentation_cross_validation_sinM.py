#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:

modelo = getattr(sys.modules[__name__], sys.argv[1])

proc = True
if (proc==False):
    directorio = 'Datos cross validation sM base EMD'
else:
    directorio = 'Datos Cross Validation sM EMD'


# In[3]:


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
                EfficientNetB0:preprocess_EfficientNetB0,
		InceptionV3:preprocess_InceptionV3}


# In[4]:


def combine_gen(gens):
    while True:
        tuplas = []
        for i in gens:
            tuplas.append(next(i))
        arrays_images = []
        arrays_labels = []
        for i in tuplas:
            arrays_images.append(i[0])
            arrays_labels.append(i[1])
        images = np.concatenate(arrays_images)
        labels = np.concatenate(arrays_labels)

        yield(tuple((images,labels)))


# In[5]:


def num(directorio,K_test,K_val):
    total = 0
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    lista_K.remove(K_val)
    for K in lista_K:
            for origen in ['iPhone','OCT','Samsung']:
                for grado in ['EMD', 'NO EMD']:
                    total += len(os.listdir(directorio + '/' + K + '/' + origen + '/' + grado + '/' + grado + '/'))

    num_train = int(total*0.8)
    num_val = int(total*0.2)
    return num_train,num_val

def generador_train(modelo,directorio,K,origen,grado,escala):
    train_datagen = ImageDataGenerator(
	preprocessing_function = dic_preprocesado[modelo],
        horizontal_flip=True,
        vertical_flip=True,
        zca_whitening=True,
        zca_epsilon=1e-06
    )

    generator = train_datagen.flow_from_directory(
        directory = directorio + '/' + K + '/' + origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,
        seed = 42
    )  
    return generator


def train_gen(modelo,directorio,K_test,K_val,escala=224):
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    lista_K.remove(K_val)
    generadores = []
    for K in lista_K:
        for origen in ['iPhone','OCT','Samsung']:
            for grado in ['EMD', 'NO EMD']:
                generadores.append(generador_train(modelo,directorio,K,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado



def generador_val(modelo,directorio,K_val,origen,grado,escala=224):
    val_datagen = ImageDataGenerator(
        preprocessing_function = dic_preprocesado[modelo]
    )

    generator = val_datagen.flow_from_directory(
        directory = directorio + '/' + K_val + '/'+ origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,#vamos a equilibrar proporciones, 1/80 ya que tenemos 4 cajas de train * 4 orígenes * 5 grados
        seed = 42
    )
    
    return generator

def val_gen(modelo,directorio,K_val,escala=224):
    generadores = []
    for origen in ['iPhone','OCT','Samsung']:
    	for grado in ['EMD', 'NO EMD']:
        	generadores.append(generador_val(modelo,directorio,K_val,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado



# In[ ]:


def transfer_learning(directorio,K_test,red):
    if (K_test == 'K1'):
        K_val = 'K5'
    elif (K_test == 'K2'):
        K_val = 'K1'
    elif (K_test == 'K3'):
        K_val = 'K2'
    elif (K_test == 'K4'):
        K_val = 'K3'
    elif (K_test == 'K5'):
        K_val = 'K4'
   
    train_generator = train_gen(red,directorio,K_test,K_val,escala = dic_escala[red])
    val_generator = val_gen(red,directorio,K_val,escala = dic_escala[red])
    
    #para el generador de tipo test debemos definir antes un nuevo ImageDataGenerator
    test_datagen = ImageDataGenerator(preprocessing_function=dic_preprocesado[red])
    
    #definimos un generador para Samsung, que coja únicamente las imágenes de Samsung de la caja de test
    test_Samsung = test_datagen.flow_from_directory(
        directory = directorio + '/' + K_test + '/Samsung/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42
    )
    
    #y lo mismo para iPhone, que solo coja las imágenes de iPhone de la caja de test
    test_iPhone = test_datagen.flow_from_directory(
        directory = directorio + '/' + K_test + '/iPhone/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42
    )
    
    #Definimos el modelo base de transfer-learning
    base_model = red(weights=None, include_top=False, input_shape=(dic_escala[red],dic_escala[red],3))
    base_model.load_weights('Pesos/' + str(red).split(' ')[1] + '.h5')
    base_model.trainable = False ## Not trainable weights
    
    #Definir fine tunning
    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(1024, activation='relu')
    dense_layer_2 = layers.Dense(512, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')
    
    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    
    #no usaremos compensador de pesos ya que las proporciones entre grados y entre dispositivos están comepnsadas
    
    #Entrenar el modelo
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20,  restore_best_weights=True)

    #ahora el .fit acepta también generators como x
    history = model.fit(
        x = train_generator,
        batch_size=batch,
        epochs=200,
        steps_per_epoch = math.ceil(num(directorio,K_test,K_val)[0] / batch),
        callbacks=[es],
        validation_data = val_generator,
        validation_steps = math.ceil(num(directorio,K_test,K_val)[1] / batch)
    )
    
    #Métricas de evaluación
    print('_________________________________________________________________________')
    print(f'MÉTRICAS DE EVALUACIÓN\n *CrossValidation\n *Balanced Generator\n *Red: {red}\n *K_test: {K_test}')
    print('_________________________________________________________________________')
    
    print('___________________________________________________________________________________')
    print('TEST: iPHONE')
    print('___________________________________________________________________________________')
    score_test_iphone = model.evaluate(x = test_iPhone, verbose = 0)
    print("Test loss:", score_test_iphone[0])
    print("Test accuracy:", score_test_iphone[1])
    
    y_test_iphone = test_iPhone.classes
    
    predictions_iphone = model.predict(test_iPhone)
    y_pred_iphone = list(map(lambda x: list(x).index(max(x)),predictions_iphone))

    matrix_iphone = confusion_matrix(y_test_iphone, y_pred_iphone)
    print(f"Matriz de confusión en test con iPhone:\n\n{matrix_iphone}\n")
    
    f_score_iphone = f1_score(y_true = y_test_iphone, y_pred = y_pred_iphone, average = 'weighted')
    print(f"Valor de 'F1 score' en test con iPhone: {f_score_iphone}\n")
    
    auc_roc_iphone = roc_auc_score(y_test_iphone, y_pred_iphone, multi_class = 'ovo')
    print(f"Valor de 'AUC' en test con iPhone: {auc_roc_iphone}\n")
    
    print('___________________________________________________________________________________')
    print('TEST: Samsung')
    print('___________________________________________________________________________________')
    
    score_test_samsung = model.evaluate(x = test_Samsung, verbose = 0)
    print("Test loss:", score_test_samsung[0])
    print("Test accuracy:", score_test_samsung[1])
    
    y_test_samsung = test_Samsung.classes
    
    predictions_samsung = model.predict(test_Samsung)
    y_pred_samsung = list(map(lambda x: list(x).index(max(x)),predictions_samsung))

    matrix_samsung = confusion_matrix(y_test_samsung, y_pred_samsung)
    print(f"Matriz de confusión en test con samsung:\n\n{matrix_samsung}\n")
    
    f_score_samsung = f1_score(y_true = y_test_samsung, y_pred = y_pred_samsung, average = 'weighted')
    print(f"Valor de 'F1 score' en test con samsung: {f_score_samsung}\n")
    
    auc_roc_samsung = roc_auc_score(y_test_samsung, y_pred_samsung, multi_class = 'ovo')
    print(f"Valor de 'AUC' en test con Samsung: {auc_roc_samsung}\n")

for K_test in ['K1', 'K2', 'K3', 'K4', 'K5']:
    transfer_learning(directorio,K_test,modelo)


