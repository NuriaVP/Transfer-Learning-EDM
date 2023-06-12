# NO FUNCIONA (SIN COMENTAR)


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


modelo = getattr(sys.modules[__name__], sys.argv[1])



def direct(proc):
    if (proc==False):
        ruta_general='Datos holdout EMD'
    else:
        ruta_general='Datos holdout preprocesados EMD'

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

def generador_train(directorio,origen,grado,escala):
    generator = train_datagen.flow_from_directory(
        directory = directorio + '/' + train + '/' + origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,
        seed = 42
    )
    return generator


def train_gen(directorio,escala=224):
    generadores = []
    for origen in ['iPhone','OCT','Samsung']:
        for grado in ['EMD', 'NO EMD']:
            generadores.append(generador_train(directorio,origen,grado,escala))            
    generador_combinado = combine_gen(generadores)
    return generador_combinado


def generador_val(directorio,origen,grado,escala):
    generator = train_datagen.flow_from_directory(
        directory = directorio + '/' + val + '/' + origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,#vamos a equilibrar proporciones, 1/80 ya que tenemos 4 cajas de train * 4 orígenes * 5 grados
        seed = 42
    )
    return generator

def val_gen(directorio,escala=224):
    generadores = []
    for origen in ['iPhone','OCT','Samsung']:
        for grado in ['EMD', 'NO EMD']:
            generadores.append(generador_val(directorio,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado


def transfer_learning(directorio,red):
    
    train_datagen = ImageDataGenerator(
        preprocessing_function = dic_preprocesado[red],
        horizontal_flip=True,
        vertical_flip=True,
        zca_whitening=True,
        zca_epsilon=1e-06
    )
    
    train_generator = train_gen(directorio,escala = dic_escala[red])
    
    train_datagen = ImageDataGenerator(
        preprocessing_function = dic_preprocesado[red],
    )
    
    val_generator = val_gen(directorio,escala = dic_escala[red])
    
    test_datagen = ImageDataGenerator(
        preprocessing_function = dic_preprocesado[red],
    )
    
    test_Samsung = test_datagen.flow_from_directory(
        directory = directorio + '/' + test + '/Samsung/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42
    )
    
    test_iPhone = test_datagen.flow_from_directory(
        directory = directorio + '/' + test + '/iPhone/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42
    )
    
    #Definimos el modelo base de transfer-learning
    base_model = red(weights=None, include_top=False, input_shape=(escala[red],escala[red],3))
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
        loss='cateogrical_crossentropy',
        metrics=['accuracy'],
    )
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20,  restore_best_weights=True)

    #ahora el .fit acepta también generators como x
    history = model.fit(
        x = train_generator,
        batch_size=batch,
        epochs=200,
        callbacks=[es],
        validation_data = val_generator
    )
    
    #Métricas de evaluación
    print('_________________________________________________________________________')
    print(f'MÉTRICAS DE EVALUACIÓN\n *CrossValidation\n *Balanced Generator\n *Red: {red}\n')
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

proc=False
directorio = direct(proc)
transfer_learning(directorio,modelo)

proc=True
directorio = direct(proc)
transfer_learning(directorio,modelo)

