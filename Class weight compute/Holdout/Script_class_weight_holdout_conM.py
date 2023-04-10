#!/usr/bin/env python
# coding: utf-8
'''
El siguiente script está diseñado para ejecutar redes de transfer learning empleando class_weight_compute() y técnicas de holdout sobre
imágenes de OCT, iPhone, Samsung y MESSIDOR (dataset público) tanto preprocesadas como no preprocesadas.

AUTOR: Nuria Velasco Pérez
FECHA: Abril 2023
TRABAJO FIN DE GRADO
'''
'''
Importaciones necesarias para el correcto funcionamiento del código
'''
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

'''
Definición de parámetros necesarios y constantes para ejecutar las redes convolucionales    

color = 'rbg': Las imágenes originales estaban en color (tres canales) y las dejamos así.
batch = 16: El batch lo establecemos a 16 porque es el más adecuado teniendo en cuenta el tamaño del dataset empleado.
dic_escala: Las dimensiones de las imágenes para entrar en las redes convolucionales dependen de la arquitectura de transfer learning
            empleada, así que creamos un diccionario con la escala correspondiente a cada red.
dic_preprocesado: Para que las imágenes tengan el mismo aspecto que las que fueron entrenadas inicialmente en cada red de transfer
                  learning, existen unas funciones en el paquete 'applications' de keras que las preprocesan adecuadamente para cada
                  arquitectura. Así que las hemos importado todas y ahora creamos un diccionario para acceder a la qe corresponda en cada                     ejecución.
'''
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

'''
cargar_datos() permite cargar en memoria un dataset y las etiquetas de cada uno de sus elementos. Esta función la empleará transferLearning_classweight_holdout() para obtener los datos con los que luego entrenar y validar el modelo.

Atributos: tenemos en cuenta que al ser un problema binario necesitamos indicar la ruta correspondiente al directorio de cada una de las
dos clases. O sea, esta función presupone que tenemos las imágenes almacenadas de la siguiente manera.
    Dataset > Tipo1
            > Tipo2
            
    - path1: ruta de la carpeta una de las clases a estudiar. (Dataset//Tipo1)
    - path2: ruta de la carpeta de la otra a estudiar. (Dataset//Tipo2)
    - escala: dimensiones que tendrán las imágenes de salida. Este valor lo obtendremos a partir de 'dic_escala'.
    - color: color de las imágenes. En principio siempre lo dejamos como 'rgb'.

Algoritmo: 

Vamos a cargar en memoria las imágenes de ambos directorios a partir de la función load_img() del paquete 'image' de keras. Esta función 
se aplica imagen por imagen y diferenciamos dos bucles for diferentes porque la ruta de cada directorio es distinta. Además tiene como parámetros de entrada la propia imagen, el 'color_mode' que identificaremos como nuestro color, y 'target_size' donde indicaremos
la escala de las imágenes.

Tras haber cargado cada imagen la convertimos en un array tridimensional de numpy, y la almacenamos en la lista 'data', que finalmente
contendrá todos los arrays de las imágenes sean de la clase que sean. 

De forma análoga, vamos a guardar en una única lista todas las etiquetas de las imágenes, esta se denominará 'labels'.

Producto:

    - data: array que contiene los arrays de todas las imágenes del dataset.
    - labels: array que contiene las clases de cada una de las imágenes en la misma posición en la que se encuentran en 'data'. En 'labels'
              almacenaremos '0' si la imagen pertence a la primera clase y '1' si pertence a la segunda.
'''
def cargar_datos(path1, path2, escala, color = color):
    
    EMD = os.listdir(path1) #Obtener la lista de las imágenes de la primera clase.
    NO_EMD = os.listdir(path2) #Obtener la lista de las imágenes de la segunda clase.
    
    data = [] #Array que contiene los arrays de todas las imágenes del dataset.
    labels = [] #Array que contiene las clases de cada una de las imágenes en la misma posición en la que se encuentran en 'data'.

    for i in EMD: #Cargar en memoria imágenes de la primera clase.  
        image=tf.keras.preprocessing.image.load_img(path1+'/'+i, color_mode= color, 
        target_size= (escala, escala))
        image=np.array(image)
        data.append(image)
        labels.append(1)
    for i in NO_EMD: #Cargar en memoria imágenes de la segunda clase.
        image=tf.keras.preprocessing.image.load_img(path2+'/'+i, color_mode= color, 
        target_size= (escala, escala))
        image=np.array(image)
        data.append(image)
        labels.append(0)
        
    data = np.array(data) #Todos los arrays de las imágenes
    labels = np.array(labels) #Todas las etiquetas de las imágenes
    
    return data, labels

'''
transferLearning_classweight_holdout(proc,red) entrena una red de transfer learning con los datos train correspondientes y evalúa su
efectividad con otro conjunto de datos test. 

Atributos:

    -proc: Es un boleano que indicará si queremos o no que las imágenes empleadas estén preprocesadas. Tenemos almacenados en memoria dos
           directorios diferentes con la misma estructura, pero en un caso con las imágenes preprocesadas y en otro caso no. 
    -red: Arquitectura de transfer learning. Todas las que vamos a considerar son: VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152,           InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0.

Algoritmo a grandes rasgos:

1) Carga de datos:

    En primer lugar seleccionamos el dataset que nos interesa en función de si trabajamos o no con imágenes preprocesadas (proc). Si las
    queremos preprocesadas accedemos a 'Datos base preprocesados EMD', y si no a 'Datos base EMD'. Ambos directorios están almacenados a la
    misma altura que este script.
    
    Después empleamos la función cargar_datos() con la escala correspondiente a la red que estamos empleando y que hemos buscado empleando
    como clave el nombre de la red en 'dic_escala'. La ejecutamos tres veces, una para dispositivo que estamos considerando: OCT, iPhone y
    Samsung. Y en cada caso, añadimos las carpetas 'EMD' y 'NO EMD' correspondientes a cada dispositivo.
    
    A partir de este punto, vamos a dividir el código en dos partes, una para entrenar la red empleando como test iPhone y otra considerando
    que test es Samsung.

TEST: iPhone

2) División train y test:

    En este caso, si vamos a testear con iPhone, entrenaremos con OCT, Samsung y MESSIDOR. Así que en 'train_ds' concatenamos los arrays de     las imágenes de OCT, Samsung y MESSIDOR. Y, en 'train_labels', concatenamos las listas de etiquetas de OCT, Samsung y MESSIDOR.
    
    Por otro lado, asignamos como 'test_ds' el array con el conjunto de imágenes de iPhone. Y como 'test_labels' la lista de etiquetas de
    las imágenes de iPhone. 

3) Preprocesado de los datos:

    Preprocesamos las imágenes tanto de train como de test con la función de preprocesamiento correspondiente a cada modelo de transfer 
    learning que hemos almacenado en 'dic_preprocesado'. Además las listas de etiquetas necesitamos que sean categóricas para luego
    usar fit() así que aplicamos sobre 'train_labels' y 'test_labels' la función to_categorical().

4) Modelo base de transfer learning:

    Recogemos la arquitectura de capas convolucionales de la red de transfer_learning indicada como parámetro de entrada, y la guardamos
    en 'base_model'. Como hemos importado todas las redes que nos interesa basta con llamar a la función que tenga el nombre indicado en
    red. Esta función tiene los siguientes parámetros de entrada.
    
        -weights: los pesos obtenidos de entrenamientos anteriores, son valores que se conoce que son adecuados para las máscaras de las                       capas convolucionales. Estos los recogemos del set público 'imagenet', pero como las ejecuciones en Scayle no tienen                         acceso a internet, indicamos en este parámetro 'None'. Luego los importaremos con 'load_weights()' de la carpeta 'Pesos'
                  que he creado donde están esos pesos de 'imagenet' que me he desacargado para no tener que acceder a internet. 
        -include_top: boleano que sirve para indicar si queremos dejar las últimas capas de la arquitectura o no. En nuestro caso queremos
                      adaptar el modelo a un nuevo conjunto de datos con dos clases, así que diseñaremos esas capas después e indicamos                           'False'.
        -input_shape: tamaño de cada uno de los tres canales de las imágenes. Los dos primeros valores los asignamos a la escala                                   correspondiente a la red que nos interesa, que la tenemos en 'dic_escala'. Y en el tercero ponemos '3' porque                               trabajamos con imágenes en color.
    
    En principio, no vamos a guardar los pesos de los nuevos entrenamientos, así que también indicamos 'False' en el atributo trainable.

5) Fine tunning:

    Como ya he adelantado, vamos a diseñar las últimas capas de la arquitectura de forma acorde al conjunto de datos. Así que como capas         finales creamos una flatten, dos densas de tipo 'relu' y un última densa de tipo 'softmax' indicando que encuentre la clase de cada         imagen evaluada dentro de las dos etiquetas existentes. 
    
    Para tener el modelo completo unimos base_model con estas últimas cuatro capas con la función Sequential() del paquete models.

6) Compensador de pesos:
    
    Se trata de una técnica de equilibrado de pesos que nos servirá para abordar el problema del desquilibrio de nuestros datos, ya que         tenemos muchas más imágenes sin edema que con enfermedad. En Python podemos realizar la compensación con la función class_weight_compute     que posee el paquete class_weight de sklearn. Esta función devuelve el peso que ha de considerarse en cada una de las dos clases para       lograr el equilibrio esperado. Y posee tres atributos de entrada:
    
        - La opción 'balanced' que se utiliza para calcular los pesos de clase equilibrados automáticamente en función del número de                   observaciones en cada clase.
        - Las clases que estamos considerando. En este caso, 0 y 1, aunque pedimos que las obtenga automáticamente aplicando la función
          unique() de numpy sobre el conjunto de etiquetas. Unique() devuelve los valores únicos de la matriz indicada.
        - Lista de etiquetas de los datos considerados para entrenar.
    
    Nosotros hemos guardado estos pesos como valores de un diccionario con dos datos que tiene como claves el nombre de las etiquetas a las     que se refiere cada peso. Este diccionario es el formato que necesitamos para emplear luego el parámetro "class_weight" en fit().

7) Entrenamiento: 

   En primer lugar, hemos de compilar el modelo neuronal creado para especificar su configuración. Para ello, utilizamos la función            compile() sobre nuestro "model", esta función acepta varios argumentos pero solo nos interesan los siguientes.
   
       - optimizer: algoritmo de optimización que se utiliza para minimizar la función de pérdida durante el entrenamiento. En este proyecto 
         emplearemos siempre "adam" como optimizador porque adapta el tamaño del paso de actualización de los pesos en función de la                  magnitud de los gradientes para cada peso individual, lo que significa que cada peso tiene su propia tasa de aprendizaje                    adaptativa. Adam se trata de un algoritmo de optimización estocástica que combina los conceptos del descenso de gradiente                    estocástico (SGD) y el descenso de gradiente con momento.
       - loss: la función de pérdida que se utiliza para evaluar la precisión del modelo durante el entrenamiento. En nuestro caso                    emplearemos "categorical_crossentropy" al estar trabajando con un problema de dos clases.
       - metrics: lista de métricas que se utilizan para evaluar el rendimiento del modelo durante el entrenamiento. Pediremos que devuelva          la precisión con la que va entrenando, el resto de métricas de interés las calcularemos al realizar predicciones.
   
   Antes de comenzar con el entrenamiento vamos a incluir Early Stopping, una técnica utilizada para detener el entrenamiento de un modelo      antes de que se complete todo el número de épocas (iteraciones sobre el conjunto de datos) programado, con el objetivo de evitar el          sobreajuste y ahorrar tiempo de entrenamiento. Early Stopping detiene el entrenamiento del modelo cuando se alcanza un cierto punto de      saturación en el rendimiento, es decir, cuando el modelo deja de mejorar en el conjunto de validación. Los atributos de esta función son:
   
       - monitor: la métrica a monitorear durante el entrenamiento, en este caso "val_accuracy".
       - mode: si la métrica debe ser maximizada o minimizada, en este caso "max".
       - patience: el número de épocas que se esperará para ver una mejora en la métrica antes de detener el entrenamiento, en este caso 20.
       - restore_best_weights: un indicador booleano que indica si se deben restaurar los pesos del modelo con los mejores valores durante            el entrenamiento.
   
   Y, a continuación, ya realizamos el entrenamiento planteado con la función fit(). Los atributos que nos interesan en este código son:
   
       - X_train = train_ds: datos para el entrenamiento.
       - y_train = train_labels: etiquetas del entrenamiento.
       - batch_size: número de muestras que se utilizarán en cada actualización de gradiente. El conjunto de datos completo se divide en              lotes de este tamaño y se procesa uno por uno.
       - epochs: cantidad de épocas en las que se dividirá el entrenamiento.
       - validation_split: sirve para separar una fracción de los datos de entrenamiento como conjunto de validación. En este caso hemos              reservado el 20%. Podría indicarse otro conjunto de datos diferente al de entrenamiento, pero en este caso no es lo que nos                  conviene.
       - callbaks: son objetos que se utilizan durante el entrenamiento del modelo para realizar acciones específicas en momentos                    determinados, en este caso solo indicamos "es" por el EarlyStopping que hemos definido.
       - class_weight: indicamos el diccionario de pesos personalizados para cada clase que hemos creado con el fin de solventar el                  desequilibrio entre clases.

8) Métricas de evaluación:

   Para obtener métricas que permitan ver el rendimiento de la red neuronal necesitamos hacer predicciones con el conjunto de datos de test.    Por eso utilizamos la función evaluate() para hacer la primera predicción indicando los datos de test definidos y las etiquetas reales      que luego compararemos con las predichas. Del resultado de evaluate obtenemos los datos de loss y accuracy, respectivamente.
   
       - loss: es una función que se utiliza para medir la discrepancia entre las predicciones de un modelo y las etiquetas verdaderas de            los datos de entrenamiento. Podríamos decir que mide cuán malo es el modelo en hacer predicciones en relación con las etiquetas              verdaderas. El objetivo del entrenamiento del modelo es minimizar la función de pérdida para mejorar la capacidad del modelo de              hacer predicciones precisas, así que buscamos valores muy bajos de loss.
       - accuracy: es una de las métricas más relevantes para medir la calidad de un modelo de clasificación. La precisión es la proporción          de ejemplos de prueba para los cuales el modelo predice la etiqueta correcta.
       
   Sin embargo, cuando se hacer predicciones con un modelo ya entrenado, también existe otra función interesante predict(). Esta devuelve      una matriz de predicciones correspondientes a las entradas proporcionadas, lo que puede ser interesante para calcular métricas más          específicas. Nosotros vamos a convertir esta matriz en una lista unidimensional quedándonos con la clase que tiene una probabilidad más      grande para cada dato. A modo ilustrativo veamos a poner un ejemplo, la función predict para un dato cualquiera devolverá la probabilidad    relativa a cada clase [0.1 0.9], y nosotros queremos quedarnos con la clase a la que se refiere la probabilidad más grande, en este caso    1 por el 0.9. Esta última conversión la conseguimos haciendo un map() que se quede con la etiqueta de la predicción más grande.
   
   Con esta nueva lista de predicciones resultantes, vamos a calcular el resto de indicadores de interés:
   
       - Matriz de confusión: con la función confusion_matrix. Es una tabla que muestra la cantidad de verdaderos positivos, falsos                  positivos, verdaderos negativos y falsos negativos que se han producido para cada clase en un conjunto de datos.
         
         - Verdaderos positivos (TP): número de casos en que el modelo predijo correctamente la clase positiva (verdadero) cuando la clase              real era positiva.
         - Falsos positivos (FP): número de casos en que el modelo predijo incorrectamente la clase positiva (falso) cuando la clase real              era negativa.
         - Verdaderos negativos (TN): número de casos en que el modelo predijo correctamente la clase negativa (verdadero) cuando la clase              real era negativa.
         - Falsos negativos (FN): número de casos en que el modelo predijo incorrectamente la clase negativa (falso) cuando la clase real              era positiva.
       
       - Valor F1: con la función f1_score. Es una medida común de la precisión de un modelo de clasificación binaria, que combina tanto la          precisión como la exhaustividad (recall) en una sola métrica.
       - Auc Roc: con la función auc_roc_score. La curva ROC (característica de operación del receptor) es una representación gráfica de la          relación entre la tasa de verdaderos positivos y la tasa de falsos positivos a través de diferentes umbrales de decisión del                modelo. La métrica AUC-ROC es el área bajo esta curva ROC y varía entre 0 y 1. Un modelo con una AUC-ROC de 1 se considera                  perfecto, mientras que un modelo con una AUC-ROC de 0.5 se considera equivalente a una elección aleatoria.
       
TEST: Samsung

2) División train y test:

    En este caso, si vamos a testear con Samsung, entrenaremos con OCT, iPhone y MESSIDOR. Así que en 'train_ds' concatenamos los arrays de     las imágenes de OCT, iPhone y MESSIDOR. Y, en 'train_labels', concatenamos las listas de etiquetas de OCT, iPhone y MESSIDOR.
    
    Por otro lado, asignamos como 'test_ds' el array con el conjunto de imágenes de iPhone. Y como 'test_labels' la lista de etiquetas de
    las imágenes de Samsung. 

El resto de pasos son idénticos al test con iPhone

3) Preprocesado de los datos

4) Modelo base de transfer learning

5) Fine tunning

6) Compensador de pesos

7) Entrenamiento

8) Métricas de evaluación
'''
def transferLearning_classweight_holdout(proc,red):
    
    #CARGA DE DATOS
    escala = dic_escala[red]

    if proc==False:
        ruta_general='Datos EMD' #Directorio de datos sin preprocesamiento
    else:
        ruta_general='Datos preprocesados EMD' #Directorio de datos con preprocesamiento
    
    dataset_iphone = cargar_datos(ruta_general+'/iPhone/EMD', ruta_general+'/iPhone/NO EMD', escala)
    dataset_samsung = cargar_datos(ruta_general+'/Samsung/EMD', ruta_general+'/Samsung/NO EMD', escala)
    dataset_oct = cargar_datos(ruta_general+'/OCT/EMD', ruta_general+'/OCT/NO EMD', escala)
    dataset_messidor = cargar_datos(ruta_general+'/MESSIDOR/EMD', ruta_general+'/MESSIDOR/NO EMD', escala)
    
    print('___________________________________________________________________________________')
    print('TEST: iPHONE')
    print('___________________________________________________________________________________')
    print(f'La red empleada es {red} por lo que las imágenes sean reescalado a {escala}x{escala}')
    print(f'Las imágenes están es {color} y están preprocesadas {proc}')
    print('___________________________________________________________________________________')

    
    train_ds = np.concatenate((dataset_oct[0],dataset_samsung[0]))
    train_ds = np.concatenate((train_ds,dataset_messidor[0]))
    train_labels = np.concatenate((dataset_oct[1],dataset_samsung[1]))
    train_labels = np.concatenate((train_ds,dataset_messidor[1]))
 
    test_ds = dataset_iphone[0]
    test_labels = dataset_iphone[1]
    
    #PREPROCESADO DE LOS DATOS
    train_ds = dic_preprocesado[red](train_ds) 
    test_ds = dic_preprocesado[red](test_ds)
    
    train_labels_categorical = to_categorical(train_labels, num_classes=2)
    test_labels_categorical = to_categorical(test_labels, num_classes=2)
    
    #IMPORTACIÓN MODELO TRANSFER LEARNING
    base_model = red(weights=None, include_top=False, input_shape=(dic_escala[red],dic_escala[red],3))
    base_model.load_weights('Pesos/' + str(red).split(' ')[1] + '.h5')
    base_model.trainable = False ## Not trainable weights
    
    #FINE TUNNING
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
    
    #COMPENSADOR DE PESOS
    classes = np.unique(train_labels)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_labels)
    dic_class_weights = {0:class_weights[0], 1:class_weights[1]}
    
    #ENTRENAMIENTO
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
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
    print(f'Las imágenes están es {color} y están preprocesadas {proc}')
    print('___________________________________________________________________________________')
    
    train_ds = np.concatenate((dataset_oct[0],dataset_iphone[0]))
    train_ds = np.concatenate((train_ds,dataset_messidor[0]))
    train_labels = np.concatenate((dataset_oct[1],dataset_iphone[1]))
    train_labels = np.concatenate((train_ds,dataset_messidor[1]))
 
    test_ds = dataset_samsung[0]
    test_labels = dataset_samsung[1]
    
    #PREPROCESADO DE LOS DATOS
    train_ds = dic_preprocesado[red](train_ds) 
    test_ds = dic_preprocesado[red](test_ds)
    
    train_labels_categorical = to_categorical(train_labels, num_classes=2)
    test_labels_categorical = to_categorical(test_labels, num_classes=2)
    
    #IMPORTACIÓN MODELO TRANSFER LEARNING
    base_model = red(weights=None, include_top=False, input_shape=(dic_escala[red],dic_escala[red],3))
    base_model.load_weights('Pesos/' + str(red).split(' ')[1] + '.h5')
    base_model.trainable = False ## Not trainable weights
    
    #FINE TUNNING
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
    
    #COMPENSADOR DE PESOS
    classes = np.unique(train_labels)
    class_weights = class_weight.compute_class_weight('balanced', classes=classes, y=train_labels)
    dic_class_weights = {0:class_weights[0], 1:class_weights[1]}
    
    #ENTRENAMIENTO
    model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
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

'''
Pruebas con las 12 arquitecturas de transfer learning y las imágenes sin preprocesamiento gaussiano.
'''
print('___________________________________________________________________________________')
print('SIN PREPROCESAMIENTO')
print('___________________________________________________________________________________')



for e in [VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0]:
    transferLearning_classweight_holdout(False,e)

'''
Pruebas con las 12 arquitecturas de transfer learning y las imágenes con preprocesamiento gaussiano.
'''
print('___________________________________________________________________________________')
print('CON PREPROCESAMIENTO')
print('___________________________________________________________________________________')


for e in [VGG16, VGG19, Xception, ResNet50V2, ResNet101, ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0]:
    transferLearning_classweight_holdout(True,e)
