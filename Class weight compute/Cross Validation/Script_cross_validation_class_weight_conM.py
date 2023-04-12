#!/usr/bin/env python
# coding: utf-8
'''
El siguiente script está diseñado para ejecutar redes de transfer learning empleando class_weight_compute() y técnicas de validación cruzada sobre imágenes de OCT, iPhone y Samsung tanto preprocesadas como no preprocesadas.

AUTOR: Nuria Velasco Pérez
FECHA: Abril 2023
TRABAJO FIN DE GRADO
'''

'''
Importaciones necesarias para el correcto funcionamiento del código.
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
Búsqueda del directorio adecuado con las imágenes de OCT, iPhone, MESSIDOR y Samsung distribuidas en las cajas de validación cruzada en función de si las queremos con preprocesamiento gaussiano o no.

He definido un variable boleana proc en la cual asignaremos True si queremos las imágenes sin preprocesamiento, y False si las queremos con preprocesamiento. 

Tanto el directorio que contiene las imágenes preprocesadas como el que no, se estructuran de la siguiente manera:

    K1: OCT:    EMD:    EMD:
                        NO EMD: vacío
                NO EMD: EMD: vacío
                        NO EMD:
        iPhone  EMD:    EMD:
                        NO EMD: vacío
                NO EMD: EMD: vacío
                        NO EMD:
        Samsung EMD:    EMD:
                        NO EMD: vacío
                NO EMD: EMD: vacío
                        NO EMD:
        MESSIDOR EMD:    EMD:
                        NO EMD: vacío
                NO EMD: EMD: vacío
                        NO EMD:
    K2: igual que el anterior
    
    K3: igual que el anterior
    
    K4: igual que el anterior
    
    K5: igual que el anterior
'''

proc=True

if (proc==False):
        directorio = 'Datos cross validation base EMD'
else:
        directorio = 'Datos Cross Validation EMD'

'''
Definición de parámetros necesarios y constantes para ejecutar las redes convolucionales    

modelo: es un argumento que indicaremos en el archivo con extensión .sh a ejecutar en Scayle. Será el primer argumento que aparezca porque 
        indicado "sys.arg[1]" y contendrá una cadena con el nombre de la red de transfer learning a ejecutar en cada prueba. Aunque lo que 
        nosotros indicamos en el archivo .sh será una String lo cambiamos a función con el getattr().
color = 'rbg': Las imágenes originales estaban en color (tres canales) y las dejamos así.
batch = 16: El batch lo establecemos a 16 porque es el más adecuado teniendo en cuenta el tamaño del dataset empleado.
dic_escala: Las dimensiones de las imágenes para entrar en las redes convolucionales dependen de la arquitectura de transfer learning
            empleada, así que creamos un diccionario con la escala correspondiente a cada red.
dic_preprocesado: Para que las imágenes tengan el mismo aspecto que las que fueron entrenadas inicialmente en cada red de transfer
                  learning, existen unas funciones en el paquete 'applications' de keras que las preprocesan adecuadamente para cada
                  arquitectura. Así que las hemos importado todas y ahora creamos un diccionario para acceder a la qe corresponda en cada                     ejecución.
'''

modelo = getattr(sys.modules[__name__], sys.argv[1])
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
combine_gen() permitirá unir en un solo objeto una lista de generadores. La idea es poder unir los generadores de imágenes que hayamos obtenido de distintos origenes, para utilizar imágenes variadas en un mismo entrenamiento o en una validación o en una predicción. 

Cada generador de la lista de generadores contendrá un array con las imágenes como tal en la primera posición, y otro con sus respectivas etiquetas en la segunda posición. Así que hemos creado un un generador que devolverá un iterable, el cual, en cada iteración, examinará uno de estos generadores guardando las imágenes de la primera posición en un array general denominado "images" y las etiquetas en otro llamado "labels". La idea es que al final tengamos todos los arrays de imágenes de todos los generadores en "images", y lo mismo con las clases en "labels", de hecho esto es lo que devuelve la función como una tupla (images, labels). Cabe mencionar que para unir dos arrays entre sí, lo que hacemos continuamente al unir el array de cada imagen al general, debemso usar la función del paquete numpy conocida concatenate().

Esta función será empleado posteriormente para crear el generador completo de train, val y test, recogiendo imágenes de todos los dispositivos que nos interesen (OCT, iPhone y Samsung) y de todas las clases consideradas (EMD, NO EMD).
''' 

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

'''
num() tiene como objetivo conocer de cuántas imágenes disponemos en total para hacer el entrenamiento y la validación, habiendo quitado la caja de test que empleemos en cada caso. 

Esta función tiene como parámetros de entrada el directorio en el que nos basamos y la caja de test que vamos a emplear, para no contabilizar en ningún caso las imágenes pertenecientes a esa caja, nos interesan el resto. 

Tras quedarse con las cuatro cajas que se utilizarán para train y val, la función recorre, para cada caja, todos los posibles dispositivos de origen, e internamente, las dos clases existentes para cada dispositivo. De esta manera, con tres bucles for concatenados, logra acceder a todas las subcarpetas pertenecientes a cada una de las cuatro cajas. En cada subcarpeta lo que hace es ver cuántas imágenes hay almacenadas. Esto es posible porque la función listdir() del paquete os devuelve una lista con los nombres de los elementos que hay en una ruta que indicamos como parámetro de entrada, y si nosotros pedimos con len() el tamaño de esa lista, ya tenemos lo que buscabamos. 

Una vez que ha guardado en la variable entera "total" esa cantidad final de imágenes asigna cuántas irán para train y cuántas para val. Para ello, consideramos que el 80% están destinadas al entrenamiento y el 20% a la validación. num() devuelve con un return esas dos cantidades de imágenes definidas.

Esta función se empleará posteriormente para los parámetros "steps_per_epoch" y "validation_steps" de la función fit(). Para escoger una cantidad lógica de pasos por época del entrenamiento es necesario conocer la cantidad de imágenes totales empleadas en train. Y, de forma análoga, para conocer la cantidad correcta de pasos de validación es necesario como la cantidad de imágenes reservadas para validar. En concreto, en ambos casos, el número de épocas correcto se calcular dividiendo el total de imágenes entre el batch asignado. Así que eso es lo que haremos en la función transfer_learning(), redondeando los resultados al entero mayor más cercano con math.ceil().

     steps_per_epoch = math.ceil(num(directorio,K_test)[0] / batch)
     math.ceil(num(directorio,K_test)[1] / batch)
'''

def num(directorio,K_test):
    total = 0
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    for K in lista_K:
            for origen in ['MESSIDOR','iPhone','OCT','Samsung']:
                for grado in ['EMD', 'NO EMD']:
                    total += len(os.listdir(directorio + '/' + K + '/' + origen + '/' + grado + '/' + grado + '/'))

    num_train = int(total*0.8)
    num_val = int(total*0.2)
    return num_train,num_val

'''
Creamos un generador de imágenes que emplearemos tanto para el conjunto de entrenamiento como para el de validación. El beneficio que nos proporciona la función ImageDataGenerator de keras es que permite crear generadores de grandes cantidades de datos sin necesidad de que estos se carguen en memoria y, de tal forma, disminuya el rendimiento del algoritmo. Es muy típico usarlo en problemas de aprendizaje profundo.

Esta función tiene muchos posibles parámetros, sobre todo relacionados con el data augmentation que trataremos en capitulos posteriores. Pero ahora solo nos interesa emplear los siguientes:

    - validation_split: proporción de datos del generador que se destinarán a validar.
    - preprocessing_function: función que permite procesar los datos de acuerdo a las características que necesite la red convolucional que       se empleará posteriormente. Estas funciones son las que modelan las imágenes para que queden igual que las que se emplearon                 incialmente con la red de transfer learning que estamos considerando. Nosotros las hemos importado ya del paquete applications de           keras y las hemos guardado en un diccionario con los nombres de la red a la que se refieren como clave. Así que para indicar este           parámetro solo tenemos que acceder a la posición de ese diccionario teniendo en cuenta como clave la red que hayamos indicado en             "modelo".
'''

train_datagen = ImageDataGenerator(
        validation_split = 0.2,
        preprocessing_function = dic_preprocesado[modelo]
)

'''
generador_train() permitirá crear un generador de imágenes considerando las especificaciones generales indicadas en "train_datagen" de una subcarpeta en concreto. O sea, carga las imágenes de la subcarpeta correspondiente a los datos indicados como parámetros, que son: caja, dispositivo de origen y grado. Estos generadores contendrán exlucisvamente las imágenes de entrenamiento.

Esto es posible gracias aplicando el método flow_from_directory() sobre train_datagen. Tiene bastantes parámetros interesantes:

    - directory: ruta de la carpeta en el sistema de archivos que contiene subdirectorios separados para cada clase de imagen. Aquí               consideramos la caja, el origen y el grado que mencionaba anteriormente.
    - target_size: alto y ancho de las imágenes que se cargan, recogemos el valor de escala que hemos asignado al inicio y que depende de la       red que estemos considerando.
    - color_mode = color
    - class_mode: indicamos 'categorical' porque trabajamos con dos clases.
    - batch_size = 2: cantidad de imágenes que se procesan en cada iteración durante el entrenamiento. Nosotros forzamos a que solo se cojan       dos de la carpeta de cada clase porque así logramos un balanceo del batch que solvente el problema del desequilibrio de los datos de         forma manual. Es decir, como hemos elegiremos un batch de entrenamiento de 16, y disponemos de 8 subcarpetas (4 dispositos * 2               clases), hacemos 16/6 para ver cuántas de cada tipo nos conviene coger, y son aproximadamente 2.
    - seed = 42: semilla para el generador aleatorio de números. Esto permite reproducir los mismos resultados en diferentes ejecuciones del       modelo, lo que es útil para la validación y comparación de resultados.
    - subset = 'training': indicamos del generador train_datagen si esta parte que estamos definiendo corresponde al 80% de entrenamiento o       al 20% de validación.
'''
def generador_train(K,origen,grado,escala):
    generator = train_datagen.flow_from_directory(
        directory = directorio + '/' + K + '/' + origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,
        seed = 42,
        subset='training'
    )
    
    return generator

'''
train_gen() sirve para unir los generadores correspondientes a todas las subcarpetas que contienen imágenes que se van a utilizar en el entrenamiento. Como parámetros únicamente deben especificarse: la caja de test para no incluirla en este subset y la escala a la que deben estar las imágenes.

Para ello, lo primero es buscar todas esas subcarpetas, que serán todas las pertenecientes a las cuatro cajas no destinadas a test. Para cada una de estas carpetas cogemos los 5 directorios correspondientes a cada dispositivo empleado, e internamente, los subdirectorios correspondientes a cada una de las dos clases consideradas. En cada uno de estos subdirectorios, se aplicará la función generador_train() previamente definida. Cada uno de los generadores creados en las iteraciones de los tres bucles for concatenados que contiene este código, se almacenará en la lista "generadores".

Como no queremos que estos generador estén de forma aislada en una lista, luego aplicamos la función combine_gen() para obtener un único generador con todas las imágenes de train.
'''

def train_gen(K_test,escala=224):
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    generadores = []
    for K in lista_K:
        for origen in ['MESSIDOR','iPhone','OCT','Samsung']:
            for grado in ['EMD', 'NO EMD']:
                generadores.append(generador_train(K,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado

'''
generador_val() permitirá crear un generador de imágenes considerando las especificaciones generales indicadas en "train_datagen" de una subcarpeta en concreto. O sea, carga las imágenes de la subcarpeta correspondiente a los datos indicados como parámetros, que son: caja, dispositivo de origen y grado. Estos generadores contendrán exlucisvamente las imágenes de validación. 

Esto es posible gracias aplicando el método flow_from_directory() sobre train_datagen. Tiene bastantes parámetros interesantes:

    - directory: ruta de la carpeta en el sistema de archivos que contiene subdirectorios separados para cada clase de imagen. Aquí               consideramos la caja, el origen y el grado que mencionaba anteriormente.
    - target_size: alto y ancho de las imágenes que se cargan, recogemos el valor de escala que hemos asignado al inicio y que depende de la       red que estemos considerando.
    - color_mode = color
    - class_mode: indicamos 'categorical' porque trabajamos con dos clases.
    - batch_size = 2: cantidad de imágenes que se procesan en cada iteración durante el entrenamiento. Nosotros forzamos a que solo se cojan       dos de la carpeta de cada clase porque así logramos un balanceo del batch que solvente el problema del desequilibrio de los datos de         forma manual. Es decir, como hemos elegiremos un batch de entrenamiento de 16, y disponemos de 6 subcarpetas (3 dispositos * 2               clases), hacemos 16/6 para ver cuántas de cada tipo nos conviene coger, y son aproximadamente 2.
    - seed = 42: semilla para el generador aleatorio de números. Esto permite reproducir los mismos resultados en diferentes ejecuciones del       modelo, lo que es útil para la validación y comparación de resultados.
    - subset = 'validation': indicamos del generador train_datagen si esta parte que estamos definiendo corresponde al 80% de entrenamiento       o al 20% de validación.
'''
def generador_val(K,origen,grado,escala):
    generator = train_datagen.flow_from_directory(
        directory = directorio + '/' + K + '/' + origen + '/' + grado,
        target_size = (escala,escala),
        color_mode = color,
        class_mode='categorical',
        batch_size = 1,
        seed = 42,
        subset='validation'
    )
    
    return generator

'''
val_gen() sirve para unir los generadores correspondientes a todas las subcarpetas que contienen imágenes que se van a utilizar en la validación. Como parámetros únicamente deben especificarse: la caja de test para no incluirla en este subset y la escala a la que deben estar las imágenes.

Para ello, lo primero es buscar todas esas subcarpetas, que serán todas las pertenecientes a las cuatro cajas no destinadas a test. Para cada una de estas carpetas cogemos los 3 directorios correspondientes a cada dispositivo empleado, e internamente, los subdirectorios correspondientes a cada una de las dos clases consideradas. En cada uno de estos subdirectorios, se aplicará la función generador_train() previamente definida. Cada uno de los generadores creados en las iteraciones de los tres bucles for concatenados que contiene este código, se almacenará en la lista "generadores".

Como no queremos que estos generador estén de forma aislada en una lista, luego aplicamos la función combine_gen() para obtener un único generador con todas las imágenes de validación.
'''
def val_gen(K_test,escala=224):
    lista_K = ['K1','K2','K3','K4','K5']
    lista_K.remove(K_test)
    generadores = []
    for K in lista_K:
        for origen in ['MESSIDOR','iPhone','OCT','Samsung']:
            for grado in ['EMD','NO EMD']:
                generadores.append(generador_val(K,origen,grado,escala))
                
    generador_combinado = combine_gen(generadores)
    return generador_combinado

'''
transferLearning(K_test,red) entrena una red de transfer learning haciendo predicciones sobre la caja indicada como primer parámetro. Las otras cuatro cajas restantes se destinarán a train y val.

Atributos:

    - K_test: caja con las imágenes que se destinarán para testear.
    - red: arquitectura de transfer learning. Todas las que vamos a considerar son: VGG16, VGG19, Xception, ResNet50V2, ResNet101,                      ResNet152, InceptionV3, InceptionResNetV2, MobileNet, DenseNet121, DenseNet201, EfficientNetB0.

1) Creación de los generadores de imágenes de train, val y test

    En primer lugar, llamamos a las funciones train_gen() y val_gen() para crear los generadores de imágenes de entrenamiento y validación.     Estas funciones tienen como argumentos de entrada la propia caja de test de la función transferLearning() y la escala correspondiente a     la red de transfer learning indicada, que la tenemos en el diccionario "dic_escala".
    
    Y, por otro lado, creamos dos generadores distintos que contendrán las imágenes de test, en un caso de Samsung, y en otro, de iPhone. Lo     primero es llamar a la función ImageDataGenerator() para inicializar el generador. Y, luego, diferenciaremos el generador de test de         Samsung y iPhone ("test_Samsung" y "test_iPhone") llamando a la función flow_from_directory(). Los parámetros de esta función son:
    
        - directory: ruta de la carpeta en el sistema de archivos que contiene subdirectorios separados para cada clase de imagen. Aquí               consideramos la caja de test ya indicada y Samsung o iPhone como dispositivos.
        - target_size: alto y ancho de las imágenes que se cargan, recogemos el valor de escala que hemos asignado al inicio y que depende             de la red que estemos considerando.
        - color_mode = color
        - class_mode: indicamos 'categorical' porque trabajamos con dos clases.
        - batch_size = 2: cantidad de imágenes que se procesan en cada iteración durante el entrenamiento. Nosotros forzamos a que solo se             cojan dos de la carpeta de cada clase porque así logramos un balanceo del batch que solvente el problema del desequilibrio de los           datos de forma manual. 
        - seed = 42: semilla para el generador aleatorio de números. Esto permite reproducir los mismos resultados en diferentes ejecuciones           del modelo, lo que es útil para la validación y comparación de resultados.
        - subset: no lo indicamos porque el generador solo se destinará a imágenes de test.

2) Importación del modelo de transfer learning

    Recogemos la arquitectura de capas convolucionales de la red de transfer_learning indicada como parámetro de entrada, y la guardamos
    en 'base_model'. Como hemos importado todas las redes que nos interesa basta con llamar a la función que tenga el nombre indicado en
    red. Esta función tiene los siguientes parámetros de entrada.
    
        -weights: los pesos obtenidos de entrenamientos anteriores, son valores que se conoce que son adecuados para las máscaras de las                       capas convolucionales. Estos los recogemos del set público 'imagenet', pero como las ejecuciones en Scayle no tienen                         acceso a internet, indicamos en este parámetro 'None'. Luego los importaremos con 'load_weights()' de la carpeta 'Pesos'
                  que he creado donde están esos pesos de 'imagenet' que me he desacargado para no tener que acceder a internet. 
        -include_top: boleano que sirve para indicar si queremos dejar las últimas capas de la arquitectura o no. En nuestro caso queremos
                      adaptar el modelo a un nuevo conjunto de datos con dos clases, así que diseñaremos esas capas después e indicamos                           'False'.
        -input_shape: tamaño de cada uno de los tres canales de las imágenes. Los dos primeros valores los asignamos a la escala                                   correspondiente a la red que nos interesa, que la tenemos en 'dic_escala'. Y en el tercero ponemos '3' porque                               trabajamos con imágenes en color.
    
    En principio, no vamos a guardar los pesos de los nuevos entrenamientos, así que también indicamos 'False' en el atributo trainable.

3) Fine Tuning:

    Como ya he adelantado, vamos a diseñar las últimas capas de la arquitectura de forma acorde al conjunto de datos. Así que como capas         finales creamos una flatten, dos densas de tipo 'relu' y un última densa de tipo 'softmax' indicando que encuentre la clase de cada         imagen evaluada dentro de las dos etiquetas existentes. 
    
    Para tener el modelo completo unimos base_model con estas últimas cuatro capas con la función Sequential() del paquete models.

4) Entrenamiento:

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
       - epochs = 200: cantidad de épocas en las que se dividirá el entrenamiento.
       - validation_split: sirve para separar una fracción de los datos de entrenamiento como conjunto de validación. En este caso hemos              reservado el 20%. Podría indicarse otro conjunto de datos diferente al de entrenamiento, pero en este caso no es lo que nos                  conviene.
       - callbaks: son objetos que se utilizan durante el entrenamiento del modelo para realizar acciones específicas en momentos                    determinados, en este caso solo indicamos "es" por el EarlyStopping que hemos definido.
       - validation_data: generador de imágenes destinadas a validacion, nosotros ya hemos definido este conjunto en "val_generator".
       - steps_per_epoch: cantidad de pasos que seguir en cada época del entrenamiento. Lo más correcto para favorecer la precisión del              modelo evitando el sobreajuste es calcular este parámetro a partir del cociente de la cantidad de imágenes empleadas entre el valor          del batch. Esto lo podemos hacer con la función num() definida previamente.
       - validation_steps: cantidad de pasos que seguir en cada época de la validación. Lo más correcto para favorecer la precisión del              modelo evitando el sobreajuste es calcular este parámetro a partir del cociente de la cantidad de imágenes empleadas entre el valor          del batch. Esto lo podemos hacer con la función num() definida previamente.

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
   
   Este proceso lo repetiremos dos veces, la primera con el conjunto "test_iphone" y la segunda con "test_samsung". Así obtendremos los        resultados para ambos dispositivos, y compararemos el rendimiento del algoritmo con cada uno de ellos. 
'''
def transfer_learning(K_test,red):

    # GENERADORES CON LAS IMÁGENES DE TRAIN / VAL / TEST
    train_generator = train_gen(K_test,escala = dic_escala[red])
    val_generator = val_gen(K_test,escala = dic_escala[red])
    
    test_datagen = ImageDataGenerator(preprocessing_function=dic_preprocesado[red])
    
    test_Samsung = test_datagen.flow_from_directory(
        directory = directorio + '/' + K_test + '/Samsung/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42)

    test_iPhone = test_datagen.flow_from_directory(
        directory = directorio + '/' + K_test + '/iPhone/',
        target_size = (dic_escala[red],dic_escala[red]),
        color_mode = color,
        shuffle = False,
        class_mode='categorical',
        batch_size=1,
        seed = 42)
    
    # IMPORTACIÓN MODELO TRANSFER LEARNING
    base_model = red(weights=None, include_top=False, input_shape=(dic_escala[red],dic_escala[red],3))
    base_model.load_weights('Pesos/' + str(red).split(' ')[1] + '.h5')
    base_model.trainable = False 
    
    # FINE TUNNING
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
    
    # ENTRENAMIENTO
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20,  restore_best_weights=True)

    history = model.fit(
        x = train_generator,
        batch_size=batch,
        epochs=200,
        steps_per_epoch = math.ceil(num(directorio,K_test)[0] / batch),
        callbacks=[es],
        validation_data = val_generator,
        validation_steps = math.ceil(num(directorio,K_test)[1] / batch)
    )
    
    # MÉTRICAS DE EVALUACIÓN
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

'''
Ejecutamos la función transfer_learning() para cada una de las 5 posibles cajas de test. Además este proceso lo realizaremos para cada una de las 12 arquitecturas de transfer learning que estamos considerando y que indicamos como parámetro del script en el archivo .sh. 

Todo este proceso lo ejecutaremos para imágenes preprocesadas y no preprocesadas, cambiando el parámetro "proc" del inicio del script.
'''
for K_test in ['K1', 'K2', 'K3', 'K4', 'K5']:
    transfer_learning(K_test,modelo)

