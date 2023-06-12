#!/usr/bin/env python
# coding: utf-8

## CLASIFICACIÓN EMD / NO EMD IMÁGENES MESSIDOR

## AUTOR: Nuria Velasco Pérez


import pandas as pd  # Importar el módulo pandas con el alias 'pd'
import os  # Importar el módulo os para trabajar con archivos y directorios
import shutil  # Importar el módulo shutil para copiar archivos

# Leer el archivo de Excel "Annotation_Base11.xls" y almacenarlo en el dataframe base11
base11 = pd.read_excel("Annotation_Base11.xls")

display(base11)  # Mostrar el contenido del dataframe base11

lista_edm = []  # Lista para almacenar nombres de imágenes con etiqueta 'edm' igual a 0
lista_no_edm = []  # Lista para almacenar nombres de imágenes con etiqueta 'edm' diferente de 0

# Recorrer las filas del dataframe base11
for e in range(len(base11)):
    name = base11.iloc[e][0]  # Obtener el nombre de la imagen en la fila actual
    edm = base11.iloc[e][2]  # Obtener el valor de la etiqueta 'edm' en la fila actual
    
    if (edm == 0):
        lista_edm.append(name)  # Agregar el nombre a la lista si la etiqueta es 0
    else:
        lista_no_edm.append(name)  # Agregar el nombre a la lista si la etiqueta es diferente de 0

print(lista_edm)  # Imprimir la lista de nombres con etiqueta 'edm' igual a 0
print(lista_no_edm)  # Imprimir la lista de nombres con etiqueta 'edm' diferente de 0

imagenes_base11 = os.listdir('Base11')  # Obtener la lista de imágenes en el directorio 'Base11'

print(imagenes_base11)  # Imprimir la lista de nombres de imágenes en el directorio 'Base11'

# Copiar las imágenes de 'Base11' a los directorios correspondientes en 'MESSIDOR'
for e in imagenes_base11:
    if e in lista_edm:
        shutil.copy("Base11/"+e, "MESSIDOR/EMD/"+e)  # Copiar la imagen a 'MESSIDOR/EMD' si el nombre está en la lista de 'edm'
    else:
        shutil.copy("Base11/"+e, "MESSIDOR/NO EMD/"+e)  # Copiar la imagen a 'MESSIDOR/NO EMD' si el nombre no está en la lista de 'edm'

# Repetir el mismo proceso para las bases de datos Base12, Base13, Base14, Base21, Base22, Base23, Base24, Base31, Base32, Base33, Base34

# Repetir el mismo código para: Base12, Base13, Base14, Base21, Base22, Base23, Base24, Base31, Base32, Base33 y Base34.

