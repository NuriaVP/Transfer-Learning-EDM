## BASELINES RETINÓLOGOS

# AUTOR: Nuria Velasco Pérez

# Importar las bibliotecas necesarias
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Cargar los datos del archivo df_OCT.xlsx en un DataFrame llamado df_oct
df_oct = pd.read_excel('df_OCT.xlsx')

# Mostrar el DataFrame df_oct
display(df_oct)

# Filtrar el DataFrame df_oct eliminando columnas innecesarias
df_oct_filt = df_oct.drop(['Unnamed: 0', '1 OCT 2 IPHONE 3 SAMSUNG', 'CALIDAD GRAL IMAGEN', 'GRADO RETINOPATÍA DIABÉTICA'], axis=1)

# Mostrar el DataFrame filtrado df_oct_filt
display(df_oct_filt)

# Convertir la columna 'Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL' a valores binarios
oct_emd_bin = []

for e in list(df_oct_filt['Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL']):
    if e == 1:
        oct_emd_bin.append(0)
    else:
        oct_emd_bin.append(1)

# Agregar la columna 'EMD binario' al DataFrame df_oct_filt con los valores convertidos
df_oct_filt['EMD binario'] = oct_emd_bin

# Mostrar el DataFrame df_oct_filt con la columna 'EMD binario' agregada
display(df_oct_filt)

# Cargar los datos del archivo df_iPhone.xlsx en un DataFrame llamado df_iphone
df_iphone = pd.read_excel('df_iPhone.xlsx')

# Filtrar el DataFrame df_iphone eliminando columnas innecesarias
df_iphone_filt = df_iphone.drop(['Unnamed: 0', '1 OCT 2 IPHONE 3 SAMSUNG', 'CALIDAD GRAL IMAGEN', 'GRADO RETINOPATÍA DIABÉTICA'], axis=1)

# Convertir la columna 'Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL' a valores binarios
iphone_emd_bin = []

for e in list(df_iphone_filt['Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL']):
    if e == 1:
        iphone_emd_bin.append(0)
    else:
        iphone_emd_bin.append(1)

# Agregar la columna 'EMD binario' al DataFrame df_iphone_filt con los valores convertidos
df_iphone_filt['EMD binario'] = iphone_emd_bin

# Mostrar el DataFrame df_iphone_filt con la columna 'EMD binario' agregada
display(df_iphone_filt)

# Cargar los datos del archivo df_Samsung.xlsx en un DataFrame llamado df_samsung
df_samsung = pd.read_excel('df_Samsung.xlsx')

# Filtrar el DataFrame df_samsung eliminando columnas innecesarias
df_samsung_filt = df_samsung.drop(['Unnamed: 0', '1 OCT 2 IPHONE 3 SAMSUNG', 'CALIDAD GRAL IMAGEN', 'GRADO RETINOPATÍA DIABÉTICA'], axis=1)

# Convertir la columna 'Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL' a valores binarios
samsung_emd_bin = []

for e in list(df_samsung_filt['Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL']):
    if e == 1:
        samsung_emd_bin.append(0)
    else:
        samsung_emd_bin.append(1)

# Agregar la columna 'EMD binario' al DataFrame df_samsung_filt con los valores convertidos
df_samsung_filt['EMD binario'] = samsung_emd_bin

# Mostrar el DataFrame df_samsung_filt con la columna 'EMD binario' agregada
display(df_samsung_filt)

# Resultados iPhone
predict_iphone = []
predict_oct_iphone = []

# Comparar las filas de df_iphone_filt con df_oct_filt basado en ciertos criterios de coincidencia
for e in range(len(df_iphone_filt)):
    bandera = False
    serie = df_iphone_filt.iloc[e]
    nhc = serie['NHC']
    lat = serie['lateralidad 1 Dch 2 izq']
    ret = serie['Retinlogo 1 y 2']
    
    for e in range(len(df_oct_filt)):
        serie_oct = df_oct_filt.iloc[e]
        nhc_oct = serie_oct['NHC']
        lat_oct = serie_oct['lateralidad 1 Dch 2 izq']
        ret_oct = serie_oct['Retinlogo 1 y 2']
        
        if (nhc == nhc_oct) and (lat == lat_oct) and (ret == ret_oct):
            predict_oct_iphone.append(serie_oct['EMD binario'])
            bandera = True
    
    if bandera:
        predict_iphone.append(serie['EMD binario'])

# Imprimir los resultados de la comparación para iPhone
print(predict_iphone)
print(len(predict_iphone))

print(predict_oct_iphone)
print(len(predict_oct_iphone))

print(len(predict_iphone) == len(predict_oct_iphone))

# Resultados Samsung
predict_samsung = []
predict_oct_samsung = []

# Comparar las filas de df_samsung_filt con df_oct_filt basado en ciertos criterios de coincidencia
for e in range(len(df_samsung_filt)):
    bandera = False
    serie = df_samsung_filt.iloc[e]
    nhc = serie['NHC']
    lat = serie['lateralidad 1 Dch 2 izq']
    ret = serie['Retinlogo 1 y 2']
    
    for e in range(len(df_oct_filt)):
        serie_oct = df_oct_filt.iloc[e]
        nhc_oct = serie_oct['NHC']
        lat_oct = serie_oct['lateralidad 1 Dch 2 izq']
        ret_oct = serie_oct['Retinlogo 1 y 2']
        
        if (nhc == nhc_oct) and (lat == lat_oct) and (ret == ret_oct):
            predict_oct_samsung.append(serie_oct['EMD binario'])
            bandera = True
    
    if bandera:
        predict_samsung.append(serie['EMD binario'])

# Imprimir los resultados de la comparación para Samsung
print(predict_samsung)
print(len(predict_samsung))

print(predict_oct_samsung)
print(len(predict_oct_samsung))

print(len(predict_samsung) == len(predict_oct_samsung))

# Resultados estadísticos para iPhone
from sklearn import metrics

# Calcular la precisión (accuracy) para iPhone
accuracy_ret_IPhone = metrics.accuracy_score(predict_oct_iphone, predict_iphone)
print(accuracy_ret_IPhone)

from sklearn.metrics import f1_score

# Calcular el F1 score para iPhone
f1_ret_IPhone = f1_score(predict_oct_iphone, predict_iphone, average='weighted')
print(f1_ret_IPhone)

from sklearn.metrics import roc_auc_score

# Calcular el área bajo la curva ROC (ROC AUC) para iPhone
auc_ret_IPhone = roc_auc_score(predict_oct_iphone, predict_iphone)
print(auc_ret_IPhone)

# Resultados estadísticos para Samsung

# Calcular la precisión (accuracy) para Samsung
accuracy_ret_Samsung = metrics.accuracy_score(predict_oct_samsung, predict_samsung)
print(accuracy_ret_Samsung)

# Calcular el F1 score para Samsung
f1_ret_Samsung = f1_score(predict_oct_samsung, predict_samsung, average='weighted')
print(f1_ret_Samsung)

# Calcular el área bajo la curva ROC (ROC AUC) para Samsung
auc_ret_Samsung = roc_auc_score(predict_oct_samsung, predict_samsung)
print(auc_ret_Samsung)

