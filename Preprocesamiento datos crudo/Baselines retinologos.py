#!/usr/bin/env python
# coding: utf-8

# # Baselines retinologos

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


# ## OCT

# In[3]:


df_oct = pd.read_excel('df_OCT.xlsx')


# In[4]:


display(df_oct)


# In[5]:


df_oct_filt = df_oct.drop(['Unnamed: 0', '1 OCT 2 IPHONE 3 SAMSUNG', 'CALIDAD GRAL IMAGEN', 'GRADO RETINOPATÍA DIABÉTICA'], axis=1)


# In[6]:


display(df_oct_filt)


# In[7]:


oct_emd_bin = []

for e in list(df_oct_filt['Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL']):
    if e==1:
        oct_emd_bin.append(0)
    else:
        oct_emd_bin.append(1)


# In[8]:


print(oct_emd_bin)


# In[9]:


df_oct_filt['EMD binario'] = oct_emd_bin


# In[10]:


display(df_oct_filt)


# ## iPhone

# In[11]:


df_iphone = pd.read_excel('df_iPhone.xlsx')


# In[12]:


df_iphone_filt = df_iphone.drop(['Unnamed: 0', '1 OCT 2 IPHONE 3 SAMSUNG', 'CALIDAD GRAL IMAGEN', 'GRADO RETINOPATÍA DIABÉTICA'], axis=1)


# In[13]:


iphone_emd_bin = []

for e in list(df_iphone_filt['Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL']):
    if e==1:
        iphone_emd_bin.append(0)
    else:
        iphone_emd_bin.append(1)


# In[14]:


df_iphone_filt['EMD binario'] = iphone_emd_bin


# In[15]:


display(df_iphone_filt)


# ## Samsung

# In[16]:


df_samsung = pd.read_excel('df_Samsung.xlsx')


# In[17]:


df_samsung_filt = df_samsung.drop(['Unnamed: 0', '1 OCT 2 IPHONE 3 SAMSUNG', 'CALIDAD GRAL IMAGEN', 'GRADO RETINOPATÍA DIABÉTICA'], axis=1)


# In[18]:


samsung_emd_bin = []

for e in list(df_samsung_filt['Clasificación EMD. 1 NO . 2 NO CENTRAL, 3 CENTRAL']):
    if e==1:
        samsung_emd_bin.append(0)
    else:
        samsung_emd_bin.append(1)


# In[20]:


df_samsung_filt['EMD binario'] = samsung_emd_bin


# In[21]:


display(df_samsung_filt)


# ## Resultados iPhone

# In[32]:


predict_iphone = []
predict_oct_iphone = []

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
        
        if (nhc==nhc_oct) and (lat==lat_oct) and (ret==ret_oct):
            predict_oct_iphone.append(serie_oct['EMD binario'])
            bandera = True
    
    if (bandera==True):
        predict_iphone.append(serie['EMD binario'])


# In[33]:


print(predict_iphone)
print(len(predict_iphone))


# In[34]:


print(predict_oct_iphone)
print(len(predict_oct_iphone))


# In[35]:


print(len(predict_iphone)==len(predict_oct_iphone))


# ## Resultados Samsung

# In[36]:


predict_samsung = []
predict_oct_samsung = []

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
        
        if (nhc==nhc_oct) and (lat==lat_oct) and (ret==ret_oct):
            predict_oct_samsung.append(serie_oct['EMD binario'])
            bandera = True
    
    if (bandera==True):
        predict_samsung.append(serie['EMD binario'])


# In[37]:


print(predict_samsung)
print(len(predict_samsung))


# In[38]:


print(predict_oct_samsung)
print(len(predict_oct_samsung))


# In[40]:


print(len(predict_samsung)==len(predict_oct_samsung))


# # Resultados estadísticos

# # iPhone

# In[44]:


from sklearn import metrics

accuracy_ret_IPhone = metrics.accuracy_score(predict_oct_iphone, predict_iphone)

print(accuracy_ret_IPhone)


# In[46]:


from sklearn.metrics import f1_score

f1_ret_IPhone = f1_score(predict_oct_iphone, predict_iphone, average='weighted')

print(f1_ret_IPhone)


# In[48]:


from sklearn.metrics import roc_auc_score

auc_ret_IPhone = roc_auc_score(predict_oct_iphone, predict_iphone)

print(auc_ret_IPhone)


# # Samsung

# In[49]:


accuracy_ret_Samsung = metrics.accuracy_score(predict_oct_samsung, predict_samsung)

print(accuracy_ret_Samsung)


# In[50]:


f1_ret_Samsung = f1_score(predict_oct_samsung, predict_samsung, average='weighted')

print(f1_ret_Samsung)


# In[51]:


auc_ret_Samsung = roc_auc_score(predict_oct_samsung, predict_samsung)

print(auc_ret_Samsung)

