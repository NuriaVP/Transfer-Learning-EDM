
# # CLASIFICAR DE FORMA BINARIA LOS PACIENTES POR EMD

# AUTOR: Nuria Velasco Pérez

# Necesitamos tener clasificadas las imágenes en subcarpetas para cada aparato en función a si el paciente tiene o no tiene edema macular diabético, vamos a crear un problema binario.

# Primero importaremos los dataframes por separado de los tres aparatos que hemos logrado en el preprocesamiento

df_OCT = pd.read_excel('df_OCT.xlsx')
df_OCT = df_OCT.drop(['Unnamed: 0'], axis=1)

display(df_OCT)

df_IPhone = pd.read_excel('df_iPhone.xlsx')
df_IPhone = df_IPhone.drop(['Unnamed: 0'], axis=1)

df_Samsung = pd.read_excel('df_Samsung.xlsx')
df_Samsung = df_Samsung.drop(['Unnamed: 0'], axis=1)

'''
Empezamos por hacer los dos grupos para las imágenes de OCT.
'''
columna_emd = list(df_OCT.iloc[:, 6])
    
columna_NHC = list(df_OCT.iloc[:, 0])
    
OCT_EMD_1 = []
                       
contador = 0
    
while(contador < len(columna_emd)):
        
    if(columna_emd[contador] == 1):
        OCT_EMD_1.append(columna_NHC[contador])
    
    contador+= 1
    
print(set(OCT_EMD_1))

columna_emd = list(df_OCT.iloc[:, 6])
    
columna_NHC = list(df_OCT.iloc[:, 0])
    
OCT_CON_EMD = []
                       
contador = 0
    
while(contador < len(columna_emd)):
        
    if(columna_emd[contador] != 1):
        OCT_CON_EMD.append(columna_NHC[contador])
    
    contador+= 1
    
print(set(OCT_CON_EMD))

'''
Continuamos por crear los dos grupos para las imágenes de iPhone
'''
columna_emd = list(df_IPhone.iloc[:, 6])
    
columna_NHC = list(df_IPhone.iloc[:, 0])
    
IPhone_EMD_1 = []
                       
contador = 0
    
while(contador < len(columna_emd)):
        
    if(columna_emd[contador] == 1):
        IPhone_EMD_1.append(columna_NHC[contador])
    
    contador+= 1
    
print(set(IPhone_EMD_1))


columna_emd = list(df_IPhone.iloc[:, 6])
    
columna_NHC = list(df_IPhone.iloc[:, 0])
    
IPhone_CON_EMD = []
                       
contador = 0
    
while(contador < len(columna_emd)):
        
    if(columna_emd[contador] != 1):
        IPhone_CON_EMD.append(columna_NHC[contador])
    
    contador+= 1
    
print(set(IPhone_CON_EMD))

'''
Terminamos con los grupos de las imágenes Samsung.
'''
columna_emd = list(df_Samsung.iloc[:, 6])
    
columna_NHC = list(df_Samsung.iloc[:, 0])
    
Samsung_EMD_1 = []
                       
contador = 0
    
while(contador < len(columna_emd)):
        
    if(columna_emd[contador] == 1):
        Samsung_EMD_1.append(columna_NHC[contador])
    
    contador+= 1
    
print(set(Samsung_EMD_1))


columna_emd = list(df_Samsung.iloc[:, 6])
    
columna_NHC = list(df_Samsung.iloc[:, 0])
    
Samsung_CON_EMD = []
                       
contador = 0
    
while(contador < len(columna_emd)):
        
    if(columna_emd[contador] != 1):
        Samsung_CON_EMD.append(columna_NHC[contador])
    
    contador+= 1
    
print(set(Samsung_CON_EMD))


# Ahora vamos a obtener una lista con los nombres correspondientes a las imágenes seleccionadas para cada grupo para cada aparato, para poder crear los directorios y mover estos nombres de las imágenes al que corresponda automáticamente.

'''
nombre_imagenes() devolverán la lista de nombres de las imágenes correspondientes a ese grupo que queremos mover
'''
def nombres_imagenes_OCT(set_NHC):
    
    nombres = []
    
    for i in set_NHC:
        nombres.append(str(i)+'TI.jpg')
        nombres.append(str(i)+'TD.jpg')
        
    return nombres

OCT_NO_EMD = nombres_imagenes_OCT(OCT_EMD_1)

print(OCT_NO_EMD)

print(len(OCT_NO_EMD))


OCT_EMD = nombres_imagenes_OCT(OCT_CON_EMD)

print(OCT_EMD)

print(len(OCT_EMD))


def nombres_imagenes_IPhone(set_NHC):
    
    nombres = []
    
    for i in set_NHC:
        nombres.append(str(i)+'EI.PNG')
        nombres.append(str(i)+'ED.PNG')
        
    return nombres

iPhone_NO_EMD = nombres_imagenes_IPhone(IPhone_EMD_1)

print(iPhone_NO_EMD)

print(len(iPhone_NO_EMD))

iPhone_EMD = nombres_imagenes_IPhone(IPhone_CON_EMD)

print(iPhone_EMD)

print(len(iPhone_EMD))

def nombres_imagenes_Samsung(set_NHC):
    
    nombres = []
    
    for i in set_NHC:
        nombres.append(str(i)+'GI.png')
        nombres.append(str(i)+'GD.png')
        
    return nombres

Samsung_NO_EMD = nombres_imagenes_Samsung(Samsung_EMD_1)

print(Samsung_NO_EMD)

print(len(Samsung_NO_EMD))

Samsung_EMD = nombres_imagenes_Samsung(Samsung_CON_EMD)

print(Samsung_EMD)

print(len(Samsung_EMD))

# Utilizamos las listas con los subgrupos que hemos creado para generar las carpetas correspondientes.

fotos_OCT = os.listdir('Datos EMD\OCT')
fotos_iPhone = os.listdir('Datos EMD\iPhone')
fotos_samsung = os.listdir('Datos EMD\Samsung')

'''
Clasificación imágenes OCT.
'''
for i in fotos_OCT:
    if i[-4:] == '.jpg':
        if i in OCT_NO_EMD:
            shutil.move('Datos EMD\\OCT\\' + i, 'Datos EMD\\OCT\\NO EMD\\' + i)
        elif i in OCT_EMD:
            shutil.move('Datos EMD\\OCT\\' + i, 'Datos EMD\\OCT\\EMD\\' + i)
        else:
            os.remove('Datos EMD\\OCT\\' + i)

'''
Clasificación imágenes iPhone.
'''
for i in fotos_iPhone:
    if i[-4:] == '.PNG':
        if i in iPhone_NO_EMD:
            shutil.move('Datos EMD\\iPhone\\' + i, 'Datos EMD\\iPhone\\NO EMD\\' + i)
        elif i in iPhone_EMD:
            shutil.move('Datos EMD\\iPhone\\' + i, 'Datos EMD\\iPhone\\EMD\\' + i)
        else:
            os.remove('Datos EMD\\iPhone\\' + i)

'''
Clasificación imágenes Samsung.
'''
for i in fotos_samsung:
    if i[-4:] == '.png':
        if i in Samsung_NO_EMD:
            shutil.move('Datos EMD\\Samsung\\' + i, 'Datos EMD\\Samsung\\NO EMD\\' + i)
        elif i in Samsung_EMD:
            shutil.move('Datos EMD\\Samsung\\' + i, 'Datos EMD\\Samsung\\EMD\\' + i)
        else:
            os.remove('Datos EMD\\Samsung\\' + i)

