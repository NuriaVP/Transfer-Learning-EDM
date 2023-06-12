#!/usr/bin/env python
# coding: utf-8

# # PROYECTO EDEMA MACULAR DIABÉTICO - PREPROCESAMIENTO

# **AUTOR: Nuria Velasco Pérez**

# En este cuaderno vamos a preparar el conjunto de datos de unos pacientes con **posible diagnóstico de retinopatía** para 
# posteriormente poder emplear la información en una red convolucional. Así que resulta interesante entender ante qué datos nos 
# enfrentamos para ser capaces de seguir los pasos que he desarrollado.
# 
# Tenemos **12 filas** para cada paciente porque se ha fotografiado tanto el ojo derecho como el ojo izquierdo de cada uno de ellos con tres aparatos diferentes: el OCT, un IPhone y un Samsung. Además tenemos el veredicto para cada una de estas imágenes de dos retinológos, por eso se genera un total de 12 fotos por inidividuo. Cabe mencionar que el estudio va enfocado a intentar entrenar neuronas artificiales para que sean capaces de dar el mismo diagnóstico con una fotografía de un IPhone o un Samsung
# que el se puede dar con una imagen de OCT según los retinologos. Por ello, consideramos que nuestro Gold Estándar es el OCT. La clasificación del aparato empleado, el ojo y el retinologo aparece en las primeras columnas; teniendo en cuenta que la primera corresponde al número de historia clínica (NHC) de los pacientes que entran en el estudio.
# 
# También resulta interesante conocer que las columnas intermedias están destinadas a determinar ciertos parámetros específicos de la calidad de las imágenes empleadas, no serán esenciales para nosotros, al menos inicialmente. Y en los últimos atributos corresponden a las variables que se quieren predecir.
# 
# - **Grado de Retinopatía** que puede tomar valores entre el 1 y el 5, de menor a mayor gravedad.
# 
# 
# - **Clasificación EMD** pretende dar una clasificación del edema macular diabético dando valores entre 1 y 3. Siendo 1 la posibilidad de no tener edema, 2 cuando este no es central y 3 cuando es central.

# # TAREA 1: Procesamiento y filtrado de los datos

# ## PASO 1: Importación de los datos

# **PAQUETE PANDAS**

# In[1]:


'''
Importar el paquete pandas. Es común utilizar el alias pd para abreviar el nombre del paquete.
'''
import pandas as pd


# In[2]:


'''
Leer el archivo excell con el método ExcelFile() de pandas.
Previamente he guardado el documento de excell en la misma carpeta que este notebook.
'''
file = pd.ExcelFile('RETICAM ANALISIS ESTADÍSTICO.xlsx')


# In[3]:


'''
Conocer la cantidad de hojas y sus respectivos nombre, para ver con cuál queremos trabajar. Existe el método sheet_names().
'''
print(file.sheet_names)


# In[4]:


'''
Seleccionar únicamente la hoja denominado 'Resultados' con el método parse() que te devuelve el conjunto de datos como un 
DataFrames.
'''
df = file.parse('Resultados')

print(type(df))


# **OBTENCIÓN DATAFRAME**

# In[5]:


'''
Visualización características de la tabla completa df.
'''
display(df) #Ver tabla completa


# **Características generales DataFrame**

# In[6]:


'''
Ver las dimensiones (filas, columnas) utilizamos el método shape().
'''
print(df.shape) 


# El conjunto de datos con el que queremos trabajar tiene **25 columnas y 968 filas**.
# 
# Sin embargo de estas 25 columnas no necesitamos todas porque entre al 10 y la 23 solo describen algunos signos clínicos de ciertos grados de retinopatía que no son de interés para este estudio, y además no están completos para la mayor parte de las observaciones. Y de la 5 a la 9 se desglosan características de la calidad de cada parte de la imagen que no son de nuestro interés.

# In[7]:


'''
Eliminamos de la columna 5 a la 9 porque no las voy a considerar para corroborar la completitud de los datos ya que describen
en distintos tipos la calidad de la imagen y vamos a evitar considerarlo para simplificar el problema, lo hacemos con
el método drop() indicando el rango de columnas deseado.
También eliminamos de la 10 a la 20 porque describen signos específicos del grado de retinopatía que no están completos en muchos
casos y no los valoraremos.
He tenido en cuenta que en Python se empiezan a enumerar las posiciones desde el 0.
'''

df = df.drop(df.columns[5:9], axis='columns') # axis=1 -> columnas
df = df.drop(df.columns[6:20], axis='columns') # axis=1 -> columnas

display(df)


# **Comprobación de que de las 969 filas un tercio pertenece a cada aparato**

# Por profundizar el conocimiento con respecto al contenido inicial del conjunto de datos vamos a comprobar si realmente las filas están divididas a partes iguales en los 3 aparatos con los que trabajaremos. De no ser así quiere decir que hay muchas anomalías en el contenido de df y que debemos ir filtrando por los críterios que nos interesan.

# In[8]:


'''
Hay 969 filas si dividimos entre 3; 323 deberían pertenecer a cada aparato.
Dataframe de las de OCT -> columna 3 = 1.
Utilizamos rebanadas para acceder a cada columna indicando su nombre entre [].
''' 
is_OCT = df['1 OCT 2 IPHONE 3 SAMSUNG'] == 1
df_OCT = df[is_OCT]
df_OCT.head()
print(len(df_OCT)) #Deberían de ser 323


# In[9]:


'''
Hay 969 filas si dividimos entre 3; 323 deberían pertenecer a cada aparato.
Dataframe de las de IPhone -> columna 3 = 2.
'''
is_IPhone = df['1 OCT 2 IPHONE 3 SAMSUNG'] == 2
df_IPhone = df[is_IPhone]
df_IPhone.head()
print(len(df_IPhone)) #Deberían de ser 323


# In[10]:


'''
Hay 969 filas si dividimos entre 3; 323 deberían pertenecer a cada aparato.
Dataframe de las de Samsung -> columna 3 = 3
'''
is_Samsung = df['1 OCT 2 IPHONE 3 SAMSUNG'] == 3
df_Samsung = df[is_Samsung]
df_Samsung.head()
print(len(df_Samsung)) #Deberían de ser 323
print(328 + 321 + 319)


# In[11]:


'''
Vamos a comprobar cuantos NHC diferentes tenemos inicialmente recogiéndolos todos en una lista porque aparecen en la primera
columnas, y convirtiendo la misma en un set() para que no salgan ocurrencias repetidas.
TENEMOS 81 PACIENTES INICIALMENTE.
'''
print(len(set(df['NHC']))) 


# Efectivamente como nos esperábamos parece que la información no está bien repartida y habrá que ir eliminando a los pacientes que generan esa disturbancia progresivamente.
# 
# Pesé a esto nos interesa establecer que inicialmente tenemos **81 PACIENTES**.

# ## PASO 2: Eliminar pacientes que no tienen las filas completas

# #### Encontrar posiciones filas con celdas incompletas (NaN)

# El objetivo es encontrar la posición de las filas que tienen algún valor desconocido en alguna columna, para ello recorreremos todas las columnas y guardaremos en listas las posiciones identificadas con valores desconocidos. Luego crearemos una gran lista que contendrá las sublistas de las posiciones vacías en cada columna.

# In[12]:


'''
Obtener una lista con el nombre de los 11 atributos.
'''
list(df.columns)


# In[13]:


'''
Hay dos columnas que no tienen el mismo tipo de datos que el resto, son float mientras que el resto son int.
'''
columnas = list(df.columns)

for e in columnas:
    print(type(df[e][0]))


# In[14]:


'''
Probamos a rellenar todos los valores desconocidos con '0' para poder encontrarlos mejor, debido a la incompatibilidad de tipos.
El método fillna() rellena todas las posiciones detectadas como NaN con el valor indicado como parámetro.
'''
df = df.fillna(0)

display(df)


# In[15]:


'''
lista_valores_desconocidos() recorrerá todos los valores para cada columna y recogerá la posición de las celdas incompletas.
Los valores vacíos son de tipo NaN.
'''
def lista_valores_desconocidos(df):
    '''
    Crear lista con todos los atributos que recorreremos.
    '''
    columnas = list(df.columns)
    '''
    Crear contadores para evitar bucles infinitos, cont permitirá recorrer todas las columnas por su posición.
    '''
    cont_inicio = 0
    cont_fin = len(list(columnas))
    cont = cont_inicio
    '''
    Lista conocidos será el resultado de la función y será una lista con tantas sublistas como columnas hay, cada sublista
    almacenará las posiciones donde si que conocemos el valor.
    '''
    lista_desconocidos = list()
    '''
    While recorrerá todas las columnas y cuando llegue a la última saldrá.
    '''
    while (cont < cont_fin):
        '''
        Quedarnos solo con los valores de la columna en cuestión
        '''
        columna = list(df[columnas[cont]])
        '''
        Inicializar lista que contendrá esas posiciones donde no que conocemos el valor.
        '''
        lista_columna_desconocidos = list()
        
        cont_posicion = 0

        while (cont_posicion < len(columna)):
            '''
            Comprobar para cada posición si hay un valor conocido. isnan() devuelve True si el valor es desconocidos, en este
            caso buscamos los False.
            '''
            if(columna[cont_posicion] == 0): 
                
                lista_columna_desconocidos.append(cont_posicion)
                
            cont_posicion += 1
            
        '''
        Guardar como sublista la lista con las posiciones que buscabamos.
        '''
        lista_desconocidos.append(lista_columna_desconocidos)
        '''
        Pasamos a la siguiente columna, para que no sea un bucle infinito.
        '''
        cont += 1
    
    return lista_desconocidos #Lista de listas


# In[16]:


'''
Utilizamos esta función con el df.
'''
df_valores_desconocidos = lista_valores_desconocidos(df)

print(df_valores_desconocidos)


# #### Generar una única lista con todas las posiciones de las filas incompletas a eliminar

# In[17]:


'''
La función ordenar_sublistas() ordena posiciones de cada sublista a eliminar.
'''
def ordenar_sublistas(lista_desconocidos_total):
    
    lista_ordenada = list()
    cont = 0
    cont_fin = len(lista_desconocidos_total)
    
    '''
    Ordenar cada sublista  de la lista de desconocidos y añadir a otra lista de listas nueva.
    '''
    while (cont < cont_fin):
        sublista = lista_desconocidos_total[cont]
        '''
        La función sorted() ordena una lista.
        '''
        lista_ordenada.append(sorted(sublista))
        cont += 1
    
    return lista_ordenada #Lista de listas


# In[18]:


'''
Utilizamos esta función la lista df_valores_desconocidos.
'''
df_valores_desconocidos_ordenada = ordenar_sublistas(df_valores_desconocidos)

'''
Podemos eliminar las cuatro primeras posiciones porque parece que esas columnas están completas y no muestran ninguna 
posición a eliminar.
'''
df_valores_desconocidos_ordenada.pop(0)
df_valores_desconocidos_ordenada.pop(0)
df_valores_desconocidos_ordenada.pop(0)
df_valores_desconocidos_ordenada.pop(0)
df_valores_desconocidos_ordenada.pop(0)

print(df_valores_desconocidos_ordenada)
print(len(df_valores_desconocidos_ordenada))


# In[19]:


'''
La función unir_lista_posiciones() fusiona en una única lista todas las posiciones detectadas como desconocidas que queremos 
eliminar por falta de información.
'''
def unir_lista_posiciones(lista_desconocidos_total_ordenada):
    
    lista_resultado = list()
    
    cont = 0
    cont_fin = len(lista_desconocidos_total_ordenada)
    '''
    Recorrer todas las posiciones almacenadas en las sublistas para añadir una por una si no se ha añadido antes en una 
    única lista.
    '''
    while (cont < cont_fin):
        
        sublista = lista_desconocidos_total_ordenada[cont]
        
        for i in sublista:
            if i not in lista_resultado:
                lista_resultado.append(i)
        
        cont += 1
    
    return lista_resultado


# In[20]:


'''
Utilizamos esta función la lista df_valores_desconocidos_ordenada.
'''
lista_desconocidos_def = unir_lista_posiciones(df_valores_desconocidos_ordenada)

lista_desconocidos_def = sorted(lista_desconocidos_def)

print(lista_desconocidos_def)


# #### Encontrar todas las filas correspondientes a los NHC que no están completos

# In[21]:


'''
Buscamos los NHC correspondientes a los pacientes que tienen filas incompletas.
'''
lista_NHC_eliminar = []

for e in lista_desconocidos_def:
    lista_NHC_eliminar.append(df.iloc[e, 0])
    
lista_eliminar = list(set(lista_NHC_eliminar))

print(lista_eliminar)


# In[22]:


'''
La función posiciones_eliminar() busca todas las posiciones de filas correspondientes a estos individuos porque debemos 
eliminarlas, al no tener la información completa.
'''
def posiciones_eliminar(lista):
    '''
    Obtenemos todo el conjunto de NHC que está en la primera columna
    '''
    columna_NHC = list(df['NHC'])

    cont = 0
    '''
    Creamos una lista salida que contendrá todas las posiciones de filas correspondientes a pacientes detectados como incorrectos
    que queremos eliminar.
    '''
    lista_posiciones_totales = []
    '''
    Saldrá del while cuando se hayan analizado todos los NHC del df.
    '''
    while(cont < len(columna_NHC)):
        '''
        Comprobamos para cada NHC si está presente entre los NHC a eliminar que hemos indicado en la lista de parámetro.
        Esto se consigue con la operación de pertenencia "in".
        En caso afirmativo guardamos su posición para posteriormente eliminar esa fila.
        '''
        if (columna_NHC[cont] in lista):
            lista_posiciones_totales.append(cont)

        cont+= 1
        
    return lista_posiciones_totales


# In[23]:


'''
Utilizar posiciones_eliminar() con lista_eliminar. Y obtenemos 146 filas a eliminar.
'''
lista_posiciones_totales = posiciones_eliminar(lista_eliminar)
print(lista_posiciones_totales)


# #### Eliminar todas las filas de pacientes incompletos

# In[24]:


'''
Eliminamos todas las filas detectadas como incompletas, esto lo conseguimos con el método drop().
Hemos seleccionado a 12 pacientes para eliminar por falta de datos y teníamos 81 deberíamos quedarnos con 69.
'''
df = df.drop(df.index[lista_posiciones_totales])

display(df)


# ## PASO 3: eliminar pacientes que no tienen las 12 filas

# Con este paso pretendemos eliminar todas las filas correspondientes a aquellos NHC que no tengan las 12 filas necesarias para tener su información completa.

# In[25]:


'''
TENEMOS 69 PACIENTES
'''
print(len(set(df['NHC']))) 


# In[26]:


'''
Buscamos los NHC de los pacientes que no tiene 12 filas a partir de la cantidad de veces que aparece cada NHC en la primera 
columna de df.
'''
columna_NHC = list(df['NHC'])

'''
Recogemos en un set todos los posibles NHC que nos quedan de los 69 pacientes.
'''
lista_NHC = list(set(columna_NHC))
'''
Creamos la lista que almacenará los NHC de los pacientes que no tienen 12 filas.
'''
lista_no_12 = []
'''
Para cada NHC de los 69 pacientes comprobamos si aparece 12 veces en df.
'''
for e in lista_NHC:
    '''
    El  método count() cuenta la cantidad de veces que aparece un valor en una lista.
    '''
    if (columna_NHC.count(e) != 12):
        
        lista_no_12.append(e)

print(lista_no_12)


# In[27]:


'''
Buscamos todas las posiciones de filas correspondientes a estos individuos porque debemos eliminarlas, al no tener la su
información completa.
'''
eliminar_posiciones_totales = posiciones_eliminar(lista_no_12)
print(eliminar_posiciones_totales)
print(len(eliminar_posiciones_totales))


# #### Eliminar todas las filas de pacientes incompletos

# In[28]:


'''
Eliminamos todas las filas detectadas como pertencientes a pacientes que no tienen las 12 necesarias.
Hemos seleccionado a 5 pacientes para eliminar y teníamos 69 deberíamos quedarnos con 64 pacientes.
'''
df = df.drop(df.index[eliminar_posiciones_totales])

df


# In[29]:


'''
TENEMOS 64 PACIENTES
'''
lista_NHC_filtrados = sorted(list(set(df['NHC'])))
print(lista_NHC_filtrados)
print(len(lista_NHC_filtrados))


# ## PASO 4: Eliminar pacientes con la columna del aparato, del ojo o del retinologo mal distribuida

# Ahora vamos a corroborar si las 12 filas de cada NHC tienen la información de las columnas relativas al aparato, el ojo y el retinologo bien distribuidas. Para que la información sea correcta cada uno debería tener 4 filas de cada tipo de aparato (OCT, IPhone y Samsung), 6 filas para cada ojo y 6 filas para cada retinologo.

# In[30]:


'''
Ejemplo NHC con la información correctamente distribuida para las tres columnas de interés.
'''
is_num = df['NHC'] == lista_NHC_filtrados[0]

df_num = df[is_num]

display(df_num)

print(len(df_num))


# In[31]:


'''
errores_distribuicion() comprueba que las 12 columnas de un NHC tengan donde corresponde los 3 aparatos, los 2 ojos y los 
2 retinologos.
'''
def errores_distribucion(df, num):
    '''
    Nos quedamos solo con las 12 filas del NHC que se introduce como parámetro.
    '''
    is_num = df['NHC'] == num

    df_num = df[is_num]
    '''
    Creamos como listas cada una de las columnas que vamos a analizar.
    '''
    columna_aparato = list(df_num['1 OCT 2 IPHONE 3 SAMSUNG'])

    columna_ojo = list(df_num['lateralidad 1 Dch 2 izq'])

    columna_retinologo = list(df_num['Retinlogo 1 y 2'])
    '''
    Creamos variables boleanas que guardarán el resultado de si está correctamente distribuida o no la información.
    '''
    val_aparato = False

    val_ojo = False

    val_retinolog = False

    val = False
    '''
    Vamos viendo en cada una de las columnas que si se cumplen las condiciones mencionadas:
    - Para el aparato tiene que haber 4 de OCT, 4 de IPhone y 4 de Samsung.
    - Para los ojo 6 tienen que corresponder al derecho y 6 al izquierdo.
    - Para el retinologo 6 tienen que corresponder al 1 y 6 al 2.
    '''

    if (columna_aparato.count(1) == 4 and columna_aparato.count(2) == 4 and columna_aparato.count(2) == 4):
        val_aparato = True

    if (columna_ojo.count(1) == 6 and columna_ojo.count(2) == 6):
        val_ojo = True

    if (columna_retinologo.count(1) == 6 and columna_retinologo.count(2) == 6):
        val_retinologo = True

    if (val_aparato == True and val_ojo == True and val_retinologo == True):
        val = True
    '''
    Devolvemos una variable boleana general que solo es True si las otras tres lo son.
    '''
    return val 


# In[32]:


'''
todos_erroneos() ejecuta la función errores_distribucion() con cada NHC de los filtrados porque son completos. Es capaz de 
devolver una lista con los NHC de pacientes que no presenta la distribución prevista.
'''
def todos_erroneos(df, lista_NHC_filtrados):
    
    cont = 0
    '''
    Esta lista contendrá los NHC de pacientes que no presenta la distribución prevista.
    '''
    lista_erroneos = []
    '''
    Sale cuando ya se ha evaluado la distribución de los 64 pacientes presentes en df.
    '''
    while (cont < len(lista_NHC_filtrados)):
        '''
        Se añade el NHC del paciente si resulta False para errores_distribucion().
        '''
        if (errores_distribucion(df, lista_NHC_filtrados[cont]) == False):
            lista_erroneos.append(lista_NHC_filtrados[cont])
        
        cont += 1
    
    return lista_erroneos


# In[33]:


'''
Utilizamos todos_erroneos() con lista_NHC_filtrados.
'''
todos_erroneos(df, lista_NHC_filtrados)


# In[34]:


'''
Buscamos todas las posiciones de filas correspondientes a estos individuos porque debemos eliminarlas, al no tener la información 
bien distribuida.
'''
eliminar_posiciones = posiciones_eliminar(todos_erroneos(df, lista_NHC_filtrados))
print(eliminar_posiciones)
print(len(eliminar_posiciones))


# In[35]:


'''
Eliminamos todas las filas detectadas como pertencientes a pacientes que no tienen las 12 filas bien distribuidas.
Hemos seleccionado a 4 pacientes para eliminar y teníamos 64 deberíamos quedarnos con 60 pacientes.
'''
df = df.drop(df.index[eliminar_posiciones])

df


# In[36]:


'''
TENEMOS 60 PACIENTES DEFINITIVAMENTE FILTRADOS
'''
lista_NHC_filtrados = sorted(list(set(df['NHC'])))
print(lista_NHC_filtrados)
print(len(lista_NHC_filtrados))


# ## PASO 5: Buscar la completitud de imágenes de los pacientes filtrados

# Vamos a comprobar si todos los pacientes seleccionados como completos y con una correcta distribución de la información tienen
# las 6 fotos que necesitamos para realziar el estudio. Requerimos que cada carpeta correspondiente a cada aparato contenga al menos
# 2 imágenes de cada paciente, una correspondiente a cada ojo. 

# In[218]:


'''
Así que lo primero vamos a cosntruir listas con el nombre de todas las imágenes que tiene cada una de las 3 carpetas.
'''
import os

contenido_fotos_IPhone = os.listdir('C:/Users/usuario/Dropbox/PC/Documents/INGENIERIA DE LA SALUD/4º UNI GIS/TFG/repeticion preprocesamiento/Datos EMD/iPhone')

contenido_fotos_OCT = os.listdir('C:/Users/usuario/Dropbox/PC/Documents/INGENIERIA DE LA SALUD/4º UNI GIS/TFG/repeticion preprocesamiento/Datos EMD/OCT')

contenido_fotos_Samsung = os.listdir('C:/Users/usuario/Dropbox/PC/Documents/INGENIERIA DE LA SALUD/4º UNI GIS/TFG/repeticion preprocesamiento/Datos EMD/Samsung')


# In[219]:


'''
Como algunas tienen la extensión .jpg o .png escrita en mayúscula y otras en minúscula, empleo el método upper() para poner
todos los nombre que almacenan las listas anteriores en mayúscula.
'''

contenido_fotos_IPhone = [i.upper() for i in contenido_fotos_IPhone]

contenido_fotos_OCT = [i.upper() for i in contenido_fotos_OCT]

contenido_fotos_Samsung = [i.upper() for i in contenido_fotos_Samsung]

print(contenido_fotos_IPhone)


# In[220]:


'''
En algunas carpetas tenemos versiones de más de ciertas imágenes algo que tampoco sería un problema y no vamos a eliminar,
simplemente lo anotaremos como una anomalía. Sin embargo, ya que tenemos la oportunidad vamos a ver en que casos ocurre esto
para cada carpeta.
'''
'''
Son nombres con 4-6 números y dos letras correspondientes al ojo. En caso de que aparezca alguna versión más de la imagen saldría
un punto y luego 2 o 3, así que vamos a buscar esos carácteres en la posición 8.
'''
def versiones(lista):
    
    cont = 0
    '''
    Almacenará la cantidad de varias versiones por carpeta.
    '''
    cont_versiones = 0
    '''
    Almacenará las posiciones de las imágenes que son versiones 2 o 3 de otras imágenes ya existentes.
    '''
    posiciones_versiones = []
    '''
    Sale cuando ha recorrido todas las imágenes de la lista.
    '''
    while (cont < len(lista)):
        '''
        Busca si hay segundas versiones.
        '''
        '''
        El métood find() busca el caracter indicado en una String en las posiciones indicadas.
        '''
        if (lista[cont].find('2', 7, 15) != -1): #Si no encuentra el carácter entre el rango de posiciones devuelve -1.
            cont_versiones += 1
            posiciones_versiones.append(lista[cont])
        '''
        Busca si hay terceras versiones.
        '''
        if (lista[cont].find('3', 7, 15) != -1):
            cont_versiones += 1
            posiciones_versiones.append(lista[cont])
            
        cont += 1
        '''
        Se devuelve tanto la cantidad de versiones como sus posiciones.
        '''
    return cont_versiones, posiciones_versiones


# In[221]:


'''
Versiones entre las imágenes del IPhone.
'''
versiones(contenido_fotos_IPhone)


# In[222]:


'''
Versiones entre las imágenes del OCT.
'''
versiones(contenido_fotos_OCT)


# In[223]:


'''
Versiones entre las imágenes del Samsung.
'''
versiones(contenido_fotos_Samsung) 


# In[224]:


'''
Vamos a comprobar si tenemos las 2 imágenes esperadas para cada paciente en cada carpeta creando los nombres que deberían 
tener dichas imágenes y buscandolas en las listas de contenido.
El nombre básico de cada imágenes conteniene el número de historia de 4-6 dígitos y dos letras correspondientes al aparato y
el ojo al que se refieren.
'''
def todas_fotos(NHC, contenido_fotos_IPhone, contenido_fotos_OCT, contenido_fotos_Samsung):
    '''
    Variable boleana que devolverá True si realmente están las 6 fotos requeridas para cada NHC.
    '''
    todas = True
    '''
    OCT utiliza la letra "T" seguida de "D" o "I" para denotar cada imagen.
    '''
    oct_d = str(int(NHC)).upper() + "TD.JPG"
    oct_i = str(int(NHC)).upper() + "TI.JPG"
    '''
    IPhone utiliza la letra "E" seguida de "D" o "I" para denotar cada imagen.
    '''
    iphone_d = str(int(NHC)).upper() + "ED.PNG"
    iphone_i = str(int(NHC)).upper() + "EI.PNG"
    '''
    Samsung utiliza la letra "G" seguida de "D" o "I" para denotar cada imagen.
    '''
    samsung_d = str(int(NHC)).upper() + "GD.PNG"
    samsung_i = str(int(NHC)).upper() + "GI.PNG"
    
    if ((oct_d not in contenido_fotos_OCT) or (oct_i not in contenido_fotos_OCT) or (iphone_d not in contenido_fotos_IPhone) or (iphone_i not in contenido_fotos_IPhone) or (samsung_d not in contenido_fotos_Samsung) or (samsung_d not in contenido_fotos_Samsung)):
        todas = False
        
    return todas


# In[225]:


'''
Caso anómalo inicialmente.
'''
todas_fotos(32094, contenido_fotos_IPhone, contenido_fotos_OCT, contenido_fotos_Samsung)


# In[226]:


'''
Utilizamos la función todas_fotos para todos los NHC de los pacientes filtrados.
'''
faltan_fotos = []

for e in lista_NHC_filtrados:
    
    if (todas_fotos(e, contenido_fotos_IPhone, contenido_fotos_OCT, contenido_fotos_Samsung) == False):
        faltan_fotos.append(e)

print(faltan_fotos)


# Aunque sale que tenemos un NHC sin las 6 fotos esto no es verdad, y es que el problema es que el nombre de una de las fotos en Samsung para el 32094 tiene el nombre anómalo: "32094GD.". Así que cambiamos el nombre y arreglado.

# In[227]:


'''
TENEMOS 60 PACIENTES DEFINITIVAMENTE FILTRADOS
'''
lista_NHC_filtrados = sorted(list(set(df['NHC'])))
print(lista_NHC_filtrados)
print(len(lista_NHC_filtrados))


# #### RESULTADOS CLASIFICACIÓN EMD

# Vamos a intentar obtener 3 listas donde cada una de ellas guarde los resultados que tiene cada aparato para la variable de interés grado de retinopatía que se encuentra en la columna 10 (posición 9).
# 
# **Etiqueta de cada aparato:**
# 
# 1 - OCT
# 
# 2 - IPhone
# 
# 3 - Samsung

# #### Creamos DataFrames independientes para cada aparato

# In[53]:


'''
Tenemos 720 filas en total así que deberían dividirse como 240 para cada aparato.
'''
is_IPhone = df['1 OCT 2 IPHONE 3 SAMSUNG'] == 2
df_IPhone = df[is_IPhone]
df_IPhone.head()
print(len(df_IPhone)) #Bien
display(df_IPhone)


# In[54]:


#Comprobación error
print(list(df_IPhone.iloc[:, 0]))


# In[55]:


print(170692 in list(df_IPhone.iloc[:, 0]))


# In[56]:


is_OCT = df['1 OCT 2 IPHONE 3 SAMSUNG'] == 1
df_OCT = df[is_OCT]
df_OCT.head()
print(len(df_OCT)) #Bien
display(df_OCT)


# In[57]:


is_Samsung = df['1 OCT 2 IPHONE 3 SAMSUNG'] == 3
df_Samsung = df[is_Samsung]
df_Samsung.head()
print(len(df_Samsung)) #Bien
display(df_Samsung)


# # TAREA 2 Comparación opinión retinólogos

# A partir de ahora emplearemos los dataframes de cada aparato independientemente vamos a comprobar que las 4 filas correspondientes a cada NHC en cada dataframe sigue la misma ordenación.

# ## Ordenar el conjunto de las 4 filas de cada NHC

# In[58]:


'''
Ejemplo primer paciente del df_OCT.
El patrón que se puede ver es que el ojo se alterna entre derecho izquierdo. Y los retinologos primero aparece el 2 en dos filas y 
luego el 1 en las otras dos.
1º FILA -> ojo derecho + retinologo 2
2º FILA -> ojo izquierdo + retinologo 2
3º FILA -> ojo derecho + retinologo 1
4º FILA -> ojo izquierdo + retinologo 1
'''
df_OCT


# In[59]:


'''
ORDENADO -> Comprobación de la distribución de los ojos en df_OCT
'''
fallo = False

contador = 0

columna_ojo = list(df_OCT['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[60]:


'''
ORDENADO -> Comprobación de la distribución de los retinologos en df_OCT. De primeras parece que no se cumple siempre que las 
dos primeras filas correspondan a retinologo 2 y las dos siguientes a retinologo 1, pero siempre y cuando sí que estén seguidas 
las dos del mismo retinologo, nos sirve para hacer las comprobaciones siguientes.
'''
fallo1 = False

contador1 = 0

columna_ret = list(df_OCT['Retinlogo 1 y 2'])

while ((contador1+1) < len(columna_ret) and fallo1 == False):
    
    if (columna_ret[contador1] != columna_ret[contador1+1]):
        fallo1 = True
        print(contador1)
    
    contador1 += 2

print(fallo1)


# In[61]:


'''
1 CAMBIO -> Comprobación de la distribución de los ojos en df_IPhone, parece que no están bien colocadas.
'''
fallo = False

contador = 0

columna_ojo = list(df_IPhone['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[62]:


print(list(df_IPhone.iloc[:, 0]))


# In[63]:


df_IPhone.iloc[160 : 180, :]


# In[64]:


cambiar = df_IPhone.iloc[172, :] 

df_IPhone.iloc[172, :] = df_IPhone.iloc[173, :]

df_IPhone.iloc[173, :] = cambiar


# In[65]:


df_IPhone.iloc[160 : 180, :] #Arreglado


# In[66]:


print(list(df_IPhone.iloc[:, 0]))


# In[67]:


'''
2 CAMBIO -> Comprobación de la distribución de los ojos en df_IPhone, parece que no están bien colocadas.
'''
fallo = False

contador = 0

columna_ojo = list(df_IPhone['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[68]:


df_IPhone.iloc[216 : 220, :]


# In[69]:


cambio_216 = df_IPhone.iloc[216, :]
cambio_217 = df_IPhone.iloc[217, :]
cambio_218 = df_IPhone.iloc[218, :]
cambio_219 = df_IPhone.iloc[219, :]

df_IPhone.iloc[216, :] = cambio_219
df_IPhone.iloc[217, :] = cambio_216
df_IPhone.iloc[218, :] = cambio_217
df_IPhone.iloc[219, :] = cambio_218


# In[70]:


df_IPhone.iloc[216 : 220, :] #Arreglado


# In[71]:


print(list(df_IPhone.iloc[:, 0]))


# In[72]:


'''
3 CAMBIO -> Comprobación de la distribución de los ojos en df_IPhone, parece que no están bien colocadas.
'''
fallo = False

contador = 0

columna_ojo = list(df_IPhone['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[73]:


df_IPhone.iloc[216 : 224, :]


# In[74]:


cambio_220 = df_IPhone.iloc[220, :]
cambio_221 = df_IPhone.iloc[221, :]
cambio_222 = df_IPhone.iloc[222, :]
cambio_223 = df_IPhone.iloc[223, :]

df_IPhone.iloc[220, :] = cambio_223
df_IPhone.iloc[221, :] = cambio_220
df_IPhone.iloc[222, :] = cambio_221
df_IPhone.iloc[223, :] = cambio_222


# In[75]:


df_IPhone.iloc[216 : 224, :] #Arreglado


# In[76]:


print(list(df_IPhone.iloc[:, 0]))


# In[77]:


'''
ORDENADO -> Comprobación de la distribución de los ojos en df_IPhone, parece que no están bien colocadas.
'''
fallo = False

contador = 0

columna_ojo = list(df_IPhone['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[78]:


'''
ORDENADO -> Comprobación de la distribución de los retinologos en df_IPhone.
'''
fallo1 = False

contador1 = 0

columna_ret = list(df_IPhone['Retinlogo 1 y 2'])

while ((contador1+1) < len(columna_ret) and fallo1 == False):
    
    if (columna_ret[contador1] != columna_ret[contador1+1]):
        fallo1 = True
        print(contador1)
    
    contador1 += 2

print(fallo1)


# In[79]:


print(list(df_IPhone.iloc[:, 0]))


# In[80]:


'''
1 CAMBIO -> Comprobación de la distribución de los ojos en df_Samsung.
'''
fallo = False

contador = 0

columna_ojo = list(df_Samsung['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[81]:


df_Samsung.iloc[208:212, :]


# In[82]:


cambio_208 = df_Samsung.iloc[208, :]
cambio_209 = df_Samsung.iloc[209, :]
cambio_210 = df_Samsung.iloc[210, :]
cambio_211 = df_Samsung.iloc[211, :]

df_Samsung.iloc[208, :] = cambio_211
df_Samsung.iloc[209, :] = cambio_208
df_Samsung.iloc[211, :] = cambio_209


# In[83]:


df_Samsung.iloc[208:212, :] #Arreglado


# In[84]:


'''
2 CAMBIO -> Comprobación de la distribución de los ojos en df_Samsung.
'''
fallo = False

contador = 0

columna_ojo = list(df_Samsung['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[85]:


df_Samsung.iloc[212:216, :]


# In[86]:


cambio_212 = df_Samsung.iloc[212, :]
cambio_213 = df_Samsung.iloc[213, :]
cambio_214 = df_Samsung.iloc[214, :]
cambio_215 = df_Samsung.iloc[215, :]

df_Samsung.iloc[212, :] = cambio_215
df_Samsung.iloc[213, :] = cambio_212
df_Samsung.iloc[214, :] = cambio_213
df_Samsung.iloc[215, :] = cambio_214


# In[87]:


df_Samsung.iloc[212:216, :] #Arreglado


# In[88]:


'''
3 CAMBIO -> Comprobación de la distribución de los ojos en df_Samsung.
'''
fallo = False

contador = 0

columna_ojo = list(df_Samsung['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[89]:


df_Samsung.iloc[228:232, :]


# In[90]:


cambio_228 = df_Samsung.iloc[228, :]
cambio_229 = df_Samsung.iloc[229, :]

df_Samsung.iloc[228, :] = cambio_229
df_Samsung.iloc[229, :] = cambio_228


# In[91]:


df_Samsung.iloc[228:232, :] #Arreglado


# In[92]:


'''
ORDENADO -> Comprobación de la distribución de los ojos en df_Samsung.
'''
fallo = False

contador = 0

columna_ojo = list(df_Samsung['lateralidad 1 Dch 2 izq'])

while ((contador+2) < len(columna_ojo) and fallo == False):
    
    if (columna_ojo[contador] != columna_ojo[contador+2]):
        fallo = True
        print(contador)
    
    contador += 1

print(fallo)


# In[93]:


'''
ORDENADO -> Comprobación de la distribución de los retinologos en df_IPhone.
'''
fallo1 = False

contador1 = 0

columna_ret = list(df_Samsung['Retinlogo 1 y 2'])

while ((contador1+1) < len(columna_ret) and fallo1 == False):
    
    if (columna_ret[contador1] != columna_ret[contador1+1]):
        fallo1 = True
        print(contador1)
    
    contador1 += 2

print(fallo1)


# #### Comprobación ordenamiento

# In[94]:


print(list(df_IPhone.iloc[:, 0]))


# In[95]:


'''
Comprobar que realmente las filas van alternándose en ojo derecho las pares y ojo izquierdo las impares.
'''
def orden_lateralidad(df):
    
    columna_lateralidad = list(df.iloc[:, 2]) #Columna lateralidad -> 2
    
    ordenado = True
    
    contador = 0
    
    while((contador+1) < len(columna_lateralidad)):
        
        if (columna_lateralidad[contador] != 1 or columna_lateralidad[contador+1] != 2):
            ordenado = False
            
        contador+= 2
        
    return ordenado


# In[96]:


'''
Probar que orden_lateralidad() se cumple para los 3 dataframes.
'''
print(orden_lateralidad(df_OCT))
print(orden_lateralidad(df_IPhone))
print(orden_lateralidad(df_Samsung))


# In[97]:


'''
Comprobar que realmente las filas se distribuyen las dos primeras para el segundo retinologo y las dos siguientes para el primero.
'''
def orden_retin(df):
    
    columna_retin = list(df.iloc[:, 3]) #Columna retinologo -> 3
    
    ordenado = True
    
    contador = 0
    
    posiciones_fallos = []
    
    while((contador+4) < len(columna_retin)):
        
        if (columna_retin[contador] != 2 or columna_retin[contador+1] != 2 or columna_retin[contador+2] != 1 or columna_retin[contador+3] != 1):
            ordenado = False
            posiciones_fallos.append(contador)
            
        contador+= 4
        
    return ordenado, posiciones_fallos


# In[98]:


'''
Probar que orden_retin() se cumplen en los 3 dataframes:
'''
print(orden_retin(df_OCT))
print(orden_retin(df_IPhone))
print(orden_retin(df_Samsung))


# In[99]:


print(list(df_IPhone.iloc[:, 0]))


# In[100]:


'''
Arreglar filas mal colocadas en df_IPhone.
1 CAMBIO
'''
df_IPhone.iloc[160:164, :]


# In[101]:


cambio_160 = df_IPhone.iloc[160, :]
cambio_161 = df_IPhone.iloc[161, :]
cambio_162 = df_IPhone.iloc[162, :]
cambio_163 = df_IPhone.iloc[163, :]

df_IPhone.iloc[160, :] = cambio_162
df_IPhone.iloc[161, :] = cambio_163
df_IPhone.iloc[162, :] = cambio_160
df_IPhone.iloc[163, :] = cambio_161


# In[102]:


df_IPhone.iloc[160:164, :] #Arreglado


# In[103]:


print(orden_retin(df_IPhone))


# In[104]:


'''
2 CAMBIO
'''
df_IPhone.iloc[204:208, :]


# In[105]:


cambio_204 = df_IPhone.iloc[204, :]
cambio_205 = df_IPhone.iloc[205, :]
cambio_206 = df_IPhone.iloc[206, :]
cambio_207 = df_IPhone.iloc[207, :]

df_IPhone.iloc[204, :] = cambio_206
df_IPhone.iloc[205, :] = cambio_207
df_IPhone.iloc[206, :] = cambio_204
df_IPhone.iloc[207, :] = cambio_205


# In[106]:


df_IPhone.iloc[204:208, :]#Arreglado


# In[107]:


print(orden_retin(df_IPhone))


# In[108]:


print(list(df_IPhone.iloc[:, 0]))


# In[109]:


'''
3 CAMBIO
'''
df_IPhone.iloc[228:232, :]


# In[110]:


cambio_228 = df_IPhone.iloc[228, :]
cambio_229 = df_IPhone.iloc[229, :]
cambio_230 = df_IPhone.iloc[230, :]
cambio_231 = df_IPhone.iloc[231, :]

df_IPhone.iloc[228, :] = cambio_230
df_IPhone.iloc[229, :] = cambio_231
df_IPhone.iloc[230, :] = cambio_228
df_IPhone.iloc[231, :] = cambio_229


# In[111]:


df_IPhone.iloc[228:232, :]#Arreglado


# In[112]:


print(orden_retin(df_IPhone))


# In[113]:


print(list(df_IPhone.iloc[:, 0]))


# In[114]:


'''
4 CAMBIO
'''
df_IPhone.iloc[232:236, :]


# In[115]:


cambio_232 = df_IPhone.iloc[232, :]
cambio_233 = df_IPhone.iloc[233, :]
cambio_234 = df_IPhone.iloc[234, :]
cambio_235 = df_IPhone.iloc[235, :]

df_IPhone.iloc[232, :] = cambio_234
df_IPhone.iloc[233, :] = cambio_235
df_IPhone.iloc[234, :] = cambio_232
df_IPhone.iloc[235, :] = cambio_233


# In[116]:


df_IPhone.iloc[232:236, :]#Arreglado


# In[117]:


print(list(df_IPhone.iloc[:, 0]))


# In[118]:


'''
Probar que orden_retin() se cumplen en los 3 dataframes:
'''
print(orden_retin(df_OCT))
print(orden_retin(df_IPhone))
print(orden_retin(df_Samsung))


# In[119]:


print(list(df_OCT['NHC']) == list(df_IPhone['NHC']) == list(df_Samsung['NHC']))
print(list(df_OCT['lateralidad 1 Dch 2 izq']) == list(df_IPhone['lateralidad 1 Dch 2 izq']) == list(df_Samsung['lateralidad 1 Dch 2 izq']))


# In[120]:


'''
Detectado fallo en el orden de df_OCT.
'''
df_OCT.iloc[236:240, :]


# In[121]:


cambio_236 = df_OCT.iloc[236, :]
cambio_237 = df_OCT.iloc[237, :]
cambio_238 = df_OCT.iloc[238, :]
cambio_239 = df_OCT.iloc[239, :]

df_OCT.iloc[236, :] = cambio_238
df_OCT.iloc[237, :] = cambio_239
df_OCT.iloc[238, :] = cambio_236
df_OCT.iloc[239, :] = cambio_237


# In[122]:


df_OCT.iloc[236:240, :]


# In[123]:


print(orden_retin(df_OCT))
print(orden_retin(df_IPhone))
print(orden_retin(df_Samsung))


# In[124]:


print(list(df_OCT['NHC']) == list(df_IPhone['NHC']) == list(df_Samsung['NHC']))
print(list(df_OCT['lateralidad 1 Dch 2 izq']) == list(df_IPhone['lateralidad 1 Dch 2 izq']) == list(df_Samsung['lateralidad 1 Dch 2 izq']))
print(list(df_OCT['Retinlogo 1 y 2']) == list(df_IPhone['Retinlogo 1 y 2']) == list(df_Samsung['Retinlogo 1 y 2']))


# ### Encontrar discrepancias retinologos

# In[125]:


'''
Tenemos la misma interpretación para cada ojo de cada paciente realizada por el retinologo 1 y 2 de con un salto de dos filas.
De tal forma que las valoraciones del retinologo 1 y 2 para el mismo ojo están en las filas pares para el ojo derecho e impares
para el izquierdo. Comprobaremos cuántas valoraciones diferentes dan y en que posiciones se encuentran.
'''
def retinologos_diferente(df, columna_interes):
    
    columna_valores = list(df.iloc[:, columna_interes]) #Grado retinopatía o clasificación EMD
    
    columna_lateralidad = list(df.iloc[:, 2])
    
    columna_NHC = list(df.iloc[:, 0])
    
    cont = 0
    
    cont_diferentes = 0
    
    lista_posiciones_diferencias = []
    
    while ((cont+2)<len(columna_valores)):
        
        if (columna_valores[cont] != columna_valores[cont+2]): #Comprobar diagnóstico en ojo derecho
            lista_posiciones_diferencias.append([columna_NHC[cont], columna_lateralidad[cont]])
            cont_diferentes += 1
        
        if (columna_valores[cont+1] != columna_valores[cont+3]): #Comprobar diagnóstico en ojo izquierdo
            lista_posiciones_diferencias.append([columna_NHC[cont+1], columna_lateralidad[cont+1]])
            cont_diferentes += 1
        
        cont += 4
        
    return cont_diferentes, lista_posiciones_diferencias


# In[126]:


'''
Probar retinologos_diferentes -> Grado retinopatía -> 5
'''
dif_ret = retinologos_diferente(df_OCT, 5)

print(dif_ret)

NHC_ret_dif = []

for e in dif_ret[1]:
    NHC_ret_dif.append(e[0])
    
print(NHC_ret_dif)


# In[127]:


'''
Probar retinologos_diferentes -> Clasificación EMD -> 6
'''
dif_emd = retinologos_diferente(df_OCT, 6)

print(dif_emd)

NHC_emd_dif = []

for e in dif_emd[1]:
    NHC_emd_dif.append(e[0])
    
print(NHC_emd_dif)


# # TAREA 3 Homogeneizar en OCT el diagnóstico retinólogos

# Considerando que en la tarea anterior hemos visto cómo hay casos en los que los dos retinologos no dan el mismo diagnóstico para grado de retinopatía o para clasificación EMD, vamos a homogeneizar su opinión en estos casos de discrepancia para tener un gols estandar óptimo. Para ello consideraremos una visión pesimista, y en caso de discrepancia pondremos ambas filas con el diagnóstico más grave. Tanto para el grado de retinopatía como para la clasificación EMD es más grave cuando mayor sea el número establecido.

# In[130]:


'''
Reutilizando parte de la tarea anterior porque recordamos como estaban organizadas las cuatro filas de cada paciente en el df_OCT,
estableceremos los siguientes condicionales para poder homogeneizar correctamente considerando la valoración más pesimista.
1º FILA -> ojo derecho (1) + retinologo 2
2º FILA -> ojo izquierdo (2) + retinologo 2
3º FILA -> ojo derecho (1) + retinologo 1
4º FILA -> ojo izquierdo (2) + retinologo 1
'''
def retinologos_homogeneizar(df, columna_interes):
    columna_valores = list(df.iloc[:, columna_interes]) #Grado retinopatía o clasificación EMD
    
    cont = 0
    
    while ((cont+2)<len(columna_valores)):
        
        if (columna_valores[cont] != columna_valores[cont+2]): #Comprobar diagnóstico en ojo derecho
            if(columna_valores[cont] > columna_valores[cont+2]):
                df.iloc[cont+2, columna_interes] = df.iloc[cont, columna_interes]
            else:
                df.iloc[cont, columna_interes] = df.iloc[cont+2, columna_interes]
        
        if (columna_valores[cont+1] != columna_valores[cont+3]): #Comprobar diagnóstico en ojo izquierdo
            if(columna_valores[cont+1] > columna_valores[cont+3]):
                df.iloc[cont+3, columna_interes] = df.iloc[cont+1, columna_interes]
            else:
                df.iloc[cont+1, columna_interes] = df.iloc[cont+3, columna_interes]            
        
        cont += 4
        
    return df


# In[131]:


'''
Probamos retinologos_homogeneizar para df_OCT considerando el grado de retinopatía donde anteriormente nos habíamos encontrado 
21 discrepancias. Vemos que la función si que corrige estos casos, porque si volvemos a analizar los diagnósticos diferentes con
retinologos_diferente() ya no aparece ninguno.
'''
df_OCT = retinologos_homogeneizar(df_OCT, 5) #Grado de retinopatía en la columna 5

retinologos_diferente(df_OCT, 5)


# In[132]:


'''
Probamos retinologos_homogeneizar para df_OCT considerando la clasificación EMD donde anteriormente nos habíamos encontrado 
19 discrepancias. Vemos que la función si que corrige estos casos, porque si volvemos a analizar los diagnósticos diferentes con
retinologos_diferente() ya no aparece ninguno.
'''
df_OCT = retinologos_homogeneizar(df_OCT, 6) #Clasificación EMD en la columna 6

retinologos_diferente(df_OCT, 6)


# ### Homgeneizar IPhone y Samsung

# In[133]:


'''
Probamos retinologos_homogeneizar para df_IPhone
'''
df_IPhone = retinologos_homogeneizar(df_IPhone, 5) #Grado de retinopatía en la columna 5

retinologos_diferente(df_IPhone, 5)


# In[134]:


df_IPhone = retinologos_homogeneizar(df_IPhone, 6) #Clasificación EMD en la columna 6

retinologos_diferente(df_IPhone, 6)


# In[135]:


'''
Probamos retinologos_homogeneizar para df_IPhone
'''
df_Samsung = retinologos_homogeneizar(df_Samsung, 5) #Grado de retinopatía en la columna 5

retinologos_diferente(df_Samsung, 5)


# In[136]:


df_Samsung = retinologos_homogeneizar(df_Samsung, 6) #Clasificación EMD en la columna 6

retinologos_diferente(df_Samsung, 6)


# ### Buscar las diferentes calidades para la misma imagen

# In[137]:


'''
Hemos encontrado como posible anomalia, que en algunos casos para la misma imagen aparecen dos calidades diferentes. Osea en la
imagen de un ojo para un retinologo aparece una imagen y en la misma imagen para otro retinologo otra calidad diferente.
Recuerdo que la misma imagen para cada retinologo se encuentra una a un salto de la otra, las imágenes para lateralidad derecha
en las filas pares y para la izquierda en las impares.
'''
def calidad_distinta(df):
    
    columna_calidad = list(df.iloc[:, 4]) # Columna calidad general -> 4
    
    columna_NHC = list(df.iloc[:, 0]) # Columna calidad NHC -> 0
    
    columna_lateralidad = list(df.iloc[:, 2]) # Columna lateralidad -> 2
    
    contador = 0
    
    lista_diferencias = []
    
    contador_diferencias = 0
    
    while((contador+4) < len(columna_calidad)): #Miramos para cada conjunto de 4 filas de cada NHC si coinciden las del mismo ojo
        
        if(columna_calidad[contador] != columna_calidad[contador+2]): #Ojo derecho
            contador_diferencias += 1
            lista_diferencias.append([columna_NHC[contador], columna_lateralidad[contador], columna_calidad[contador], columna_calidad[contador+2]])
            
        if(columna_calidad[contador+1] != columna_calidad[contador+3]): #Ojo izquierdo
            contador_diferencias += 1
            lista_diferencias.append([columna_NHC[contador+1], columna_lateralidad[contador+1], columna_calidad[contador+1], columna_calidad[contador+3]])
            
        contador += 4
    
    return contador_diferencias, lista_diferencias


# In[138]:


'''
Buscar en qué casos aparecen estas diferencias en la calidad para cada dataframe.
'''
dif_calidad_OCT = calidad_distinta(df_OCT)

print(dif_calidad_OCT)


# In[139]:


df_IPhone[df_IPhone['NHC']==170692]


# In[140]:


dif_calidad_IPhone = calidad_distinta(df_IPhone)

print(sorted(dif_calidad_IPhone[1]))


# In[141]:


dif_calidad_Samsung = calidad_distinta(df_Samsung)

print(dif_calidad_Samsung)


# In[142]:


'''
Reutilizando la idea de retinologos_homogeneizar() para establecer la misma calidad para las dos filas relativas a la misma
imagen en caso de que haya discrepancia, definimos calidad_homogeneizar() que en caso de discordia coge la calidad más alta.
'''
def calidad_homogeneizar(df, columna_interes):
    columna_valores = list(df.iloc[:, columna_interes]) #Grado retinopatía o clasificación EMD
    
    cont = 0
    
    while ((cont+2)<len(columna_valores)):
        
        if (columna_valores[cont] != columna_valores[cont+2]): #Comprobar diagnóstico en ojo derecho
            if(columna_valores[cont] > columna_valores[cont+2]):
                df.iloc[cont+2, columna_interes] = df.iloc[cont, columna_interes]
            else:
                df.iloc[cont, columna_interes] = df.iloc[cont+2, columna_interes]
        
        if (columna_valores[cont+1] != columna_valores[cont+3]): #Comprobar diagnóstico en ojo izquierdo
            if(columna_valores[cont+1] > columna_valores[cont+3]):
                df.iloc[cont+3, columna_interes] = df.iloc[cont+1, columna_interes]
            else:
                df.iloc[cont+1, columna_interes] = df.iloc[cont+3, columna_interes]            
        
        cont += 4
        
    return df


# In[143]:


'''
calidad_homogeneizar() para df_OCT.
'''
df_OCT = calidad_homogeneizar(df_OCT, 4) # Calidad general se encuentra en la columna 4

print(calidad_distinta(df_OCT))

display(df_OCT)


# In[144]:


'''
calidad_homogeneizar() para df_IPhone.
'''
df_IPhone = calidad_homogeneizar(df_IPhone, 4) # Calidad general se encuentra en la columna 4

print(calidad_distinta(df_IPhone))


# In[145]:


df_IPhone[df_IPhone['NHC']==170692]


# In[146]:


'''
calidad_homogeneizar() para df_Samsung.
'''
df_Samsung = calidad_homogeneizar(df_Samsung, 4) # Calidad general se encuentra en la columna 4

print(calidad_distinta(df_Samsung))


# # TAREA 4 Filtrar solo imágenes con calidad 4-5

# In[176]:


'''
Vamos a obtener el porcentaje de imágenes con calidad superior e inferior para el dataframe de cada aparato y luego hacemos la media
de los tres. Calidad superior significa 4-5 y sino inferior.
'''
def calidad_superior(df):
    
    columna_calidad = list(df.iloc[:, 4]) #Columna calidad general -> 4
    
    calidades_imagenes = []
    
    contador = 0
    
    #Solo cogemos una observacion de cada imagen.
    while ((contador+4) < len(columna_calidad)):
        
        if(columna_calidad[contador]>3):
            calidades_imagenes.append(columna_calidad[contador])
            
        if(columna_calidad[contador+1]>3):
            calidades_imagenes.append(columna_calidad[contador+1])
        
        contador += 4
    
    return len(calidades_imagenes)/(len(columna_calidad)/2)*100


# In[177]:


'''
Obtener calidades para el conjunto de imágenes de cada aparato
'''
sup_OCT = calidad_superior(df_OCT)

sup_IPhone = calidad_superior(df_IPhone)

sup_Samsung = calidad_superior(df_Samsung)

print(f"El porcentaje de imágenes con calidad superior para OCT es del: {sup_OCT} %")

print(f"El porcentaje de imágenes con calidad superior para IPhone es del: {sup_IPhone} %")

print(f"El porcentaje de imágenes con calidad superior para Samsung es del: {sup_Samsung} %")

print(f"El porcentaje general de imágenes con calidad superior (4-5) es del: {(sup_OCT + sup_IPhone + sup_Samsung) / 3} %")


# Nos quedamos en los 3 dataframes solo con las filas de los NHC que tengan calidad superior a 3 en las imágenes relativas a sus dos ojos, porque si eliminamos un solo ojo porque el otro cumple la calidad exigida, perdemos la completitud de los datos. Así que primero buscamos los NHC de los que tienen alguna imagen con calidad inferior a 4.

# In[178]:


display(df_OCT)


# In[179]:


display(df_IPhone)


# In[180]:


display(df_Samsung)


# In[181]:


# Preguntar si cogemos todas las filas que tengan calidad superior a 3 aunque algunos pacientes queden con un solo ojo.

'''
Filtrar imágenes calidad superior a 3 con máscara booleana.

CALIDAD GRAL IMAGEN > 3 para df_OCT
'''
df_OCT = df_OCT[df_OCT['CALIDAD GRAL IMAGEN']>3]

display(df_OCT) #230 filas


# In[182]:


'''
Filtrar imágenes calidad superior a 3 con máscara booleana.

CALIDAD GRAL IMAGEN > 3 para df_IPhone
'''
df_IPhone = df_IPhone[df_IPhone['CALIDAD GRAL IMAGEN']>3]

display(df_IPhone) #178 filas


# In[183]:


'''
Filtrar imágenes calidad superior a 3 con máscara booleana.

CALIDAD GRAL IMAGEN > 3 para df_Samsung
'''
df_Samsung = df_Samsung[df_Samsung['CALIDAD GRAL IMAGEN']>3]

display(df_Samsung) #144 filas


# **Importamos los dataframes definitivos a archivos csv.**

# In[264]:


df_OCT.to_excel('df_OCT.xlsx')


# In[265]:


df_IPhone.to_excel('df_iPhone.xlsx')


# In[266]:


df_Samsung.to_excel('df_Samsung.xlsx')
