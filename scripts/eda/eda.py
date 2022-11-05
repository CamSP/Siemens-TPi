import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
data_path = "../../data/merged/"

# # Datos por día

# ## Carga de datos

data_day = pd.read_csv(data_path + "daily_data.csv")
data_day

# ## Análisis

# ### Estadisticas basicas

# * Descripción estadistica del dataset

data_day.describe()

# * Promedio de cada valvula

data_day[data_day.columns[:-2]].mean().plot().grid()

# * Valores promedio por dia de cada variable del dataset

# +
fig = make_subplots(rows=3, cols=2,
                    subplot_titles=("POZ_PIT_1501A", 
                                    "POZ_PIT_1401B", 
                                    "POZ_PIT_1400A", 
                                    "POZ_PIT_1400B",
                                    "Volumen Transportado  [bls]",
                                    "Consumo Bombas [MBTU]"))

fig.add_trace(
    go.Line(x=np.arange(519), y=data_day.POZ_PIT_1501A, name="POZ_PIT_1501A"),
    row=1, col=1
)
fig.add_trace(
    go.Line(x=np.arange(519), y=data_day.POZ_PIT_1401B, name="POZ_PIT_1401B"),
    row=1, col=2
)

fig.add_trace(
    go.Line(x=np.arange(519), y=data_day.POZ_PIT_1400A, name="POZ_PIT_1400A"),
    row=2, col=1
)
fig.add_trace(
    go.Line(x=np.arange(519), y=data_day.POZ_PIT_1400B, name="POZ_PIT_1400B"),
    row=2, col=2
)

fig.add_trace(
    go.Line(x=np.arange(519), y=data_day["Volumen Transportado  [bls]"], name="Volumen Transportado  [bls]"),
    row=3, col=1
)
fig.add_trace(
    go.Line(x=np.arange(519), y=data_day["Consumo Bombas [MBTU]"], name="Consumo Bombas [MBTU]"),
    row=3, col=2
)

fig.update_layout(height=1000, width=900, title_text="Datos por día")
fig.show()
# -

# ### Correlaciones entre variables

# * Matriz de correlación

data_day.corr()

f = plt.figure(figsize=(15, 13))
plt.matshow(data_day.corr(), fignum=f.number)
plt.xticks(range(data_day.select_dtypes(['number']).shape[1]), data_day.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(data_day.select_dtypes(['number']).shape[1]), data_day.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Matriz de correlación', fontsize=16);

# * Pairplot

sns.pairplot(data_day)

# * Dado que las graficas de POZ_PIT_1400B y POZ_PIT_1501A tienen la misma forma, se mide la relación para ver la diferencia en magnitudes

px.line(data_day["POZ_PIT_1400B"]/data_day["POZ_PIT_1501A"], title="POZ_PIT_1400B/POZ_PIT_1501A")

# * De la grafica se puede ver que en la mayoria de los casos, ambas variables son iguales, por lo que se puede decir que es información redundante.

# ## Outlayers

# * En el pairplot se observa un valor atipico en la correlación entre el volumen transportado y el consumo de las bombas. Viendo la relación entre ambas variables se encuentra el siguiente outlayer:

px.line(data_day["Volumen Transportado  [bls]"]/data_day["Consumo Bombas [MBTU]"], title="Volumen Transportado  [bls]/Consumo Bombas [MBTU]")

# ### Outlayer \#1: Producción vs consumo = 2222

# Este dato presenta un consumo practicamente nulo en comparación con los demás datos:

data_day.iloc[405]

# Se observa que los valores de las presiones de las bombas son cercanos a los valores promedio, por lo que se busca la fila con valores más cercanos para comparar el valor de consumo vs producción:

query_data_point = [1768.5, 127.4, 127.18, 1765.57]
euclidean_dist = np.square(data_day[data_day.columns[1:5]] - query_data_point).sum(axis=1)
ind = np.argmax(-euclidean_dist[2])
closest = data_day.loc[ind]
closest

# De estos datos se observa que aunque los valores de las bombas son cercanos, las diferencias de producción y consumo son gigantes, por lo que es un dato totalmente atípico.

# ### Outlayer \#2: Producción vs consumo = 14.69

# De este dato se observa que la producción es muchisimo menor a la producción normal. De forma similar al dato anterior, los valores de las valvulas son normales pero el resultado es completamente atípico.

data_day.iloc[55]

# ### Outlayer \#3: POZ_PIT_1400B vs POZ_PIT_1501A 

# De la grafica se puede observar que los valores 123 y 160 presentan valores atipicos alejados de 1. 

px.line(data_day["POZ_PIT_1400B"]/data_day["POZ_PIT_1501A"], title="POZ_PIT_1400B/POZ_PIT_1501A")

# Del dato 123 se evidencia que la valvula POZ_PIT_1401B practicamente no tuvo presión, por lo que ese día no estuvo en funcionamiento

data_day.iloc[123]

data_day.iloc[160]

data_day.columns[1:7]

fig = go.Figure()
for column in data_day.columns[1:5]:
    fig.add_trace(go.Box(y=data_day[column]))
fig.show()

# ## Preprocesamiento

# * Se eliminan los outlayers

data_day_pp = data_day.copy()
data_day_pp = data_day_pp.drop(np.arange(120, 160))
data_day_pp = data_day_pp.drop(np.arange(160, 190))
data_day_pp = data_day_pp.drop(np.arange(468, 519))
data_day_pp = data_day_pp.drop([17, 57, 55, 405])

# * Dado que POZ_PIT_1400B y POZ_PIT_1501A son practicamente iguales, para evitar redundancias y reducir la dimensionalidad, se juntan en una sola columna con el promedio aritmetico para cada valor.

data_day_pp["1400B and 1501A"] = (data_day_pp["POZ_PIT_1400B"] + data_day_pp["POZ_PIT_1501A"])/2
data_day_pp

output = data_day_pp[["POZ_PIT_1401B", 
                      "POZ_PIT_1400A", 
                      "1400B and 1501A", 
                      "Volumen Transportado  [bls]",
                      "Consumo Bombas [MBTU]"
                     ]]

output.to_csv('../../data/preprocessed/data_day.csv', index=False)  

# # Datos por minuto

# ## Carga de datos

data_min = pd.read_csv(data_path+"min_data.csv")
data_min

data_min.info()

data_min["date"] = data_min['Timestamp'].str.split(' ',expand=True)[0]

data_day_std = data_min.groupby(['date']).std()
data_day_std

data_day_std.describe()


