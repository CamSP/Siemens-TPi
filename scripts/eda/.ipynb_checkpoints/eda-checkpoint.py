import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_path = "../../data/merged/"

# # Datos por día

# Carga de datos

data_day = pd.read_csv(data_path + "daily_data.csv")
data_day

# Información basica del dataset

data_day.info()

# Estadisticas basicas del dataset

data_day.describe()

# Promedio de cada valvula

data_day[data_day.columns[:-2]].mean().plot().grid()

# Valores promedio por dia de cada variable del dataset

time = range(len(data_day))
fig, axs = plt.subplots(3, 2, figsize=(15, 15))
axs[1][1].plot(time, data_day.POZ_PIT_1501A)
axs[1][1].set_title("POZ_PIT_1501A")
axs[0][1].plot(time, data_day.POZ_PIT_1401B)
axs[0][1].set_title("POZ_PIT_1401B")
axs[0][0].plot(time, data_day.POZ_PIT_1400A)
axs[0][0].set_title("POZ_PIT_1400A")
axs[1][0].plot(time, data_day.POZ_PIT_1400B)
axs[1][0].set_title("POZ_PIT_1400B")
axs[2][0].plot(time, data_day["Volumen Transportado  [bls]"])
axs[2][0].set_title("Volumen Transportado [bls]")
axs[2][1].plot(time, data_day["Consumo Bombas [MBTU]"])
axs[2][1].set_title("Consumo Bombas [MBTU]")

# Correlaciones entre variables

sns.pairplot(data_day)

# Matriz de correlación

data_day.corr()

# # Datos por minutos

# Carga de datos

data_min = pd.read_csv(data_path+"min_data.csv")
data_min

data_min.info()

data_min["date"] = data_min['Timestamp'].str.split(' ',expand=True)[0]

data_day_std = data_min.groupby(['date']).std()
data_day_std

data_day_std.describe()
