import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
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

# Correlaciones entre variables

sns.pairplot(data_day)

# Matriz de correlación

data_day.corr()

px.line(data_day["Volumen Transportado  [bls]"]/data_day["Consumo Bombas [MBTU]"])

# Outlayer \#1: Producción vs consumo = 2222

data_day.iloc[405]

# Outlayer \#2: Producción vs consumo = 14.69

data_day.iloc[55]

# # Datos por minutos

# Carga de datos

data_min = pd.read_csv(data_path+"min_data.csv")
data_min

data_min.info()

data_min["date"] = data_min['Timestamp'].str.split(' ',expand=True)[0]

data_day_std = data_min.groupby(['date']).std()
data_day_std

data_day_std.describe()


