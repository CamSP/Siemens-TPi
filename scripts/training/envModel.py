import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks  import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
tf.random.set_seed(
    1234
)
data_path = "../../data/preprocessed/"

# # Carga de datos

data = pd.read_csv(data_path+"data_day.csv")
data

# # Estandarización de los datos

scaler = StandardScaler()
X = scaler.fit_transform(data[data.columns[:3]])
y = scaler.fit_transform(data[data.columns[3:5]])

# # Train/test split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, shuffle = True)

# +
valve1_train, valve2_train, valve34_train =  np.transpose(X_train)
valve1_test, valve2_test, valve34_test =  np.transpose(X_test)

production_train, energy_train = Y_train[:, 0], Y_train[:, 1]
production_test, energy_test = Y_test[:, 0], Y_test[:, 1]


# -

# # Modelo

# ## Contrucción del modelo

def build_model():
    # Tamaño de las entradas
    shape = (1, )
    
    # Entradas
    valve1 = Input(shape=shape, name="1401B")
    valve2 = Input(shape=shape, name="1400A")
    valve34 = Input(shape=shape, name="1400B&1501A")
    list_inputs = [valve1, valve2, valve34]
    x = concatenate(list_inputs)
    
    # 4 capas densas con activación ReLu
    x = Dense(32, kernel_initializer='normal', activation="relu")(x)
    x = Dense(64, kernel_initializer='normal', activation="relu")(x)
    
    energy = Dense(1, kernel_initializer='normal', activation="linear", name="energy")(x)
    production = Dense(1, kernel_initializer='normal', activation="linear", name="production")(x)
    list_outputs = [energy, production]
    
    model = tf.keras.Model(inputs = list_inputs, outputs = list_outputs)
    return model


model = build_model()

# ## Compilación del modelo

model.compile(optimizer=Adam(1e-4), 
             loss = ["mse", "mse"],
             metrics=['mse', 'mae'])

tf.keras.utils.plot_model(model, show_shapes = True)

model.summary()

# ## Asignación de datos de entrenamiento

# +
inputs_train = {'1401B': valve1_train, '1400A': valve2_train, '1400B&1501A': valve34_train}

outputs_train = {'energy': energy_train, 'production': production_train}

# +
inputs_test = {'1401B': valve1_test, '1400A': valve2_test, '1400B&1501A': valve34_test}

outputs_test = {'energy': energy_test, 'production': production_test}
# -

# ## Callbacks

# * Early Stopping

earlyStopping = EarlyStopping(
    monitor="val_energy_mae",
    patience=100,
    restore_best_weights=True
)

# ## Entrenamiento

history = model.fit(inputs_train, outputs_train, 
         epochs = 2000,
         batch_size = 128,
         #callbacks = [earlyStopping],
         validation_data=(inputs_test, outputs_test),
         verbose = 1)

# ## Graficas de entrenamiento

# +
# Plot the training accuracy

mse_keys = [k for k in history.history.keys() if k in ('energy_mse', 'production_mse', 'val_energy_mse', 'val_production_mse')] 
loss_keys = [k for k in history.history.keys() if not k in mse_keys]

for k, v in history.history.items():
    if k in mse_keys:
        plt.figure(1)
        plt.plot(v)
    else:
        plt.figure(2)
        plt.plot(v)

plt.figure(1)
plt.title('mse vs. epochs')
plt.ylabel('mse')
plt.xlabel('Epoch')
plt.legend(mse_keys, loc='upper right')

plt.figure(2)
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loss_keys, loc='upper right')

plt.show()
# -

# # Evaluación del modelo

model.evaluate([valve1_test, valve2_test, valve34_test], [energy_test, production_test])

y_test_inverse = scaler.inverse_transform(np.transpose([production_test, energy_test]))
y_pred_energy, y_pred_production = model.predict([valve1_test, valve2_test, valve34_test])
y_pred = scaler.inverse_transform(np.transpose([y_pred_production, y_pred_energy]))

# +
x_graph = np.arange(len(valve1_test))

fig = go.Figure()
# Energy
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = y_test_inverse[:, 0],  # Datos del eje Y.
                         name = 'Real', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = y_pred[:, :,0][0],  # Datos del eje Y.
                         name = 'Predicción', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))
# -

fig = go.Figure()
# Production
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = y_test_inverse[:, 1],  # Datos del eje Y.
                         name = 'Real', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = y_pred[:, :, 1][0],  # Datos del eje Y.
                         name = 'Predicción', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))

# +
fig = make_subplots(rows=2, cols=1, subplot_titles=("Producción", "Energía"))

fig.add_trace(
    go.Scatter(x=y_test_inverse[:, 0], y=y_pred[:, :, 0][0], name="Production", mode="markers"),
    row=1, col=1
)
fig.add_shape(
    type="line",
    x0=2, y0=2, x1=y_test_inverse[:, 0].max(), y1=y_test_inverse[:, 0].max(),
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    ),
     row=1, col=1
)
fig.add_trace(
    go.Scatter(x=y_test_inverse[:, 1], y=y_pred[:, :, 1][0], name="Energy", mode="markers"),
    row=2, col=1
)
fig.add_shape(
    type="line",
    x0=2, y0=2, x1=y_test_inverse[:, 1].max(), y1=y_test_inverse[:, 1].max(),
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    ),
     row=2, col=1
)


fig.update_layout(height=1000, width=900, title_text="Real vs Pred")
fig.show()
# -


