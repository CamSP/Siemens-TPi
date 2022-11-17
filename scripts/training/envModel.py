import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, concatenate, Normalization
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks  import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score
tf.random.set_seed(
    824
)
data_path = "../../data/preprocessed/"

# # Carga de datos

data = pd.read_csv(data_path+"data_day.csv")
data

# # Una salida

# ## XGBoost

scaler = StandardScaler()

data_scaled = scaler.fit_transform(data)

X_scaled = data_scaled[:,:4]
y_scaled = data_scaled[:,5]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, shuffle = True)

import xgboost
from sklearn.model_selection import GridSearchCV

params = {'max_depth': [10, 100],
          'learning_rate': [0.01, 0.001],
          'n_estimators': [100, 1000],
          'colsample_bytree': [0.3, 0.9],
          'subsample': [0.6, 1],
          'gamma': [0, 20]
         }

xg_model = xgboost.XGBRegressor(seed = 824)
clf = GridSearchCV(estimator=xg_model, 
                   param_grid=params,
                   cv = 5,
                   scoring='r2', 
                   verbose=3)
clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (clf.best_score_)**(1/2))

predictions = clf.predict(X_test)
print(r2_score(y_test, predictions))

# +
fig = make_subplots(rows=2, cols=1, subplot_titles=("Producción", "Energía"))

fig.add_trace(
    go.Scatter(x=y_test, y=predictions, name="Production", mode="markers"),
    row=1, col=1
)
fig.add_shape(
    type="line",
    x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    ),
     row=1, col=1
)

fig.update_layout(height=1000, width=900, title_text="Real vs Pred")
fig.show()


# -

# ## Redes neuropna

def build_model():
    # Tamaño de las entradas
    shape = (4, )
    
    # Entradas
    input_layer = Input(shape=shape, name="input")
    #normalization = Normalization()(input_layer)
    
    # 4 capas densas con activación ReLu
    x = Dense(16, kernel_initializer='normal', activation="relu")(input_layer)
    x = Dense(32, kernel_initializer='normal', activation="relu")(x)
    x = Dense(64, kernel_initializer='normal', activation="relu")(x)
    x = Dense(32, kernel_initializer='normal', activation="relu")(x)
    
    energy = Dense(1, kernel_initializer='normal', activation="linear", name="energy")(x)
    
    model = tf.keras.Model(inputs = input_layer, outputs = energy)
    return model


model = build_model()
model.compile(optimizer=Adam(learning_rate=1e-3), 
             loss = "mse",
             metrics="mse")

earlyStopping = EarlyStopping(
    monitor="val_energy_mse",
    patience=200,
    restore_best_weights=True
)

history = model.fit(X_train, y_train, 
         epochs = 5000,
         batch_size = 64,
         callbacks = [earlyStopping],
         validation_data=(X_test, y_test),
         verbose = 1)

# +
# Plot the training accuracy

mse_keys = [k for k in history.history.keys() if k in ('mse', 'val_mse')] 
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

model.evaluate(X_test, y_test)

# # Multisalida

# # Estandarización de los datos

X = data[data.columns[:4]]
y = data[data.columns[4:6]]

# # Train/test split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, shuffle = True)

energy_train = Y_train["Consumo Bombas [MBTU]"]
energy_test = Y_test["Consumo Bombas [MBTU]"]
production_train = Y_train["Volumen Transportado  [bls]"]
production_test = Y_test["Volumen Transportado  [bls]"]


# # Modelo

# ## Contrucción del modelo

def build_model():
    # Tamaño de las entradas
    shape = (4, )
    
    # Entradas
    input_layer = Input(shape=shape, name="input")
    #normalization = Normalization()(input_layer)
    
    # 4 capas densas con activación ReLu
    x = Dense(16, kernel_initializer='normal', activation="relu")(input_layer)
    x = Dense(32, kernel_initializer='normal', activation="relu")(x)
    x = Dense(64, kernel_initializer='normal', activation="relu")(x)
    x = Dense(32, kernel_initializer='normal', activation="relu")(x)
    
    energy = Dense(1, kernel_initializer='normal', activation="linear", name="energy")(x)
    production = Dense(1, kernel_initializer='normal', activation="linear", name="production")(x)
    list_outputs = [energy, production]
    
    model = tf.keras.Model(inputs = input_layer, outputs = list_outputs)
    return model


model = build_model()

# ## Compilación del modelo

model.compile(optimizer=Adam(learning_rate=0.00001), 
             loss = ["mse", "mse"],
             metrics=['mse', 'mae'])

tf.keras.utils.plot_model(model, show_shapes = True)

model.summary()

# ## Asignación de datos de entrenamiento

input_train = X_train
outputs_train = {'energy': energy_train, 'production': production_train}

input_test = X_test
outputs_test = {'energy': energy_test, 'production': production_test}

# ## Callbacks

# * Early Stopping

earlyStopping = EarlyStopping(
    monitor="val_energy_mse",
    patience=200,
    restore_best_weights=True
)

# ## Entrenamiento

history = model.fit(input_train, outputs_train, 
         epochs = 5000,
         batch_size = 64,
         #callbacks = [earlyStopping],
         validation_data=(input_test, outputs_test),
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

model.evaluate(X_test, [energy_test, production_test])

preds = model.predict(X_test)

y_pred_energy = preds[0][:,0]
y_pred_production = preds[1][:,0]

# +
x_graph = np.arange(len(Y_test))

fig = go.Figure()
# Energy
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = energy_test,  # Datos del eje Y.
                         name = 'Real', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = y_pred_energy,  # Datos del eje Y.
                         name = 'Predicción', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))
# -

fig = go.Figure()
# Production
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = production_test,  # Datos del eje Y.
                         name = 'Real', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))
fig.add_trace(go.Scatter(x = x_graph,  # Datos del eje X.
                         y = y_pred_production,  # Datos del eje Y.
                         name = 'Predicción', # Nombre del objeto Scatter.
                         mode='lines+markers'   # Modo de la línea (Líneas y puntos)                        
                         ))

# +
fig = make_subplots(rows=2, cols=1, subplot_titles=("Producción", "Energía"))

fig.add_trace(
    go.Scatter(x=production_test, y=y_pred_production, name="Production", mode="markers"),
    row=1, col=1
)
fig.add_shape(
    type="line",
    x0=60000, y0=60000, x1=production_test.max(), y1=production_test.max(),
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    ),
     row=1, col=1
)
fig.add_trace(
    go.Scatter(x=energy_test, y=y_pred_energy, name="Energy", mode="markers"),
    row=2, col=1
)
fig.add_shape(
    type="line",
    x0=500, y0=500, x1=energy_test.max(), y1=energy_test.max(),
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
model.save('../../models/optimization/model.h5')

# # XGBoost

import xgboost
from sklearn.model_selection import GridSearchCV

params = {'max_depth': [3, 10, 100, 1000],
          'learning_rate': [0.01, 0.1, 0.5, 0.001],
          'n_estimators': [100, 500, 1000],
          'colsample_bytree': [0.3, 0.7, 0.9],
          'subsample': [0.6 ,0.8, 1],
          'gamma': [0, 10, 20]
         }

xg_model = xgboost.XGBRegressor(seed = 824)
clf = GridSearchCV(estimator=xg_model, 
                   param_grid=params,
                   cv = 5,
                   scoring='r2', 
                   verbose=3)
clf.fit(X_train, Y_train)

print("Best parameters:", clf.best_params_)
print("Lowest RMSE: ", (clf.best_score_)**(1/2.0))

predictions = clf.predict(X_test)
y_pred_xgb = scaler.inverse_transform(np.transpose([predictions[:, 0], predictions[:, 1]]))
print(r2_score(Y_test, predictions))

# +
fig = make_subplots(rows=2, cols=1, subplot_titles=("Producción", "Energía"))

fig.add_trace(
    go.Scatter(x=y_test_inverse[:, 0], y=y_pred_xgb[:, 0], name="Production", mode="markers"),
    row=1, col=1
)
fig.add_shape(
    type="line",
    x0=60000, y0=60000, x1=y_test_inverse[:, 0].max(), y1=y_test_inverse[:, 0].max(),
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    ),
     row=1, col=1
)
fig.add_trace(
    go.Scatter(x=y_test_inverse[:, 1], y=y_pred_xgb[:, 1], name="Energy", mode="markers"),
    row=2, col=1
)
fig.add_shape(
    type="line",
    x0=500, y0=500, x1=y_test_inverse[:, 1].max(), y1=y_test_inverse[:, 1].max(),
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




