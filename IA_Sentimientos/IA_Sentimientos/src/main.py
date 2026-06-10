import os
import pandas as pd
import matplotlib.pyplot as plt

# TensorFlow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Scikit-Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# CREAR CARPETAS SI NO EXISTEN

os.makedirs("../capturas", exist_ok=True)
os.makedirs("../resultados", exist_ok=True)

# DATASET

datos = {
    "texto": [
        "Excelente servicio",
        "Muy mala atención",
        "Producto de calidad",
        "No recomiendo comprar",
        "Entrega rápida",
        "Demasiado lento",
        "Muy satisfecho",
        "Pésima experiencia"
    ],
    "sentimiento": [1,0,1,0,1,0,1,0]
}

df = pd.DataFrame(datos)

print("\nDATAFRAME")
print(df)

# CAPTURA 1 - DATAFRAME

fig, ax = plt.subplots(figsize=(8,3))

ax.axis('off')

tabla = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center'
)

tabla.auto_set_font_size(False)
tabla.set_fontsize(10)

plt.savefig(
    "../capturas/01_dataframe.png",
    bbox_inches="tight"
)

plt.close()

# CAPTURA 2 - DISTRIBUCIÓN

conteo = df["sentimiento"].value_counts()

plt.figure(figsize=(6,4))

conteo.plot(kind="bar")

plt.title("Distribución de Sentimientos")
plt.xlabel("Sentimiento")
plt.ylabel("Cantidad")

plt.savefig(
    "../capturas/02_distribucion.png",
    bbox_inches="tight"
)

plt.close()

# VARIABLES

X = df["texto"]
y = df["sentimiento"]


# VECTORIZACIÓN

vectorizador = CountVectorizer()

X = vectorizador.fit_transform(X)

# 
# DIVISIÓN DE DATOS

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

#
# MODELO DEEP LEARNING

modelo = Sequential()

modelo.add(
    Dense(
        16,
        activation="relu",
        input_shape=(X_train.shape[1],)
    )
)

modelo.add(
    Dense(
        8,
        activation="relu"
    )
)

modelo.add(
    Dense(
        1,
        activation="sigmoid"
    )
)


# COMPILAR

modelo.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ENTRENAR


print("\nEntrenando modelo...\n")

historial = modelo.fit(
    X_train.toarray(),
    y_train,
    epochs=20,
    verbose=1
)


# CAPTURA 3 - ACCURACY


plt.figure(figsize=(8,5))

plt.plot(
    historial.history["accuracy"]
)

plt.title("Accuracy por Época")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.savefig(
    "../capturas/03_accuracy.png",
    bbox_inches="tight"
)

plt.close()


# CAPTURA 4 - LOSS
plt.figure(figsize=(8,5))

plt.plot(
    historial.history["loss"]
)

plt.title("Loss por Época")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.savefig(
    "../capturas/04_loss.png",
    bbox_inches="tight"
)

plt.close()


# EVALUACIÓN


loss, accuracy = modelo.evaluate(
    X_test.toarray(),
    y_test,
    verbose=0
)

print("\n================================")
print("RESULTADOS")
print("================================")
print("Loss:", loss)
print("Accuracy:", accuracy)


# CAPTURA 5 - RESULTADO

fig, ax = plt.subplots(figsize=(7,3))

ax.axis("off")

texto = f"""
RESULTADOS DEL MODELO

Loss: {loss:.4f}

Accuracy: {accuracy:.4f}
"""

ax.text(
    0.05,
    0.5,
    texto,
    fontsize=12
)

plt.savefig(
    "../capturas/05_resultados.png",
    bbox_inches="tight"
)

plt.close()


# PREDICCIÓN
nuevo_texto = [
    "Excelente producto"
]

nuevo_vector = vectorizador.transform(
    nuevo_texto
)

prediccion = modelo.predict(
    nuevo_vector.toarray(),
    verbose=0
)

probabilidad = prediccion[0][0]

resultado = (
    "POSITIVO"
    if probabilidad > 0.5
    else "NEGATIVO"
)

print("\n================================")
print("PREDICCIÓN")
print("================================")
print("Texto:", nuevo_texto[0])
print("Probabilidad:", probabilidad)
print("Resultado:", resultado)


# CAPTURA 6 - PREDICCIÓN

fig, ax = plt.subplots(figsize=(8,3))

ax.axis("off")

ax.text(
    0.05,
    0.5,
    f"Texto Analizado:\n{nuevo_texto[0]}\n\nPredicción: {resultado}",
    fontsize=12
)

plt.savefig(
    "../capturas/06_prediccion.png",
    bbox_inches="tight"
)

plt.close()

print("\n================================")
print("CAPTURAS GENERADAS")
print("================================")
print("01_dataframe.png")
print("02_distribucion.png")
print("03_accuracy.png")
print("04_loss.png")
print("05_resultados.png")
print("06_prediccion.png")
print("\nRevisar carpeta CAPTURAS")