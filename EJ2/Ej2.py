import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =============================================
# FASE 2: DATOS
# =============================================
ventas = {
    "Semana": [1,2,3,4,5,6,7,8,9,10,11,12],
    "Ventas": [120,130,125,140,150,160,170,180,175,190,200,210]
}
df = pd.DataFrame(ventas)
print("=== TABLA DE DATOS ===")
print(df)

# =============================================
# FASE 3: VISUALIZACIÓN Y GUARDADO
# =============================================
plt.figure(figsize=(8,4))
plt.plot(df["Semana"], df["Ventas"], marker="o", color="blue")
plt.title("Ventas Semanales")
plt.xlabel("Semana")
plt.ylabel("Ventas")
plt.grid(True)
plt.savefig(r"D:\GUIA11_SI\EJ2\Gráficos\ventas_semanales.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado en: D:\\GUIA11_SI\\EJ2\\Gráficos\\ventas_semanales.png")

# =============================================
# FASE 4: MODELO PREDICTIVO
# =============================================
X = df[["Semana"]]
y = df["Ventas"]

modelo = LinearRegression()
modelo.fit(X, y)

prediccion = modelo.predict([[13]])
print("\n=== PREDICCIÓN ===")
print("Predicción Semana 13:", prediccion[0])

# =============================================
# FASE 5: EVALUACIÓN R²
# =============================================
predicciones = modelo.predict(X)
r2 = r2_score(y, predicciones)
print("\n=== EVALUACIÓN DEL MODELO ===")
print("R²:", round(r2, 4))

if r2 >= 0.9:
    print("Interpretación: Excelente ajuste del modelo")
elif r2 >= 0.7:
    print("Interpretación: Buen ajuste del modelo")
else:
    print("Interpretación: Ajuste deficiente")