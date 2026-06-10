from flask import Flask, render_template, request, jsonify
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# =========================
# BASE DE CONOCIMIENTO
# =========================

productos = [
    {
        "nombre": "Polo básico",
        "categoria": "polo",
        "precio": 35,
        "tallas": ["S", "M", "L"],
        "color": "blanco",
        "oferta": "2 polos por S/60"
    },

    {
        "nombre": "Casaca jean",
        "categoria": "casaca",
        "precio": 120,
        "tallas": ["M", "L"],
        "color": "azul",
        "oferta": "10% de descuento"
    },

    {
        "nombre": "Vestido casual",
        "categoria": "vestido",
        "precio": 90,
        "tallas": ["S", "M"],
        "color": "rojo",
        "oferta": "S/10 de descuento"
    },

    {
        "nombre": "Pantalón cargo",
        "categoria": "pantalon",
        "precio": 85,
        "tallas": ["M", "L", "XL"],
        "color": "negro",
        "oferta": "Envío gratis"
    }
]

# =========================
# PREPROCESAMIENTO
# =========================

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[¿?¡!.,]', '', texto)
    return texto.strip()

# =========================
# BUSQUEDA DE PRODUCTOS
# =========================

def buscar_producto(consulta):
    consulta = limpiar_texto(consulta)

    for producto in productos:
        if producto["categoria"] in consulta:
            return producto

        if producto["nombre"].lower() in consulta:
            return producto

    return None

# =========================
# DATOS DE ENTRENAMIENTO
# =========================

datos_entrenamiento = [
    ("cuanto cuesta el polo", "precio"),
    ("precio de la casaca", "precio"),
    ("valor del vestido", "precio"),

    ("que ofertas tienen", "oferta"),
    ("hay promociones", "oferta"),
    ("descuentos disponibles", "oferta"),

    ("tienen talla m", "talla"),
    ("que tallas hay", "talla"),
    ("disponible en talla l", "talla"),

    ("que colores tienen", "color"),
    ("color de la casaca", "color"),
    ("hay en color rojo", "color"),

    ("hola", "saludo"),
    ("buenos dias", "saludo"),
    ("buenas tardes", "saludo")
]

X = [x[0] for x in datos_entrenamiento]
y = [x[1] for x in datos_entrenamiento]

modelo = Pipeline([
    ("vectorizador", CountVectorizer()),
    ("clasificador", MultinomialNB())
])

modelo.fit(X, y)

# =========================
# CHATBOT
# =========================

def responder_chatbot(consulta):

    consulta_limpia = limpiar_texto(consulta)

    if consulta_limpia == "":
        return "Escribe una consulta."

    intencion = modelo.predict([consulta_limpia])[0]

    producto = buscar_producto(consulta_limpia)

    if intencion == "saludo":
        return "Hola. Bienvenido a Moda Perú."

    if intencion == "precio":
        if producto:
            return f"El precio de {producto['nombre']} es S/{producto['precio']}."
        return "¿De qué producto deseas conocer el precio?"

    if intencion == "oferta":
        if producto:
            return f"La oferta de {producto['nombre']} es {producto['oferta']}."

        ofertas = []

        for p in productos:
            ofertas.append(f"{p['nombre']} : {p['oferta']}")

        return "Ofertas disponibles: " + " | ".join(ofertas)

    if intencion == "talla":
        if producto:
            return f"Tallas disponibles: {', '.join(producto['tallas'])}"

        return "Indica el producto para revisar tallas."

    if intencion == "color":
        if producto:
            return f"Color disponible: {producto['color']}"

        return "Indica el producto para revisar colores."

    return "No entendí tu consulta."

# =========================
# RUTAS FLASK
# =========================

@app.route("/")
def inicio():
    return render_template("index.html", productos=productos)

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    mensaje = data["mensaje"]

    respuesta = responder_chatbot(mensaje)

    return jsonify({
        "respuesta": respuesta
    })

# =========================

if __name__ == "__main__":
    app.run(debug=True)