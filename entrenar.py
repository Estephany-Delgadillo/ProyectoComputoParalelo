import cv2
import numpy as np
import os
import pickle

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

CARPETA = "dataset"

CLASES = [
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111"
]




def extraer_caracteristicas(imagen):
    """Vector de 1024 pixeles binarios; igual en entrenamiento e inferencia."""
    imagen = cv2.resize(imagen, (32, 32), interpolation=cv2.INTER_AREA)
    _, imagen = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return (imagen.astype(np.float32) / 255.0).reshape(-1)


X = []
y = []

print()
print("CARGANDO DATASET")
print("-" * 40)

for clase in CLASES:

    ruta = os.path.join(
        CARPETA,
        clase
    )

    archivos = [
        archivo
        for archivo in os.listdir(ruta)
        if archivo.lower().endswith(".png")
    ]

    cantidad = 0

    for archivo in archivos:

        ruta_imagen = os.path.join(
            ruta,
            archivo
        )

        imagen = cv2.imread(
            ruta_imagen,
            cv2.IMREAD_GRAYSCALE
        )

        if imagen is None:
            continue

        caracteristicas = extraer_caracteristicas(
            imagen
        )

        X.append(caracteristicas)
        y.append(clase)

        cantidad += 1

    print(
        f"{clase}: {cantidad} imagenes"
    )


X = np.array(
    X,
    dtype=np.float32
)

y = np.array(y)

print()
print("Total de muestras:", len(X))
print(
    "Caracteristicas de pixeles por imagen:",
    X.shape[1]
)

print()
print("Dividiendo dataset...")


X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


print(
    "Entrenamiento:",
    len(X_entrenamiento)
)

print(
    "Prueba:",
    len(X_prueba)
)


print()
print("ENTRENANDO SVM...")
print()


modelo = SVC(
    kernel="rbf",
    C=10,
    gamma="scale"
)

modelo.fit(
    X_entrenamiento,
    y_entrenamiento
)


predicciones = modelo.predict(
    X_prueba
)


precision = accuracy_score(
    y_prueba,
    predicciones
)


print("=" * 50)
print("RESULTADOS")
print("=" * 50)

print()
print(
    f"Exactitud general: "
    f"{precision * 100:.2f}%"
)


print()
print("REPORTE POR CLASE")
print()

print(
    classification_report(
        y_prueba,
        predicciones,
        labels=CLASES,
        zero_division=0
    )
)


print("MATRIZ DE CONFUSION")
print()

matriz = confusion_matrix(
    y_prueba,
    predicciones,
    labels=CLASES
)

print(matriz)


print()
print("Orden de las clases:")

for i, clase in enumerate(CLASES):
    print(
        f"{i} -> {clase}"
    )


datos_modelo = {
    "modelo": modelo,
    "clases": CLASES
}


with open(
    "modelo_3bits.pkl",
    "wb"
) as archivo:

    pickle.dump(
        datos_modelo,
        archivo
    )


print()
print("=" * 50)
print("Modelo guardado como modelo_3bits.pkl")
print("=" * 50)