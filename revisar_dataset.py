import os
import cv2
import numpy as np

CARPETA = "dataset"

clases = [
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111"
]

print("\nREVISION DEL DATASET")
print("-" * 35)

total = 0
todo_correcto = True

for clase in clases:

    ruta = os.path.join(CARPETA, clase)

    if not os.path.exists(ruta):
        print(f"{clase}: carpeta no encontrada")
        todo_correcto = False
        continue

    archivos = [
        archivo
        for archivo in os.listdir(ruta)
        if archivo.lower().endswith(".png")
    ]

    cantidad = len(archivos)
    total += cantidad

    if cantidad == 150:
        estado = "OK"
    else:
        estado = "REVISAR"
        todo_correcto = False

    print(
        f"{clase}: {cantidad} imagenes - {estado}"
    )

print("-" * 35)
print(f"TOTAL: {total} imagenes")

if total == 1200 and todo_correcto:
    print("\nDATASET COMPLETO Y BALANCEADO")
else:
    print("\nHAY QUE REVISAR EL DATASET")


# Crear una vista con ejemplos de cada clase

ejemplos = []

for clase in clases:

    ruta = os.path.join(CARPETA, clase)

    if not os.path.exists(ruta):
        continue

    archivos = [
        archivo
        for archivo in os.listdir(ruta)
        if archivo.lower().endswith(".png")
    ]

    if len(archivos) == 0:
        continue

    # Elegir una imagen aproximadamente de la mitad
    archivo = archivos[len(archivos) // 2]

    imagen = cv2.imread(
        os.path.join(ruta, archivo),
        cv2.IMREAD_GRAYSCALE
    )

    if imagen is None:
        continue

    imagen = cv2.resize(
        imagen,
        (180, 180)
    )

    imagen_color = cv2.cvtColor(
        imagen,
        cv2.COLOR_GRAY2BGR
    )

    cv2.putText(
        imagen_color,
        clase,
        (60, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    ejemplos.append(imagen_color)


if len(ejemplos) == 8:

    fila1 = np.hstack(ejemplos[0:4])
    fila2 = np.hstack(ejemplos[4:8])

    mosaico = np.vstack([
        fila1,
        fila2
    ])

    cv2.imshow(
        "Ejemplos del dataset",
        mosaico
    )

    print(
        "\nPresiona cualquier tecla "
        "sobre la ventana para cerrar."
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()