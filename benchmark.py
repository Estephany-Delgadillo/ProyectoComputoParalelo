import cv2
import numpy as np
import pickle
import os
import time
import multiprocessing

CARPETA = "dataset"

CLASES = [
    "000", "001", "010", "011",
    "100", "101", "110", "111"
]

REPETICIONES = 5

modelo_proceso = None


def extraer_caracteristicas(imagen):
    """Vector de 1024 pixeles binarios; igual en entrenamiento e inferencia."""
    imagen = cv2.resize(imagen, (32, 32), interpolation=cv2.INTER_AREA)
    _, imagen = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return (imagen.astype(np.float32) / 255.0).reshape(-1)

def cargar_modelo():

    with open("modelo_3bits.pkl", "rb") as archivo:
        datos = pickle.load(archivo)

    return datos["modelo"]


def cargar_rutas():

    datos = []

    for clase in CLASES:

        carpeta = os.path.join(
            CARPETA,
            clase
        )

        for archivo in os.listdir(carpeta):

            if archivo.lower().endswith(".png"):

                ruta = os.path.join(
                    carpeta,
                    archivo
                )

                datos.append(
                    (ruta, clase)
                )

    return datos


def procesar_imagen_secuencial(
    dato,
    modelo,
):

    ruta, clase_real = dato

    imagen = cv2.imread(
        ruta,
        cv2.IMREAD_GRAYSCALE
    )

    if imagen is None:
        return clase_real, None

    caracteristicas = extraer_caracteristicas(imagen)

    prediccion = modelo.predict(
        caracteristicas.reshape(1, -1)
    )[0]

    return clase_real, prediccion


def inicializar_proceso():

    global modelo_proceso

    modelo_proceso = cargar_modelo()


def procesar_imagen_paralela(dato):

    global modelo_proceso

    ruta, clase_real = dato

    imagen = cv2.imread(
        ruta,
        cv2.IMREAD_GRAYSCALE
    )

    if imagen is None:
        return clase_real, None

    caracteristicas = extraer_caracteristicas(imagen)

    prediccion = modelo_proceso.predict(
        caracteristicas.reshape(1, -1)
    )[0]

    return clase_real, prediccion


def prueba_secuencial(datos):

    modelo = cargar_modelo()

    resultados = []

    inicio = time.perf_counter()

    for dato in datos:

        resultado = procesar_imagen_secuencial(
            dato,
            modelo
        )

        resultados.append(resultado)

    tiempo = time.perf_counter() - inicio

    return resultados, tiempo


def prueba_paralela(
    datos,
    procesos
):

    inicio = time.perf_counter()

    with multiprocessing.Pool(
        processes=procesos,
        initializer=inicializar_proceso
    ) as pool:

        resultados = pool.map(
            procesar_imagen_paralela,
            datos
        )

    tiempo = time.perf_counter() - inicio

    return resultados, tiempo


def calcular_precision(resultados):

    correctos = 0
    total = 0

    for real, prediccion in resultados:

        if prediccion is None:
            continue

        total += 1

        if real == prediccion:
            correctos += 1

    if total == 0:
        return 0

    return (
        correctos / total
    ) * 100


if __name__ == "__main__":

    multiprocessing.freeze_support()

    originales = cargar_rutas()

    datos = originales * REPETICIONES

    nucleos = multiprocessing.cpu_count()

    procesos = min(
        4,
        nucleos
    )

    print()
    print("=" * 55)
    print("BENCHMARK DEFINITIVO - TRADUCTOR BINARIO")
    print("=" * 55)

    print(
        f"Imagenes originales : {len(originales)}"
    )

    print(
        f"Repeticiones        : {REPETICIONES}"
    )

    print(
        f"Procesamientos      : {len(datos)}"
    )

    print(
        f"Nucleos logicos     : {nucleos}"
    )

    print(
        f"Procesos utilizados : {procesos}"
    )

    print()
    print("Ejecutando version secuencial...")

    resultados_sec, tiempo_sec = (
        prueba_secuencial(datos)
    )

    precision_sec = calcular_precision(
        resultados_sec
    )

    print(
        f"Tiempo secuencial: {tiempo_sec:.4f} s"
    )

    print(
        f"Precision: {precision_sec:.2f}%"
    )

    print()
    print("Ejecutando version paralela...")

    resultados_par, tiempo_par = (
        prueba_paralela(
            datos,
            procesos
        )
    )

    precision_par = calcular_precision(
        resultados_par
    )

    speedup = tiempo_sec / tiempo_par

    eficiencia = (
        speedup / procesos
    ) * 100

    reduccion = (
        (tiempo_sec - tiempo_par)
        / tiempo_sec
    ) * 100

    fps_sec = (
        len(datos) / tiempo_sec
    )

    fps_par = (
        len(datos) / tiempo_par
    )

    print()
    print("=" * 55)
    print("RESULTADOS DEFINITIVOS")
    print("=" * 55)

    print(
        f"Tiempo secuencial   : {tiempo_sec:.4f} s"
    )

    print(
        f"Tiempo paralelo     : {tiempo_par:.4f} s"
    )

    print(
        f"Speedup             : {speedup:.3f}x"
    )

    print(
        f"Eficiencia          : {eficiencia:.2f}%"
    )

    print(
        f"Reduccion de tiempo : {reduccion:.2f}%"
    )

    print(
        f"FPS secuencial      : {fps_sec:.2f}"
    )

    print(
        f"FPS paralelo        : {fps_par:.2f}"
    )

    print(
        f"Precision secuencial: {precision_sec:.2f}%"
    )

    print(
        f"Precision paralela  : {precision_par:.2f}%"
    )

    print("=" * 55)