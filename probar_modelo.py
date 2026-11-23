import cv2
import numpy as np
import pickle
from collections import deque

TAMANO = 200

# Cargar modelo
with open("modelo_3bits.pkl", "rb") as archivo:
    datos = pickle.load(archivo)

modelo = datos["modelo"]
CLASES = datos["clases"]



# Historial para estabilizar predicciones
historial = deque(maxlen=9)


def crear_mascara(zona):

    zona_suave = cv2.GaussianBlur(
        zona,
        (7, 7),
        0
    )

    ycrcb = cv2.cvtColor(
        zona_suave,
        cv2.COLOR_BGR2YCrCb
    )

    inferior = np.array(
        [0, 133, 77],
        dtype=np.uint8
    )

    superior = np.array(
        [255, 173, 127],
        dtype=np.uint8
    )

    mascara = cv2.inRange(
        ycrcb,
        inferior,
        superior
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_OPEN,
        kernel
    )

    mascara = cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        kernel
    )

    mascara = cv2.GaussianBlur(
        mascara,
        (5, 5),
        0
    )

    _, mascara = cv2.threshold(
        mascara,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return mascara


def normalizar_mano(mascara):

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos:
        return None

    contorno = max(
        contornos,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contorno)

    if area < 4000:
        return None

    x, y, w, h = cv2.boundingRect(contorno)

    mano = mascara[
        y:y + h,
        x:x + w
    ]

    if mano.size == 0:
        return None

    lado = max(w, h) + 40

    cuadrado = np.zeros(
        (lado, lado),
        dtype=np.uint8
    )

    inicio_x = (lado - w) // 2
    inicio_y = (lado - h) // 2

    cuadrado[
        inicio_y:inicio_y + h,
        inicio_x:inicio_x + w
    ] = mano

    cuadrado = cv2.resize(
        cuadrado,
        (TAMANO, TAMANO),
        interpolation=cv2.INTER_AREA
    )

    _, cuadrado = cv2.threshold(
        cuadrado,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return cuadrado


def extraer_caracteristicas(imagen):
    """Vector de 1024 pixeles binarios; igual en entrenamiento e inferencia."""
    imagen = cv2.resize(imagen, (32, 32), interpolation=cv2.INTER_AREA)
    _, imagen = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)
    return (imagen.astype(np.float32) / 255.0).reshape(-1)


def estabilizar():

    if len(historial) < 5:
        return None

    conteo = {}

    for prediccion in historial:

        if prediccion not in conteo:
            conteo[prediccion] = 0

        conteo[prediccion] += 1

    return max(
        conteo,
        key=conteo.get
    )


camara = cv2.VideoCapture(0)

camara.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

camara.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)

if not camara.isOpened():
    print("No se pudo abrir la camara")
    exit()


while True:

    correcto, frame = camara.read()

    if not correcto:
        break

    frame = cv2.flip(
        frame,
        1
    )

    alto, ancho, _ = frame.shape

    x1 = int(ancho * 0.25)
    y1 = int(alto * 0.10)

    x2 = int(ancho * 0.75)
    y2 = int(alto * 0.90)

    zona = frame[
        y1:y2,
        x1:x2
    ]

    mascara = crear_mascara(
        zona
    )

    mano = normalizar_mano(
        mascara
    )

    resultado = None

    if mano is not None:

        caracteristicas = extraer_caracteristicas(
            mano
        )

        caracteristicas = caracteristicas.reshape(
            1,
            -1
        )

        prediccion = modelo.predict(
            caracteristicas
        )[0]

        historial.append(
            prediccion
        )

        resultado = estabilizar()

        cv2.imshow(
            "Mano normalizada",
            mano
        )

    else:

        historial.clear()

        vacia = np.zeros(
            (TAMANO, TAMANO),
            dtype=np.uint8
        )

        cv2.imshow(
            "Mano normalizada",
            vacia
        )

    if resultado is not None:

        decimal = int(
            resultado,
            2
        )

        texto_binario = (
            f"BINARIO: {resultado}"
        )

        texto_decimal = (
            f"DECIMAL: {decimal}"
        )

        color = (0, 255, 0)

    else:

        texto_binario = "BINARIO: ---"
        texto_decimal = "DECIMAL: -"

        color = (0, 200, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    cv2.putText(
        frame,
        texto_binario,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2
    )

    cv2.putText(
        frame,
        texto_decimal,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2
    )

    cv2.putText(
        frame,
        "ESC = salir",
        (20, 455),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Prueba Traductor Binario 3 Bits",
        frame
    )

    cv2.imshow(
        "Mascara",
        mascara
    )

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == 27:
        break


camara.release()
cv2.destroyAllWindows()