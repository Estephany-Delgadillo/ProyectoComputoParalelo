import cv2
import numpy as np

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la camara")
    exit()

hog = cv2.HOGDescriptor(
    (200, 200),
    (40, 40),
    (20, 20),
    (20, 20),
    9
)

while True:

    correcto, frame = camara.read()

    if not correcto:
        break

    frame = cv2.flip(frame, 1)

    alto, ancho = frame.shape[:2]

    x1 = int(ancho * 0.25)
    y1 = int(alto * 0.10)
    x2 = int(ancho * 0.75)
    y2 = int(alto * 0.90)

    roi = frame[y1:y2, x1:x2]

    suavizada = cv2.GaussianBlur(roi, (7, 7), 0)

    ycrcb = cv2.cvtColor(
        suavizada,
        cv2.COLOR_BGR2YCrCb
    )

    limite_inferior = np.array([0, 133, 77], dtype=np.uint8)
    limite_superior = np.array([255, 173, 127], dtype=np.uint8)

    mascara = cv2.inRange(
        ycrcb,
        limite_inferior,
        limite_superior
    )

    kernel = np.ones((5, 5), np.uint8)

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

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    estado = "BUSCANDO MANO"
    caracteristicas_hog = 0

    silueta = np.zeros((200, 200), dtype=np.uint8)

    if len(contornos) > 0:

        contorno = max(
            contornos,
            key=cv2.contourArea
        )

        area = cv2.contourArea(contorno)

        if area > 4000:

            estado = "MANO DETECTADA"

            x, y, w, h = cv2.boundingRect(contorno)

            recorte = mascara[y:y+h, x:x+w]

            lado = max(w, h) + 80

            imagen = np.zeros(
                (lado, lado),
                dtype=np.uint8
            )

            inicio_x = (lado - w) // 2
            inicio_y = (lado - h) // 2

            imagen[
                inicio_y:inicio_y+h,
                inicio_x:inicio_x+w
            ] = recorte

            silueta = cv2.resize(
                imagen,
                (200, 200)
            )

            _, silueta = cv2.threshold(
                silueta,
                127,
                255,
                cv2.THRESH_BINARY
            )

            caracteristicas = hog.compute(silueta)

            caracteristicas_hog = len(caracteristicas)

            cv2.rectangle(
                frame,
                (x1 + x, y1 + y),
                (x1 + x + w, y1 + y + h),
                (0, 255, 0),
                2
            )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 200, 0),
        2
    )

    cv2.rectangle(
        frame,
        (0, 0),
        (ancho, 110),
        (25, 25, 25),
        -1
    )

    cv2.putText(
        frame,
        "TRADUCTOR BINARIO - AVANCE 50%",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    color_estado = (
        (0, 255, 0)
        if estado == "MANO DETECTADA"
        else (0, 165, 255)
    )

    cv2.putText(
        frame,
        estado,
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_estado,
        2
    )

    cv2.putText(
        frame,
        f"Caracteristicas HOG: {caracteristicas_hog}",
        (15, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Traductor Binario - Avance",
        frame
    )

    cv2.imshow(
        "Segmentacion de la mano",
        mascara
    )

    cv2.imshow(
        "Silueta normalizada",
        silueta
    )

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()