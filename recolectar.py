import cv2
import numpy as np
import os

CLASES = ["000", "001", "010", "011", "100", "101", "110", "111"]

for clase in CLASES:
    os.makedirs(os.path.join("dataset", clase), exist_ok=True)

camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("No se pudo abrir la camara")
    exit()

clase_actual = 0
contador = 0
capturando = False

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

    ycrcb = cv2.cvtColor(suavizada, cv2.COLOR_BGR2YCrCb)

    mascara = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8)
    )

    kernel = np.ones((5, 5), np.uint8)

    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    mascara = cv2.GaussianBlur(mascara, (5, 5), 0)

    _, mascara = cv2.threshold(
        mascara, 127, 255, cv2.THRESH_BINARY
    )

    contornos, _ = cv2.findContours(
        mascara,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    silueta = None

    if contornos:
        contorno = max(contornos, key=cv2.contourArea)

        if cv2.contourArea(contorno) > 4000:
            x, y, w, h = cv2.boundingRect(contorno)

            recorte = mascara[y:y+h, x:x+w]

            lado = max(w, h) + 80

            imagen = np.zeros((lado, lado), dtype=np.uint8)

            inicio_x = (lado - w) // 2
            inicio_y = (lado - h) // 2

            imagen[
                inicio_y:inicio_y+h,
                inicio_x:inicio_x+w
            ] = recorte

            silueta = cv2.resize(imagen, (200, 200))

            _, silueta = cv2.threshold(
                silueta, 127, 255, cv2.THRESH_BINARY
            )

    clase = CLASES[clase_actual]

    cv2.rectangle(
        frame, (x1, y1), (x2, y2), (255, 200, 0), 2
    )

    cv2.putText(
        frame,
        f"Gesto: {clase}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Imagenes: {contador}/150",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "ESPACIO: Capturar | N: Siguiente | Q: Salir",
        (10, alto - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1
    )

    cv2.imshow("Recoleccion de datos", frame)

    if silueta is not None:
        cv2.imshow("Silueta", silueta)

    tecla = cv2.waitKey(1) & 0xFF

    if tecla == ord(" "):
        if silueta is not None and contador < 150:
            ruta = os.path.join(
                "dataset",
                clase,
                f"{contador:03d}.png"
            )

            cv2.imwrite(ruta, silueta)
            contador += 1

    elif tecla == ord("n"):
        if clase_actual < len(CLASES) - 1:
            clase_actual += 1
            contador = 0

    elif tecla == ord("q"):
        break

camara.release()
cv2.destroyAllWindows()