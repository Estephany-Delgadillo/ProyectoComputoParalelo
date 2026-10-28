import cv2
import numpy as np
import threading
import queue
import time

cola_frames = queue.Queue(maxsize=2)
detener = threading.Event()

frames_capturados = 0
frames_procesados = 0

lock = threading.Lock()

hog = cv2.HOGDescriptor(
    (200, 200),
    (40, 40),
    (20, 20),
    (20, 20),
    9
)


def capturar_camara():

    global frames_capturados

    camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("No se pudo abrir la camara")
        detener.set()
        return

    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while not detener.is_set():

        correcto, frame = camara.read()

        if not correcto:
            detener.set()
            break

        frame = cv2.flip(frame, 1)

        with lock:
            frames_capturados += 1

        if cola_frames.full():
            try:
                cola_frames.get_nowait()
            except queue.Empty:
                pass

        try:
            cola_frames.put_nowait(frame)
        except queue.Full:
            pass

    camara.release()


def procesar_imagen():

    global frames_procesados

    while not detener.is_set() or not cola_frames.empty():

        try:
            frame = cola_frames.get(timeout=0.1)
        except queue.Empty:
            continue

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

        mascara = cv2.inRange(
            ycrcb,
            np.array([0, 133, 77], dtype=np.uint8),
            np.array([255, 173, 127], dtype=np.uint8)
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

        if contornos:

            contorno = max(
                contornos,
                key=cv2.contourArea
            )

            if cv2.contourArea(contorno) > 4000:

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
            (ancho, 115),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "TRADUCTOR BINARIO - AVANCE",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            estado,
            (15, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
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

        cv2.putText(
            frame,
            "Captura y procesamiento concurrentes",
            (10, alto - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1
        )

        with lock:
            frames_procesados += 1

        cv2.imshow("Procesamiento concurrente", frame)
        cv2.imshow("Segmentacion", mascara)
        cv2.imshow("Silueta HOG", silueta)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord("q"):
            detener.set()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":

    print("Iniciando procesamiento concurrente...")

    inicio = time.perf_counter()

    hilo_camara = threading.Thread(
        target=capturar_camara
    )

    hilo_procesamiento = threading.Thread(
        target=procesar_imagen
    )

    hilo_camara.start()
    hilo_procesamiento.start()

    try:
        while hilo_camara.is_alive() and hilo_procesamiento.is_alive():
            hilo_camara.join(timeout=0.1)
            hilo_procesamiento.join(timeout=0.1)

    except KeyboardInterrupt:
        detener.set()

    finally:
        detener.set()

        hilo_camara.join()
        hilo_procesamiento.join()

        tiempo = time.perf_counter() - inicio

        print("\nResultados de ejecucion:")
        print("Frames capturados:", frames_capturados)
        print("Frames procesados:", frames_procesados)
        print("Tiempo total:", round(tiempo, 2), "segundos")

        if tiempo > 0:
            fps = frames_procesados / tiempo
            print("FPS de procesamiento:", round(fps, 2))

        print("Arquitectura: productor-consumidor con 2 hilos")