import cv2
import numpy as np
import pickle
import customtkinter as ctk

from PIL import Image
from collections import deque


# Configuracion general

TAMANO = 200

FONDO = "#0B0F12"
FONDO_2 = "#101519"

TARJETA = "#12171B"
TARJETA_2 = "#0E1317"

BORDE = "#2A3035"

DORADO = "#C79A45"
DORADO_CLARO = "#E0BD72"

BLANCO = "#E8E8E8"

GRIS = "#8C9298"
GRIS_CLARO = "#B8BDC1"

ROJO = "#D64A4A"

VERDE = "#45B978"
VERDE_OSCURO = "#1D4934"

AZUL = "#48B9D4"


# Configuracion de CustomTkinter

ctk.set_appearance_mode("dark")


# Cargar modelo SVM

with open("modelo_3bits.pkl", "rb") as archivo:
    datos_modelo = pickle.load(archivo)

modelo = datos_modelo["modelo"]





# Historial para estabilizar las predicciones

historial = deque(maxlen=9)

resultado_estable = None


# Crear mascara de la mano

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


# Normalizar la silueta

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

    area = cv2.contourArea(
        contorno
    )

    if area < 4000:
        return None

    x, y, w, h = cv2.boundingRect(
        contorno
    )

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

    inicio_x = (
        lado - w
    ) // 2

    inicio_y = (
        lado - h
    ) // 2

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


# Estabilizar reconocimiento

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


# Ventana principal

app = ctk.CTk()

app.title(
    "Traductor Binario 3 Bits"
)

app.geometry(
    "1180x720"
)

app.minsize(
    1100,
    680
)

app.configure(
    fg_color=FONDO
)

app.grid_columnconfigure(
    0,
    weight=3
)

app.grid_columnconfigure(
    1,
    weight=2
)

app.grid_rowconfigure(
    0,
    weight=1
)


# Zona izquierda

zona_camara = ctk.CTkFrame(
    app,
    fg_color=FONDO,
    corner_radius=0
)

zona_camara.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(20, 10),
    pady=20
)

zona_camara.grid_columnconfigure(
    0,
    weight=1
)

zona_camara.grid_rowconfigure(
    2,
    weight=1
)


titulo_camara = ctk.CTkLabel(
    zona_camara,
    text="TRADUCTOR BINARIO",
    font=ctk.CTkFont(
        family="Segoe UI",
        size=13,
        weight="bold"
    ),
    text_color=DORADO
)

titulo_camara.grid(
    row=0,
    column=0,
    sticky="w",
    padx=5,
    pady=(0, 3)
)


subtitulo_camara = ctk.CTkLabel(
    zona_camara,
    text=(
        "Reconocimiento de numeros "
        "binarios de 3 bits"
    ),
    font=ctk.CTkFont(
        family="Segoe UI",
        size=12
    ),
    text_color=GRIS
)

subtitulo_camara.grid(
    row=1,
    column=0,
    sticky="w",
    padx=5,
    pady=(0, 12)
)


marco_camara = ctk.CTkFrame(
    zona_camara,
    fg_color=TARJETA_2,
    border_width=1,
    border_color=BORDE,
    corner_radius=14
)

marco_camara.grid(
    row=2,
    column=0,
    sticky="nsew"
)

marco_camara.grid_rowconfigure(
    0,
    weight=1
)

marco_camara.grid_columnconfigure(
    0,
    weight=1
)


label_camara = ctk.CTkLabel(
    marco_camara,
    text=""
)

label_camara.grid(
    row=0,
    column=0,
    padx=10,
    pady=10
)


# Panel derecho

panel = ctk.CTkFrame(
    app,
    fg_color=FONDO_2,
    corner_radius=14,
    border_width=1,
    border_color=BORDE
)

panel.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(10, 20),
    pady=20
)

panel.grid_columnconfigure(
    0,
    weight=1
)


# Encabezado

encabezado = ctk.CTkFrame(
    panel,
    fg_color="transparent"
)

encabezado.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=22,
    pady=(20, 5)
)

encabezado.grid_columnconfigure(
    1,
    weight=1
)


icono = ctk.CTkLabel(
    encabezado,
    text="010",
    width=52,
    height=42,
    corner_radius=8,
    fg_color="#171C1F",
    text_color=DORADO_CLARO,
    font=ctk.CTkFont(
        family="Consolas",
        size=13,
        weight="bold"
    )
)

icono.grid(
    row=0,
    column=0,
    rowspan=2,
    padx=(0, 14)
)


titulo = ctk.CTkLabel(
    encabezado,
    text="TRADUCTOR BINARIO",
    anchor="w",
    font=ctk.CTkFont(
        family="Segoe UI",
        size=25,
        weight="bold"
    ),
    text_color=BLANCO
)

titulo.grid(
    row=0,
    column=1,
    sticky="w"
)


subtitulo = ctk.CTkLabel(
    encabezado,
    text="V I S I O N   C O M P U T A C I O N A L",
    anchor="w",
    font=ctk.CTkFont(
        family="Segoe UI",
        size=8
    ),
    text_color=GRIS
)

subtitulo.grid(
    row=1,
    column=1,
    sticky="w"
)


# Estado general

tarjeta_estado = ctk.CTkFrame(
    panel,
    fg_color=TARJETA,
    corner_radius=12,
    border_width=1,
    border_color="#4B3C23"
)

tarjeta_estado.grid(
    row=1,
    column=0,
    sticky="ew",
    padx=20,
    pady=(15, 10)
)

tarjeta_estado.grid_columnconfigure(
    1,
    weight=1
)


indicador_general = ctk.CTkLabel(
    tarjeta_estado,
    text="●",
    font=ctk.CTkFont(
        size=25
    ),
    text_color=DORADO
)

indicador_general.grid(
    row=0,
    column=0,
    rowspan=2,
    padx=(20, 12),
    pady=14
)


texto_estado = ctk.CTkLabel(
    tarjeta_estado,
    text="BUSCANDO MANO...",
    anchor="w",
    font=ctk.CTkFont(
        size=14,
        weight="bold"
    ),
    text_color=DORADO_CLARO
)

texto_estado.grid(
    row=0,
    column=1,
    sticky="sw",
    pady=(12, 0)
)


texto_estado_secundario = ctk.CTkLabel(
    tarjeta_estado,
    text="Coloca tu mano dentro del recuadro",
    anchor="w",
    font=ctk.CTkFont(
        size=10
    ),
    text_color=GRIS
)

texto_estado_secundario.grid(
    row=1,
    column=1,
    sticky="nw",
    pady=(0, 12)
)


# Tabla de bits

contenedor_bits = ctk.CTkFrame(
    panel,
    fg_color=TARJETA,
    corner_radius=12,
    border_width=1,
    border_color=BORDE
)

contenedor_bits.grid(
    row=2,
    column=0,
    sticky="ew",
    padx=20
)

contenedor_bits.grid_columnconfigure(
    0,
    weight=1
)


cabecera = ctk.CTkFrame(
    contenedor_bits,
    fg_color="transparent"
)

cabecera.grid(
    row=0,
    column=0,
    sticky="ew",
    padx=12,
    pady=(10, 4)
)

cabecera.grid_columnconfigure(
    0,
    weight=2
)

cabecera.grid_columnconfigure(
    1,
    weight=2
)

cabecera.grid_columnconfigure(
    2,
    weight=1
)


ctk.CTkLabel(
    cabecera,
    text="DEDO",
    text_color=GRIS,
    font=ctk.CTkFont(
        size=9,
        weight="bold"
    )
).grid(
    row=0,
    column=0,
    sticky="w",
    padx=8
)


ctk.CTkLabel(
    cabecera,
    text="ESTADO",
    text_color=GRIS,
    font=ctk.CTkFont(
        size=9,
        weight="bold"
    )
).grid(
    row=0,
    column=1,
    sticky="w"
)


ctk.CTkLabel(
    cabecera,
    text="BIT",
    text_color=GRIS,
    font=ctk.CTkFont(
        size=9,
        weight="bold"
    )
).grid(
    row=0,
    column=2,
    sticky="e",
    padx=8
)


nombres_bits = [
    "MEDIO",
    "INDICE",
    "PULGAR"
]

pesos_bits = [
    "4",
    "2",
    "1"
]

filas_bits = []


for i, nombre in enumerate(nombres_bits):

    fila = ctk.CTkFrame(
        contenedor_bits,
        height=58,
        fg_color=TARJETA_2,
        corner_radius=9,
        border_width=1,
        border_color="#20272C"
    )

    fila.grid(
        row=i + 1,
        column=0,
        sticky="ew",
        padx=10,
        pady=5
    )

    fila.grid_columnconfigure(
        0,
        weight=2
    )

    fila.grid_columnconfigure(
        1,
        weight=2
    )

    fila.grid_columnconfigure(
        2,
        weight=1
    )


    nombre_frame = ctk.CTkFrame(
        fila,
        fg_color="transparent"
    )

    nombre_frame.grid(
        row=0,
        column=0,
        sticky="w",
        padx=15,
        pady=10
    )


    ctk.CTkLabel(
        nombre_frame,
        text=nombre,
        font=ctk.CTkFont(
            size=11,
            weight="bold"
        ),
        text_color=BLANCO
    ).pack(
        anchor="w"
    )


    ctk.CTkLabel(
        nombre_frame,
        text=f"PESO BINARIO: {pesos_bits[i]}",
        font=ctk.CTkFont(
            size=8
        ),
        text_color=GRIS
    ).pack(
        anchor="w"
    )


    estado_frame = ctk.CTkFrame(
        fila,
        fg_color="transparent"
    )

    estado_frame.grid(
        row=0,
        column=1,
        sticky="w"
    )


    punto = ctk.CTkLabel(
        estado_frame,
        text="●",
        width=20,
        font=ctk.CTkFont(
            size=20
        ),
        text_color=ROJO
    )

    punto.pack(
        side="left"
    )


    estado_texto = ctk.CTkLabel(
        estado_frame,
        text="CERRADO",
        font=ctk.CTkFont(
            size=8
        ),
        text_color=GRIS
    )

    estado_texto.pack(
        side="left",
        padx=(4, 0)
    )


    valor = ctk.CTkLabel(
        fila,
        text="0",
        width=48,
        height=32,
        corner_radius=7,
        fg_color="#191E22",
        text_color=BLANCO,
        font=ctk.CTkFont(
            family="Consolas",
            size=18,
            weight="bold"
        )
    )

    valor.grid(
        row=0,
        column=2,
        padx=(5, 12),
        pady=7,
        sticky="e"
    )


    filas_bits.append({
        "punto": punto,
        "estado": estado_texto,
        "valor": valor
    })


# Resultados

zona_resultados = ctk.CTkFrame(
    panel,
    fg_color="transparent"
)

zona_resultados.grid(
    row=3,
    column=0,
    sticky="ew",
    padx=20,
    pady=(15, 10)
)

zona_resultados.grid_columnconfigure(
    0,
    weight=3
)

zona_resultados.grid_columnconfigure(
    1,
    weight=2
)


tarjeta_binario = ctk.CTkFrame(
    zona_resultados,
    fg_color=TARJETA,
    corner_radius=12,
    border_width=1,
    border_color="#4B3C23"
)

tarjeta_binario.grid(
    row=0,
    column=0,
    sticky="nsew",
    padx=(0, 6)
)


ctk.CTkLabel(
    tarjeta_binario,
    text="BINARIO",
    text_color=GRIS_CLARO,
    font=ctk.CTkFont(
        size=11,
        weight="bold"
    )
).pack(
    fill="x",
    padx=14,
    pady=(12, 4)
)


valor_binario = ctk.CTkLabel(
    tarjeta_binario,
    text="0 0 0",
    height=65,
    corner_radius=10,
    fg_color="#0A0D0F",
    text_color=DORADO_CLARO,
    font=ctk.CTkFont(
        family="Consolas",
        size=30,
        weight="bold"
    )
)

valor_binario.pack(
    fill="x",
    padx=12,
    pady=(4, 12)
)


tarjeta_decimal = ctk.CTkFrame(
    zona_resultados,
    fg_color=TARJETA,
    corner_radius=12,
    border_width=1,
    border_color="#21414A"
)

tarjeta_decimal.grid(
    row=0,
    column=1,
    sticky="nsew",
    padx=(6, 0)
)


ctk.CTkLabel(
    tarjeta_decimal,
    text="DECIMAL",
    text_color=GRIS_CLARO,
    font=ctk.CTkFont(
        size=11,
        weight="bold"
    )
).pack(
    fill="x",
    padx=14,
    pady=(12, 4)
)


valor_decimal = ctk.CTkLabel(
    tarjeta_decimal,
    text="0",
    height=65,
    corner_radius=10,
    fg_color="#0A0D0F",
    text_color=AZUL,
    font=ctk.CTkFont(
        size=32,
        weight="bold"
    )
)

valor_decimal.pack(
    fill="x",
    padx=12,
    pady=(4, 12)
)


# Informacion inferior

info = ctk.CTkLabel(
    panel,
    text="PIXeles  •  SVM  •  OpenCV  •  3 BITS",
    font=ctk.CTkFont(
        size=9
    ),
    text_color=GRIS
)

info.grid(
    row=4,
    column=0,
    pady=(2, 15)
)


# Abrir camara

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

    print(
        "No se pudo abrir la camara"
    )


# Procesamiento principal

def actualizar():

    global resultado_estable

    correcto, frame = camara.read()

    if not correcto:

        app.after(
            30,
            actualizar
        )

        return

    frame = cv2.flip(
        frame,
        1
    )

    alto, ancho, _ = frame.shape

    x1 = int(
        ancho * 0.25
    )

    y1 = int(
        alto * 0.10
    )

    x2 = int(
        ancho * 0.75
    )

    y2 = int(
        alto * 0.90
    )


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


    if mano is not None:

        caracteristicas = extraer_caracteristicas(
            mano
        )

        caracteristicas = (
            caracteristicas.reshape(
                1,
                -1
            )
        )

        prediccion = modelo.predict(
            caracteristicas
        )[0]

        historial.append(
            prediccion
        )

        estable = estabilizar()

        if estable is not None:
            resultado_estable = estable


        texto_estado.configure(
            text="MANO LISTA",
            text_color=VERDE
        )

        texto_estado_secundario.configure(
            text=(
                "Reconocimiento activo "
                "de 3 bits"
            )
        )

        indicador_general.configure(
            text_color=VERDE
        )

        tarjeta_estado.configure(
            border_color=VERDE_OSCURO
        )

        color_recuadro = (
            69,
            185,
            120
        )

    else:

        historial.clear()

        resultado_estable = None

        texto_estado.configure(
            text="BUSCANDO MANO...",
            text_color=DORADO_CLARO
        )

        texto_estado_secundario.configure(
            text=(
                "Coloca tu mano "
                "dentro del recuadro"
            )
        )

        indicador_general.configure(
            text_color=DORADO
        )

        tarjeta_estado.configure(
            border_color="#4B3C23"
        )

        color_recuadro = (
            69,
            154,
            199
        )


    if resultado_estable is not None:

        bits = [
            int(bit)
            for bit in resultado_estable
        ]

        decimal = int(
            resultado_estable,
            2
        )

    else:

        bits = [
            0,
            0,
            0
        ]

        decimal = 0


    # Actualizar dedos

    for i, bit in enumerate(bits):

        if bit == 1:

            filas_bits[i][
                "punto"
            ].configure(
                text_color=VERDE
            )

            filas_bits[i][
                "estado"
            ].configure(
                text="EXTENDIDO",
                text_color=VERDE
            )

            filas_bits[i][
                "valor"
            ].configure(
                text="1",
                text_color=VERDE
            )

        else:

            filas_bits[i][
                "punto"
            ].configure(
                text_color=ROJO
            )

            filas_bits[i][
                "estado"
            ].configure(
                text="CERRADO",
                text_color=GRIS
            )

            filas_bits[i][
                "valor"
            ].configure(
                text="0",
                text_color=BLANCO
            )


    # Actualizar resultados

    valor_binario.configure(
        text=" ".join(
            str(bit)
            for bit in bits
        )
    )

    valor_decimal.configure(
        text=str(decimal)
    )


    # Dibujar zona de reconocimiento

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color_recuadro,
        2
    )


    cv2.putText(
        frame,
        "ZONA DE RECONOCIMIENTO",
        (x1, y1 - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color_recuadro,
        1
    )


    # Mostrar resultado sobre la camara

    if resultado_estable is not None:

        cv2.putText(
            frame,
            (
                f"{resultado_estable}  =  "
                f"{decimal}"
            ),
            (x1 + 10, y2 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color_recuadro,
            2
        )


    # Convertir imagen para CustomTkinter

    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    alto_img, ancho_img, _ = (
        frame_rgb.shape
    )

    max_ancho = 650
    max_alto = 570

    escala = min(
        max_ancho / ancho_img,
        max_alto / alto_img
    )

    nuevo_ancho = int(
        ancho_img * escala
    )

    nuevo_alto = int(
        alto_img * escala
    )

    frame_rgb = cv2.resize(
        frame_rgb,
        (
            nuevo_ancho,
            nuevo_alto
        )
    )

    imagen = Image.fromarray(
        frame_rgb
    )

    imagen_ctk = ctk.CTkImage(
        light_image=imagen,
        dark_image=imagen,
        size=(
            nuevo_ancho,
            nuevo_alto
        )
    )

    label_camara.configure(
        image=imagen_ctk
    )

    label_camara.image = (
        imagen_ctk
    )

    app.after(
        30,
        actualizar
    )


# Cerrar correctamente

def cerrar():

    camara.release()

    cv2.destroyAllWindows()

    app.destroy()


app.protocol(
    "WM_DELETE_WINDOW",
    cerrar
)


# Iniciar

actualizar()

app.mainloop()