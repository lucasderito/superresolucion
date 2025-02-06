import os

# Define un directorio de caché temporal y asegúrate de que existe.
cache_dir = "/tmp/.cache/BasicSR"
os.makedirs(cache_dir, exist_ok=True)

# Fuerza a basicsr (y otras librerías) a usar este directorio para la caché.
os.environ["BASICSR_CACHE_DIR"] = cache_dir

# Opcional: también puedes forzar que HOME apunte a /tmp para que expanduser("~") dé un directorio escribible.
os.environ["HOME"] = "/tmp"

import sys
import torchvision.transforms.functional as F
sys.modules["torchvision.transforms.functional_tensor"] = F

import streamlit as st
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer



# Establecer dispositivo: usa GPU si está disponible, de lo contrario CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo Real-ESRGAN para GPU (o CPU si no hay GPU)
@st.cache_resource
def load_model():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    # Mover el modelo al dispositivo (GPU o CPU)
    model = model.to(device)
    # Configurar half=True si se usa GPU, lo que acelera el procesamiento
    half_setting = True if torch.cuda.is_available() else False
    upsampler = RealESRGANer(
        scale=4,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=model,
        tile=400,  # Divide la imagen en bloques para optimizar el uso de memoria
        tile_pad=10,
        pre_pad=0,
        half=half_setting
    )
    return upsampler

upsampler = load_model()

st.title("Superresolución de Imágenes con Real-ESRGAN by Lucas De Rito (GPU)")

# Subir imagen
uploaded_file = st.file_uploader("Sube una imagen de baja resolución o en mal estado", type=["jpg", "png", "jpeg", "webp"])

if uploaded_file is not None:
    # Leer imagen
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Aplicar superresolución
    with st.spinner("Mejorando la imagen (usando GPU)..."):
        output, _ = upsampler.enhance(image, outscale=4)

    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Imagen Original (Baja Resolución)", use_container_width=True)
    with col2:
        st.image(output, caption="Imagen Mejorada (Superresolución)", use_container_width=True)

# Línea separatoria
st.markdown('---')

st.markdown("""
### Aclaración sobre el rendimiento y despliegue

Esta aplicación está diseñada para aprovechar la aceleración por **GPU** cuando esté disponible, lo que permite que los modelos de superresolución se ejecuten de forma mucho más rápida y eficiente. Sin embargo, en el entorno de **Streamlit Cloud (versión gratuita)**, generalmente no se dispone de GPU, por lo que la aplicación se desplegará utilizando **CPU**. Esto puede resultar en tiempos de procesamiento más largos, pero la aplicación se adaptará automáticamente al entorno y seguirá funcionando correctamente, ofreciendo una mejora significativa en la calidad de las imágenes. Se recomienda usar imagenes no muy grandes debido a que es una version desplegada en una nube gratuita que cuenta con poca infraestructura.

### Si quieren descargar el código, lo pueden encontrar en el siguiente repositorio de GitHub:  
[🔗 **Superresolución - GitHub**](https://github.com/lucasderito/superresolucion)

### Descripción de la Aplicación

Esta aplicación utiliza un modelo preentrenado de **Real-ESRGAN** para mejorar la resolución de imágenes de baja calidad. Con esta herramienta, puedes transformar imágenes borrosas o de baja resolución en versiones más nítidas y detalladas.

### Características y Técnicas Utilizadas
- **Superresolución con Real-ESRGAN:**  
  Utiliza redes neuronales profundas para aumentar la resolución de la imagen, recuperando detalles que no eran visibles en la versión original.
- **Optimización para GPU:**  
  Configurada para aprovechar la aceleración por hardware, reduciendo significativamente los tiempos de procesamiento.

### Tipos de Imágenes Sugeridas
- **Fotografías antiguas o deterioradas:**  
  Ideal para restaurar imágenes históricas o fotografías con calidad limitada.
- **Imágenes escaneadas de documentos y fotos:**  
  Mejora la claridad de imágenes provenientes de escaneos o dispositivos de baja resolución.
- **Capturas de dispositivos móviles:**  
  Perfecto para mejorar imágenes tomadas en condiciones de poca iluminación o con cámaras de menor calidad.

### Casos de Uso
#### Restauración y Mejora de Fotografías
- **Restauración de imágenes históricas:**  
  Recupera detalles en fotos antiguas, facilitando su preservación y análisis.
- **Mejora de selfies y fotos de eventos:**  
  Convierte imágenes capturadas en entornos con iluminación deficiente en imágenes más nítidas.

#### Aplicaciones Comerciales e Industriales
- **Optimización de imágenes para e-commerce:**  
  Aumenta la calidad visual de fotos de productos, mejorando la experiencia del cliente.
- **Análisis en investigaciones:**  
  Proporciona imágenes de alta calidad para estudios científicos o industriales donde el detalle es crucial.

Esta aplicación es parte del portafolio de proyectos de **Lucas De Rito**, demostrando habilidades en inteligencia artificial, visión por computadora y procesamiento de imágenes.

*Desarrollada con Streamlit, OpenCV, Real-ESRGAN y PyTorch.*
""")
