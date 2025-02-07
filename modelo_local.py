import os

# Desactivar el file watcher de Streamlit para evitar inspección problemática
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Forzar que LD_LIBRARY_PATH incluya el directorio habitual de libGL.so.1
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + ld_path

# Forzar que HOME y XDG_CACHE_HOME apunten a directorios en /tmp (escrituibles en Streamlit Cloud)
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache"

# Usar un directorio de caché en /tmp que se sepa que es escribible
cache_dir = "/tmp/BasicSR_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["BASICSR_CACHE_DIR"] = cache_dir

# Sobreescribir os.path.expanduser para que "~" se convierta en "/tmp"
os.path.expanduser = lambda path: path.replace("~", "/tmp")

import sys
import torchvision.transforms.functional as F
sys.modules["torchvision.transforms.functional_tensor"] = F

import streamlit as st
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Establecer dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4
    )
    model = model.to(device)
    upsampler = RealESRGANer(
        scale=4,
        model_path="models/RealESRGAN_x4plus.pth",  # Usar el archivo local
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False
    )
    return upsampler

upsampler = load_model()

st.title("Superresolución de Imágenes con Real-ESRGAN by Lucas De Rito")

# Subir imagen
uploaded_file = st.file_uploader(
    "Sube una imagen de baja resolución o en mal estado",
    type=["jpg", "png", "jpeg", "webp"]
)

if uploaded_file is not None:
    # Leer imagen
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Redimensionar la imagen si es demasiado grande (por ejemplo, ancho mayor a 1024 píxeles)
    max_width = 1024
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        st.info(f"La imagen fue redimensionada a {new_width}x{new_height} para optimizar el procesamiento.")

    # Procesar la imagen con superresolución (capturamos errores para evitar que la app se caiga)
    try:
        with st.spinner("Mejorando la imagen (puede tardar en CPU)..."):
            output, _ = upsampler.enhance(image, outscale=4)
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")
    else:
        # Mostrar resultados en dos columnas
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagen Original (Redimensionada, si aplicó)", use_container_width=True)
        with col2:
            st.image(output, caption="Imagen Mejorada (Superresolución)", use_container_width=True)

# Línea separatoria
st.markdown('---')

st.markdown("""
### Aclaración sobre el rendimiento y despliegue

Esta aplicación está diseñada para aprovechar la aceleración por **GPU** cuando esté disponible, lo que permite que los modelos de superresolución se ejecuten de forma mucho más rápida y eficiente. Sin embargo, en el entorno de **Streamlit Cloud (versión gratuita)**, generalmente no se dispone de GPU, por lo que la aplicación se desplegará utilizando **CPU**. Esto puede resultar en tiempos de procesamiento más largos, por lo que se recomienda usar imágenes de tamaño moderado para obtener resultados óptimos.

### Si quieren descargar el código, lo pueden encontrar en el siguiente repositorio de GitHub:  
[🔗 **Superresolución - GitHub**](https://github.com/lucasderito/superresolucion)

## Descripción de la Aplicación

Esta aplicación utiliza un modelo preentrenado de **Real-ESRGAN** para mejorar la resolución de imágenes de baja calidad. Con esta herramienta, puedes transformar imágenes borrosas o de baja resolución en versiones más nítidas y detalladas.

### Características y Técnicas Utilizadas
- **Superresolución con Real-ESRGAN:**  
  Utiliza redes neuronales profundas para aumentar la resolución de la imagen, recuperando detalles que no eran visibles en la versión original.
- **Optimización para CPU:**  
  Adaptada para entornos sin GPU, lo que permite su despliegue en plataformas de nube como Streamlit Cloud.

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

