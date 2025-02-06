"# superresolucion" 

Aplicación de Superresolución de Imágenes con Real-ESRGAN

Esta aplicación interactiva permite mejorar la calidad de imágenes de baja resolución utilizando Real-ESRGAN, un modelo basado en redes neuronales profundas. Implementada en Streamlit, permite transformar imágenes borrosas o deterioradas en versiones más nítidas y detalladas.

Características principales

Real-ESRGAN: Modelo de superresolución basado en IA que recupera detalles invisibles en la imagen original.

Soporte para CPU y GPU: Se incluyen versiones optimizadas para distintos entornos.

Ejecución local con modelo descargado: Posibilidad de utilizar el modelo predescargado sin necesidad de acceder a la nube.

Mejora de imágenes históricas, escaneadas y capturas móviles: Ideal para restauración fotográfica, optimización en e-commerce y análisis en investigaciones.

Instalación y ejecución

1. Clonar el repositorio

git clone https://github.com/lucasderito/superresolucion.git
cd superresolucion

2. Crear un entorno virtual e instalar dependencias

python -m venv venv
source venv/bin/activate  # En Windows usar: venv\Scripts\activate
pip install -r requirements.txt

3. Ejecutar la aplicación

Para la versión en CPU:

streamlit run superresolucion.py

Para la versión en GPU (requiere CUDA):

streamlit run superresolucion_gpu.py

Para la versión con el modelo local descargado:

streamlit run superresolucion_modelo_descargado.py

Notas sobre los archivos adicionales

requirements.txt: Contiene todas las dependencias necesarias para ejecutar la aplicación.

apt.txt y runtime.txt: Solo requeridos en ciertos entornos de despliegue. No son necesarios para ejecución en CPU o GPU.

Carpeta model/: Contiene el modelo descargado para ejecución local. Si utilizas las versiones de CPU o GPU que cargan el modelo en tiempo de ejecución, no necesitas esta carpeta.

Notas sobre la caché y permisos en algunos entornos

En ciertos entornos de despliegue restringidos, puede ser necesario definir un directorio de caché temporal y ajustar permisos para garantizar que las librerías como BasicSR funcionen correctamente. En estos casos, el siguiente código puede ser útil:

import os

Define un directorio de caché temporal y asegúrate de que existe.

cache_dir = "/tmp/.cache/BasicSR"
os.makedirs(cache_dir, exist_ok=True)

Fuerza a basicsr (y otras librerías) a usar este directorio para la caché.

os.environ["BASICSR_CACHE_DIR"] = cache_dir

Opcional: también puedes forzar que HOME apunte a /tmp para que expanduser("~") dé un directorio escribible.

os.environ["HOME"] = "/tmp"

Este ajuste no es necesario para ejecuciones en CPU y GPU locales, donde el usuario tiene permisos completos.

Notas sobre el rendimiento

Esta es una versión demo desplegada en Streamlit Cloud, con limitaciones de procesamiento por la ausencia de GPU y restricciones de memoria. Para un funcionamiento óptimo, se recomienda utilizar imágenes pequeñas y de baja complejidad. Imágenes grandes o con demasiados detalles pueden exceder los recursos disponibles y generar errores.

Para un uso profesional o en producción, se recomienda una infraestructura adecuada con soporte para procesamiento acelerado.

Contacto

Para cualquier consulta o mejora, puedes contactarme a través de LinkedIn o a lucasderito@issd.edu.ar

Este proyecto forma parte de mi portafolio en inteligencia artificial y visión por computadora, donde combino modelos de deep learning con interfaces interactivas para resolver problemas del mundo real.
