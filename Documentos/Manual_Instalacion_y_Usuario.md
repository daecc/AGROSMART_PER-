# MANUAL DE INSTALACIÓN Y GUÍA DE USUARIO
## AGROSMART PERÚ — SOFTWARE INTELIGENTE
### Versión: 1.0 (Entregable Avance 50%)

---

# PARTE I: MANUAL DE INSTALACIÓN

Este software ha sido desarrollado en lenguaje **Python (versión 3.10 o superior)** y utiliza la librería **Streamlit** para proveer una interfaz gráfica moderna, reactiva y multiplataforma (compatible con Windows, Linux y macOS).

### 1. Requisitos Previos del Sistema
* **Sistema Operativo:** Windows 10/11, Ubuntu 20.04+, o macOS.
* **Procesador:** Intel Core i3 / AMD Ryzen 3 o superior (ejecutable 100% en CPU).
* **Memoria RAM:** Mínimo 4 GB (Recomendado 8 GB).
* **Python:** Versión 3.10, 3.11 o 3.12 instalada con `pip`.

---

### 2. Pasos de Instalación

#### Paso 1: Clonar o descomprimir el proyecto
Abra una terminal (PowerShell o CMD en Windows) y sitúese en la carpeta raíz del proyecto:
```bash
cd C:\Users\camil\Desktop\Proyecto_IA
```

#### Paso 2: Verificar la versión de Python
Compruebe que Python y pip estén disponibles:
```bash
python --version
pip --version
```

#### Paso 3: Instalar dependencias
Ejecute la instalación de los paquetes requeridos listados en `requirements.txt`:
```bash
pip install -r requirements.txt
```
*Las librerías principales instaladas son: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn` y `pillow`.*

#### Paso 4: Cuadernos de Entrenamiento y Experimentación (Jupyter / Colab)
Si desea revisar el proceso de entrenamiento paso a paso con código visible, métricas formales y gráficos, cada módulo cuenta con su propio cuaderno en la carpeta `notebooks/`:
* [01_Prediccion_Supervisada_Rendimiento.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/01_Prediccion_Supervisada_Rendimiento.ipynb): Random Forest vs Baseline con $R^2$ y MAE.
* [02_Series_Temporales_Pronostico_Precios.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/02_Series_Temporales_Pronostico_Precios.ipynb): Forecasting con ICEN exógeno y MAPE.
* [03_Clasificacion_Imagenes_Patologias_Hojas.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/03_Clasificacion_Imagenes_Patologias_Hojas.ipynb): Clasificación probabilística foliar.
* [04_Clustering_Vulnerabilidad_Provincias.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/04_Clustering_Vulnerabilidad_Provincias.ipynb): K-Means ($k=3$) con Método del Codo y PCA.
* [05_NLP_Resumen_Comunicados_ENFEN.ipynb](file:///c:/Users/camil/Desktop/Proyecto_IA/notebooks/05_NLP_Resumen_Comunicados_ENFEN.ipynb): Resumen extractivo TextRank con TF-IDF.

Para regenerar todos los modelos `.pkl` desde consola sin abrir Jupyter:
```bash
python src/data_generator.py
python src/prediction.py
python src/timeseries.py
python src/clustering.py
python -m src.vision_dataset   # descarga (~91 MB) y limpia las fotos reales de hojas de arroz
python -m src.vision           # entrena y exporta models/vision_hojas_arroz.pkl
```
*(Los modelos quedarán automáticamente serializados en la carpeta `models/` y los datos en `data/`).*

#### Paso 5: Ejecución de la Interfaz Gráfica
Inicie la aplicación web interactiva con el comando de Streamlit:
```bash
streamlit run app.py
```

La aplicación se abrirá de forma automática en su navegador web predeterminado en la dirección local:
```
http://localhost:8501
```

---

# PARTE II: GUÍA DE USUARIO (MANUAL DE OPERACIÓN)

La interfaz de **AgroSmart Perú** cuenta con un menú de navegación en la barra lateral izquierda que permite alternar entre el panel general y los 5 componentes de IA.

---

### 1. Panel General y Diagnóstico El Niño
* **Uso:** Presenta la situación climática actual según el informe más reciente del ENFEN.
* **Interacción:**
  1. Observe las métricas principales superiores (Índice ICEN, anomalía de lluvias y porcentaje de pérdida proyectada).
  2. En el menú desplegable *"Seleccione Cultivo para Exploración"*, elija entre **Limón** o **Arroz**.
  3. Visualice la gráfica histórica (2007–2026), que muestra cómo las líneas de temperatura oceánica roja (*ICEN*) provocan caídas drásticas en las curvas verdes de rendimiento agrícola en los años Niño 2017, 2023 y 2026.

---

### 2. Módulo 1: Predicción de Rendimiento Agrícola (Supervisado)
* **Objetivo:** Calcular el rendimiento estimado ($t/ha$) y la pérdida frente al promedio histórico.
* **Pasos para operar:**
  1. En el panel izquierdo, seleccione el **Cultivo a Evaluar** (Limón o Arroz).
  2. Ajuste los controles deslizantes (*sliders*):
     - **Índice ICEN:** Valores superiores a +2.0 °C simulan un Niño Fuerte.
     - **Anomalía TSM Niño 1+2:** Grados centígrados de calentamiento marino.
     - **Precipitación Mensual:** Milímetros de lluvia acumulada esperada en el valle.
     - **Temperatura Máxima:** Temperatura ambiental en °C.
  3. Haga clic en el botón azul **"🚀 Ejecutar Predicción de Rendimiento"**.
  4. En el panel derecho aparecerá la tarjeta de resultado con el rendimiento proyectado, la pérdida porcentual y el nivel de semáforo de riesgo (Verde, Amarillo o Rojo), junto al gráfico de importancia de factores agronómicos.

---

### 3. Módulo 2: Series Temporales de Precios en Chacra (Forecasting)
* **Objetivo:** Proyectar la tendencia de precios en chacra a futuro bajo diferentes escenarios de El Niño.
* **Pasos para operar:**
  1. Seleccione el producto (**Limón** o **Arroz**).
  2. Indique el **Horizonte de Pronóstico** (de 3 a 12 meses).
  3. Elija el **Escenario Climático Futuro** (Extraordinario, Fuerte, Moderado o Normal).
  4. Presione **"📊 Proyectar Precios Futuros"**.
  5. El sistema graficará la serie de precios observados en los últimos meses unida a la proyección futura y desplegará la tabla de valores mensuales esperados.

---

### 4. Módulo 3: Diagnóstico Fitosanitario de Hojas (Visión Artificial)
* **Objetivo:** Clasificar patologías en hojas de arroz aceleradas por la humedad y altas temperaturas.
* **Alcance:** el modelo fue entrenado con fotografías de hojas de **arroz** (en campo o sobre fondo blanco). No es válido para limón. El tizón bacteriano solo aparece en fotos de campo, así que para esa enfermedad se recomienda fotografiar la planta en el campo.
* **Pasos para operar:**
  1. Seleccione el método de entrada:
     - **"Usar foto de muestra (conjunto de prueba)":** Elija una de las 6 fotos reales de `data/sample_leaves/` (`hoja_sana_1.jpg`, `tizon_bacteriano_1.jpg`, `mancha_marron_1.jpg`, etc.), que el modelo no vio durante el entrenamiento.
     - **"Subir mi propia foto":** Permite cargar una foto JPG o PNG del cultivo de arroz tomada con cámara o celular.
  2. De manera instantánea, el sistema desplegará:
     - El diagnóstico patológico identificado con su porcentaje de certeza.
     - El nivel de severidad y descripción del patógeno.
     - La recomendación de control fitosanitario aprobada por SENASA.
     - Gráfico de barras de distribución probabilística entre clases.

---

### 5. Módulo 4: Clustering de Vulnerabilidad Provincial (K-Means)
* **Objetivo:** Segmentar las provincias de Piura para focalización de recursos de emergencia.
* **Pasos para operar:**
  1. Ingrese a la pestaña de Clustering.
  2. Observe el plano de dispersión PCA con los 3 grupos de riesgo:
     - **Rojo (Prioridad 1 - Vulnerabilidad Extrema):** Morropón, Sechura, Piura y Sullana.
     - **Naranja (Prioridad 2 - Vulnerabilidad Alta):** Ayabaca y Huancabamba.
     - **Verde (Prioridad 3 - Vulnerabilidad Moderada):** Paita y Talara.
  3. Revise la tabla detallada inferior para consultar el cultivo predominante, hectáreas inundables y porcentaje histórico de daño.

---

### 6. Módulo 5: Resumen Inteligente de Alertas ENFEN / SENAMHI (NLP)
* **Objetivo:** Extraer las conclusiones críticas de comunicados oficiales técnicos.
* **Pasos para operar:**
  1. Seleccione un documento oficial precargado del selector (o pegue directamente el texto de un nuevo comunicado en el área de texto).
  2. Seleccione con el deslizador la cantidad de **Conclusiones Clave** deseadas (de 2 a 6 oraciones).
  3. Haga clic en **"📝 Generar Ficha Ejecutiva"**.
  4. El algoritmo procesará la redacción con TextRank y mostrará las conclusiones prioritarias en viñetas claras junto con el porcentaje de compresión de texto y las palabras clave identificadas.

---

### 7. Detener la Aplicación
Para cerrar el servidor de la aplicación, regrese a la terminal donde ejecutó `streamlit run app.py` y presione la combinación de teclas:
```bash
Ctrl + C
```
