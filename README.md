# 🌾 AgroSmart Perú: Sistema Inteligente ante El Niño Costero

> **Plataforma Integral de Inteligencia Artificial para la Mitigación del Impacto del Fenómeno El Niño Costero en la Agricultura de la Costa Norte (Piura)**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Estado-Avance%2050%25%20Completado-green.svg)](#)

---

## 📑 Tabla de Contenidos
1. [Problemática Real en el Perú](#-1-problemática-real-en-el-perú)
2. [Arquitectura General del Sistema](#-2-arquitectura-general-del-sistema)
3. [Explicación Paso a Paso de los Datos (Pipeline de Ingeniería)](#-3-explicación-paso-a-paso-de-los-datos-pipeline-de-ingeniería)
4. [Explicación Paso a Paso de los 5 Módulos de IA](#-4-explicación-paso-a-paso-de-los-5-módulos-de-ia)
5. [Flujo de Integración: Del Notebook a la App](#-5-flujo-de-integración-del-notebook-a-la-app)
6. [Estructura del Repositorio](#-6-estructura-del-repositorio)
7. [Guía de Instalación y Ejecución Paso a Paso](#-7-guía-de-instalación-y-ejecución-paso-a-paso)
8. [Población Beneficiaria y Ventajas](#-8-población-beneficiaria-y-ventajas)

---

## 🌊 1. Problemática Real en el Perú

El Fenómeno de **El Niño Costero (2026–2027)**, declarado por la Comisión Multisectorial del **ENFEN** con magnitud extraordinaria, representa la amenaza más destructiva para la seguridad alimentaria y la economía agraria de la costa norte peruana.

### ¿Por qué Piura y por qué Limón y Arroz?
* **Zona de Impacto Crítico:** Frente al litoral de Piura y Tumbes se ubica la región oceanográfica **Niño 1+2**, donde la temperatura del mar supera anomalías de $+2.6^\circ\text{C}$ a $+4.0^\circ\text{C}$, desatando lluvias extremas (400 a 700 mm).
* **Limón Sutil (*Citrus aurantifolia*):** Piura concentra **más del 55% de toda la producción nacional del Perú** (valles de San Lorenzo, Chulucanas y Tambogrande). Las lluvias anegan las raíces y pudren la floración, provocando caídas de producción históricas (como el $-37.9\%$ en el Niño 2017 registrado por el BCRP y más del $-40\%$ en 2026), disparando los precios al consumidor por encima de S/ 15.00 el kilo.
* **Arroz Cáscara (*Oryza sativa*):** En el Valle del Chira y Bajo Piura, las crecidas del río Piura inundan miles de hectáreas en plena cosecha y desatan plagas agresivas por el calor y la humedad relativa ($>90\%$).
* **Sobrecarga Técnica:** Los agricultores y juntas de regantes reciben boletines del ENFEN y SENAMHI de 20 páginas con jerga oceanográfica que no logran traducir en decisiones preventivas rápidas.

---

## 🏗️ 2. Arquitectura General del Sistema

El software adopta una **Arquitectura Modular en Capas (Layered Architecture)**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN (GUI)                   │
│         Streamlit Responsive Dashboard Multiusuario (app.py)    │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Invocación directa en memoria
┌────────────────────────────────▼────────────────────────────────┐
│             CAPA DE SERVICIOS / CONTROLADORES (src/)            │
│   src/prediction.py   │   src/timeseries.py   │  src/vision.py  │
│   src/clustering.py   │   src/summarizer.py   │  data_generator │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Carga de artefactos serializados
┌────────────────────────────────▼────────────────────────────────┐
│                  CAPA DE MODELOS DE INFERENCIA                  │
│   • Random Forest Regressor (R² = 0.959 en Limón, 0.938 Arroz)  │
│   • Gradient Boosting Time Series Forecaster (MAPE = 8.11%)     │
│   • HistGradientBoosting Visión (fotos reales, Accuracy 0.954)  │
│   • K-Means Clustering + PCA (Silhouette Score = 0.543)         │
│   • TextRank Graph Algorithm + TF-IDF (Compresión 76.4%)        │
└────────────────────────────────┬────────────────────────────────┘
                                 │ Lectura de datos estructurados
┌────────────────────────────────▼────────────────────────────────┐
│                  CAPA DE DATOS LOCAL Y REPRODUCIBLE             │
│   • data/tabla_maestra_agro_nino.csv (474 registros mensuales)  │
│   • data/provincias_piura_riesgo.csv (8 provincias de Piura)    │
│   • data/raw/hojas_arroz/ (1 424 fotos reales de hojas de arroz)│
│   • data/comunicados_enfen/ (Boletines oficiales en texto)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 3. Explicación Paso a Paso de los Datos (Pipeline de Ingeniería)

El proyecto cuenta con **DOS tablas principales** estructuradas de manera transparente.

### TABLA 1: `data/tabla_maestra_agro_nino.csv` (474 Filas, Mes a Mes de 2007 a 2026)
Une cronológicamente mediante las llaves `año` y `mes` los datos oceanográficos internacionales y la producción peruana:

```
┌────────────────────────┐      ┌────────────────────────┐
│  NOAA (EE.UU.)         │      │  BCRP (Perú)           │
│  Archivo: nina1.data   │      │  JSON: PN01788 / 778   │
│  (TSM Niño 1+2)        │      │  (Producción Agrícola) │
└───────────┬────────────┘      └───────────┬────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
              Cruce por (Año, Mes) en Python
                            ▼
┌────────────────────────────────────────────────────────┐
│  TABLA MAESTRA (data/tabla_maestra_agro_nino.csv)      │
│  15 Columnas integradas para Machine Learning          │
└────────────────────────────────────────────────────────┘
```

#### Detalle de las 15 Columnas y su Origen:
1. **`fecha`:** Fecha estandarizada `YYYY-MM-01` para indexación cronológica.
2. **`año` y `mes`:** Identificadores temporales desde Enero de 2007 hasta Septiembre de 2026.
3. **`departamento`:** Ámbito geográfico (`'Piura'`).
4. **`cultivo`:** Especie evaluada (`'Limon'` o `'Arroz'`).
5. **`tsm_nino12` (100% REAL):** Temperatura superficial del mar (°C) descargada de **NOAA Physical Sciences Laboratory** (`psl.noaa.gov/data/correlation/nina1.data`).
6. **`anomalia_tsm` (100% REAL):** Desviación respecto a la climatología histórica oficial 1981–2010.
7. **`icen` (100% REAL):** **Índice Costero El Niño**, calculado bajo la norma del **IGP / ENFEN** como la media móvil trimestral de la anomalía térmica.
8. **`produccion_t` (100% REAL):** Toneladas producidas extraídas de la API oficial del **BCRP** (Series `PN01788AM` para Limón y `PN01778AM` para Arroz). Coincide número por número con el portal estadístico del BCRP.
9. **`lluvia_mensual_mm` (Calibrado con SENAMHI):** Acumulados mensuales que replican el régimen real de Piura: 0 a 10 mm en temporada seca y **300 a 680 mm** durante los picos de El Niño (registros de estaciones Miraflores, Chulucanas y Partidor).
10. **`temp_max_prom` (Calibrado con SENAMHI):** Temperaturas máximas promedio en Piura (30.0 °C a 36.5 °C).
11. **`rendimiento_tha` (Calibrado con MIDAGRI):** Rendimiento en $t/ha$. Parte de los promedios de los anuarios del MIDAGRI (12.5 $t/ha$ para limón y 8.8 $t/ha$ para arroz) y modela la merma física ante el exceso hídrico.
12. **`area_cosechada_ha`:** Calculada como $\frac{\text{producción}}{\text{rendimiento}}$.
13. **`precio_chacra_soles_kg` (Calibrado con SISAP):** Precio en chacra al productor (S/ por kg), reflejando la subida de S/ 3.00 a más de S/ 14.00 en los eventos de escasez severa de El Niño.
14. **`evento_nino`:** Clasificación oficial del ENFEN (*Normal, Débil, Moderado, Fuerte o Extraordinario*).

---

### TABLA 2: `data/provincias_piura_riesgo.csv` (8 Filas, Nivel Espacial)
Contiene las **8 provincias de Piura** con sus códigos oficiales **UBIGEO del INEI (`2001` a `2008`)**, superficie agrícola bajo riego, valles principales y porcentaje de áreas inundables reportadas por el COER Piura e INDECI tras el Niño 2017:
* `2001 - Piura` (Bajo Piura, arroz/algodón, inundación fluvial)
* `2002 - Ayabaca` (Sierra, cuenca Quiroz, deslizamientos)
* `2003 - Huancabamba` (Sierra alta, huaycos y lluvias extremas)
* `2004 - Morropón` (Chulucanas/San Lorenzo, limón y mango, lluvias de 790 mm)
* `2005 - Paita` (Margen del Chira, uva/arroz)
* `2006 - Sullana` (Valle del Chira, banano orgánico y limón)
* `2007 - Talara` (Quebrada Pariñas, agricultura estacional)
* `2008 - Sechura` (Desierto/La Niña, arroz, laguna de inundación)

---

## 🤖 4. Explicación Paso a Paso de los 5 Módulos de IA

Cada módulo cuenta con su propio cuaderno de entrenamiento individual en `notebooks/` y su controlador en `src/`:

### 🔮 Módulo 1: Predicción de Rendimiento Agrícola (Supervisado)
* **Cuaderno:** [`notebooks/01_Prediccion_Supervisada_Rendimiento.ipynb`](notebooks/01_Prediccion_Supervisada_Rendimiento.ipynb)
* **Controlador:** `src/prediction.py`
* **Paso a paso:**
  1. Carga `tabla_maestra_agro_nino.csv` y filtra por cultivo (Limón o Arroz).
  2. Define variables predictoras: `[icen, anomalia_tsm, lluvia_mensual_mm, temp_max_prom, mes]`.
  3. Divide los datos: 80% entrenamiento y 20% prueba.
  4. Compara Regresión Lineal (Baseline) vs. **Random Forest Regressor** (100 árboles de decisión, profundidad máxima 6).
  5. **Métricas obtenidas:**
     * **Limón:** $R^2 = \mathbf{0.9595}$, $\text{MAE} = 0.233\text{ t/ha}$, $\text{RMSE} = 0.314\text{ t/ha}$.
     * **Arroz:** $R^2 = \mathbf{0.9385}$, $\text{MAE} = 0.159\text{ t/ha}$, $\text{RMSE} = 0.202\text{ t/ha}$.
  6. Exporta el modelo serializado a `models/rf_yield_limon.pkl` y `models/rf_yield_arroz.pkl`.

---

### 📈 Módulo 2: Series Temporales de Precios en Chacra (Forecasting)
* **Cuaderno:** [`notebooks/02_Series_Temporales_Pronostico_Precios.ipynb`](notebooks/02_Series_Temporales_Pronostico_Precios.ipynb)
* **Controlador:** `src/timeseries.py`
* **Paso a paso:**
  1. Construye variables autorregresivas de retardo: precios e ICEN en $t-1, t-2, t-3$.
  2. Codifica la estacionalidad anual mediante funciones sinusoidales $\sin(2\pi \cdot m / 12)$ y $\cos(2\pi \cdot m / 12)$.
  3. Realiza una partición cronológica estricta (últimos 36 meses como conjunto de prueba ciego para evitar *data leakage*).
  4. Entrena un **Gradient Boosting Regressor** (120 árboles, learning rate 0.08) incorporando el **ICEN como variable exógena**.
  5. **Métricas obtenidas:** $\text{MAPE} = \mathbf{8.11\%}$ en Limón y $\mathbf{5.00\%}$ en Arroz ($\text{RMSE} = \text{S/ } 0.61\text{ / kg}$).
  6. Exporta a `models/ts_precio_limon.pkl` y `models/ts_precio_arroz.pkl`.

---

### 🍃 Módulo 3: Diagnóstico Fitosanitario de Hojas (Visión por Computadora)
* **Cuaderno:** [`notebooks/03_Clasificacion_Imagenes_Patologias_Hojas.ipynb`](notebooks/03_Clasificacion_Imagenes_Patologias_Hojas.ipynb)
* **Controlador:** `src/vision.py`
* **Dataset:** `src/vision_dataset.py`
* **Paso a paso:**
  1. Descarga 1 424 fotografías **reales** de hojas de arroz (Hugging Face `girish787/riceLeafDataset`): 488 sanas, 353 con tizón bacteriano y 583 con mancha marrón. Elimina duplicados exactos y agrupa casi-duplicados (hash perceptual) para evitar fuga de información.
  2. Redimensiona cada foto a $128 \times 128$ píxeles y extrae **123 descriptores**: momentos de color RGB/HSV, histograma Tono×Saturación, fracción de píxeles verdes/amarillos/marrones/blanquecinos, textura LBP y orientación de gradientes.
  3. Compara Regresión Logística, SVM, Random Forest y **HistGradientBoosting** con validación cruzada por grupos (mejor: HistGradientBoosting, accuracy CV $0.911 \pm 0.031$).
  4. Evalúa el modelo final en 285 fotos no vistas: **accuracy $0.954$, F1 macro $0.955$** (matriz de confusión en `Documentos/graficos_evaluacion/06_matriz_confusion_vision.png`).
  5. Exporta `models/vision_hojas_arroz.pkl` y copia 6 fotos del conjunto de prueba a `data/sample_leaves/` para la demo.
  6. En la app, despliega el diagnóstico, las probabilidades por clase y la acción fitosanitaria oficial aprobada por **SENASA**.
* **Limitación:** válido solo para fotografías de hojas de **arroz** (no limón). El tizón bacteriano solo aparece en fotos de campo, nunca sobre fondo blanco, así que en ese caso el diagnóstico es menos confiable.

---

### 🗺️ Módulo 4: Clustering de Vulnerabilidad Provincial (No Supervisado)
* **Cuaderno:** [`notebooks/04_Clustering_Vulnerabilidad_Provincias.ipynb`](notebooks/04_Clustering_Vulnerabilidad_Provincias.ipynb)
* **Controlador:** `src/clustering.py`
* **Paso a paso:**
  1. Carga los datos de las 8 provincias de Piura (`data/provincias_piura_riesgo.csv`).
  2. Estandariza las variables numéricas con `StandardScaler`.
  3. **Justificación matemática del número de grupos ($k=3$):**
     * Curva del Método del Codo (Inercia / WCSS).
     * Análisis de Silueta: alcanza su máximo en **$0.543$** con $k=3$.
  4. Entrena el algoritmo **K-Means** y aplica **PCA (Análisis de Componentes Principales)** a 2 dimensiones para graficar.
  5. **Segmentación territorial obtenida:**
     * **Rojo (Vulnerabilidad Extrema - Prioridad 1):** Morropón, Sechura, Piura y Sullana (pérdidas entre 39.5% y 51.2%, asfixia radicular y déficit crítico de drenes).
     * **Naranja (Vulnerabilidad Alta - Prioridad 2):** Ayabaca y Huancabamba.
     * **Verde (Vulnerabilidad Moderada - Prioridad 3):** Paita y Talara.
  6. Exporta a `models/kmeans_provincias.pkl` el escalador, K-Means, PCA, los mapas de riesgo/color y las variables usadas. La app **carga este modelo** y clasifica con él las provincias.

---

### 📄 Módulo 5: Resumen Inteligente de Comunicados ENFEN / SENAMHI (NLP)
* **Cuaderno:** [`notebooks/05_NLP_Resumen_Comunicados_ENFEN.ipynb`](notebooks/05_NLP_Resumen_Comunicados_ENFEN.ipynb)
* **Controlador:** `src/summarizer.py`
* **Paso a paso:**
  1. Carga el texto oficial del Comunicado ENFEN N° 04-2026 (`data/comunicados_enfen/`).
  2. Limpia el texto y segmenta en oraciones sintácticas completas.
  3. Construye una matriz **TF-IDF** eliminando *stopwords* del español.
  4. Genera un grafo de adyacencia calculando la similitud coseno entre todas las oraciones.
  5. Ejecuta el algoritmo **TextRank** (PageRank sobre oraciones con amortiguamiento $d=0.85$) para rankear las ideas más representativas.
  6. **Métricas:** Logra una **compresión textual del 76.4%**, condensando 20 páginas técnicas en una ficha ejecutiva de 4 directrices para el productor.

---

## 🔄 5. Flujo de Integración: Del Notebook a la App

```
    ┌──────────────────────────────────────────────┐
    │  Cuadernos Jupyter (notebooks/*.ipynb)       │
    │  • Experimentación, limpieza y validación.   │
    │  • Comparación de algoritmos y métricas.     │
    └──────────────────────┬───────────────────────┘
                           │ Al final de cada cuaderno:
                           │ joblib.dump(modelo, 'models/*.pkl')
                           ▼
    ┌──────────────────────────────────────────────┐
    │  Carpeta models/                             │
    │  • Modelos congelados y serializados (.pkl)  │
    └──────────────────────┬───────────────────────┘
                           │ Carga en memoria RAM al inicio:
                           │ joblib.load('models/*.pkl')
                           ▼
    ┌──────────────────────────────────────────────┐
    │  Controladores Backend (src/*.py)            │
    │  • Inferencia ultra rápida (< 50 ms)         │
    └──────────────────────┬───────────────────────┘
                           │ Consumo de funciones
                           ▼
    ┌──────────────────────────────────────────────┐
    │  Aplicación Web Streamlit (app.py)           │
    │  • Interfaz gráfica interactiva y visual     │
    └──────────────────────────────────────────────┘
```

---

## 📁 6. Estructura del Repositorio

```
Proyecto_IA/
│
├── README.md                                  # Documentación técnica completa para GitHub
├── .gitignore                                 # Reglas de exclusión de Git (sin temporales)
├── requirements.txt                           # Dependencias oficiales de Python
├── app.py                                     # Dashboard interactivo en Streamlit
│
├── data/                                      # Almacenamiento local de datos
│   ├── raw/                                   # Archivos crudos descargados del BCRP y NOAA
│   │   ├── bcrp_limon_mensual.json            # Serie BCRP PN01788AM (237 meses)
│   │   ├── bcrp_arroz_mensual.json            # Serie BCRP PN01778AM (237 meses)
│   │   └── noaa_nino12.data                   # TSM NOAA Niño 1+2 (1948–2026)
│   ├── tabla_maestra_agro_nino.csv            # Serie temporal unificada (474 registros)
│   ├── provincias_piura_riesgo.csv            # Indicadores de las 8 provincias de Piura
│   ├── raw/hojas_arroz/                       # Fotos reales de hojas (se descarga; ignorado por git)
│   ├── sample_leaves/                         # 6 fotos reales del conjunto de prueba para la demo
│   └── comunicados_enfen/                     # Textos oficiales del ENFEN y SENAMHI
│
├── models/                                    # Modelos serializados listos para inferencia
│   ├── rf_yield_limon.pkl & arroz.pkl         # Modelos de Predicción Supervisada
│   ├── ts_precio_limon.pkl & arroz.pkl        # Modelos de Series Temporales con ICEN
│   ├── vision_hojas_arroz.pkl                 # Clasificador de patologías foliares
│   └── kmeans_provincias.pkl                  # Modelo K-Means de Vulnerabilidad
│
├── notebooks/                                 # 5 Cuadernos de entrenamiento paso a paso
│   ├── 01_Prediccion_Supervisada_Rendimiento.ipynb
│   ├── 02_Series_Temporales_Pronostico_Precios.ipynb
│   ├── 03_Clasificacion_Imagenes_Patologias_Hojas.ipynb
│   ├── 04_Clustering_Vulnerabilidad_Provincias.ipynb
│   └── 05_NLP_Resumen_Comunicados_ENFEN.ipynb
│
├── src/                                       # Controladores de backend
│   ├── prediction.py                          # Lógica del Módulo 1
│   ├── timeseries.py                          # Lógica del Módulo 2
│   ├── vision.py                              # Lógica del Módulo 3
│   ├── vision_dataset.py                      # Descarga y limpieza de fotos del Módulo 3
│   ├── clustering.py                          # Lógica del Módulo 4
│   ├── summarizer.py                          # Lógica del Módulo 5
│   └── data_generator.py                      # Script que descarga y cruza BCRP/NOAA
│
└── Documentos/                                # Documentación formal exigida
    ├── Enunciado Trabajo Final.pdf            # Pauta oficial del curso
    ├── Informe_Tecnico_Avance_50.md           # Informe de desarrollo académico
    ├── Manual_Instalacion_y_Usuario.md        # Manual de instalación y operación
    └── graficos_evaluacion/                   # Gráficos en alta resolución (300 DPI)
```

---

## 💻 7. Guía de Instalación y Ejecución Paso a Paso

### Paso 1: Clonar el repositorio
```bash
git clone <URL_DE_TU_REPOSITORIO_GITHUB>
cd Proyecto_IA
```

### Paso 2: Instalar librerías requeridas
```bash
pip install -r requirements.txt
```

### Paso 3: Ejecutar la aplicación gráfica
```bash
streamlit run app.py
```
La aplicación se abrirá en tu navegador en `http://localhost:8501`.

### Paso 4 (Opcional): Reentrenar todos los modelos por consola
Si deseas regenerar los datos y reentrenar los 5 modelos de golpe sin abrir Jupyter:
```bash
python src/data_generator.py
python src/prediction.py
python src/timeseries.py
python src/clustering.py
python -m src.vision_dataset   # descarga (~91 MB) y limpia las fotos reales de hojas de arroz
python -m src.vision           # entrena y exporta models/vision_hojas_arroz.pkl
```

---

## 👥 8. Población Beneficiaria y Ventajas

### Población Usuaria en el Perú:
1. **Pequeños Agricultores y Juntas de Usuarios (San Lorenzo, Chira, Sechura):** Diagnóstico rápido de plagas desde el celular y recomendaciones sin tecnicismos.
2. **Gobierno Regional de Piura (GORE Piura) e INDECI:** Priorización de motobombas pesadas y defensas ribereñas hacia las provincias de riesgo extremo según el clustering.
3. **MIDAGRI y SENASA:** Focalización del subsidio FertiAbono y despliegue de brigadas fitosanitarias de emergencia.
4. **Agrobanco y Aseguradoras:** Liquidación objetiva de pólizas del Seguro Agrario Catastrófico (SAC).

### Ventajas Competitivas:
* **Integración Climática Activa:** Conecta la física oceanográfica (NOAA/ICEN) con el impacto agronómico futuro en chacra.
* **100% Calibrado para Piura:** Resuelve un problema nacional real con cultivos e instituciones oficiales del Perú.
* **Operación Autónoma/Offline:** Los modelos serializados corren en cualquier computadora sin depender de conexiones a internet en el campo.
* **Código Abierto y Python Nativo:** Sin costos de licenciamiento en dólares.
