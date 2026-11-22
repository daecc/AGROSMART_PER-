# 🌾 AgroSmart Perú: Sistema Inteligente ante El Niño Costero

> **Plataforma Integral de Inteligencia Artificial para la Mitigación del Impacto del Fenómeno El Niño Costero en la Agricultura de la Costa Norte (Piura)**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.64.0-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.1-F7931E.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Estado-Avance%2050%25%20Completado-green.svg)](#)

---

## 📌 1. Descripción y Problemática Real

El Fenómeno de **El Niño Costero (2026–2027)**, clasificado por el **ENFEN** con magnitud extraordinaria, representa la principal amenaza hidroclimática para el agro en la costa norte del Perú. 

En la región **Piura** (valles de San Lorenzo, Chulucanas y Bajo Piura), las anomalías térmicas en la región **Niño 1+2** superan los $+2.6^\circ\text{C}$, provocando lluvias torrenciales e inundaciones. Esto genera:
1. **Pérdidas de cosecha severas:** El limón sutil sufre asfixia radicular y aborto floral (caída del 37.9% en producción registrada por el BCRP en 2017 y más del 40% en 2026), disparando los precios al consumidor en más de 300%.
2. **Proliferación de patógenos foliares:** La alta humedad desata plagas bacterianas (*Xanthomonas oryzae*) y fúngicas (*Bipolaris oryzae*) en el arroz.
3. **Sobrecarga de alertas técnicas:** Dificultad de los agricultores para interpretar extensos reportes técnicos del ENFEN y SENAMHI.

**AgroSmart Perú** resuelve esta problemática integrando **5 módulos de Inteligencia Artificial** con interfaz gráfica web interactiva en **Streamlit**.

---

## 🚀 2. Los 5 Módulos de Inteligencia Artificial

| # | Módulo | Tipo de IA | Algoritmo y Desempeño | Cuaderno de Entrenamiento |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Predicción de Rendimiento** | Supervisado (Regresión) | **Random Forest Regressor** ($R^2 = 0.9595$ en Limón, $R^2 = 0.9385$ en Arroz, MAE: $0.23\text{ t/ha}$) | [`notebooks/01_Prediccion_Supervisada_Rendimiento.ipynb`](notebooks/01_Prediccion_Supervisada_Rendimiento.ipynb) |
| **2** | **Series Temporales de Precios** | Forecasting Multivariado | **Multi-Lag Gradient Boosting** con ICEN como regresor exógeno (MAPE: $8.11\%$) | [`notebooks/02_Series_Temporales_Pronostico_Precios.ipynb`](notebooks/02_Series_Temporales_Pronostico_Precios.ipynb) |
| **3** | **Diagnóstico Fitosanitario** | Visión por Computadora | **Clasificador cromático/textural RGB** ($>88\%$ certeza diagnóstica + directriz SENASA) | [`notebooks/03_Clasificacion_Imagenes_Patologias_Hojas.ipynb`](notebooks/03_Clasificacion_Imagenes_Patologias_Hojas.ipynb) |
| **4** | **Clustering de Vulnerabilidad** | No Supervisado | **K-Means ($k=3$) + PCA** (Coeficiente Silhouette $= 0.543$) en 8 provincias de Piura | [`notebooks/04_Clustering_Vulnerabilidad_Provincias.ipynb`](notebooks/04_Clustering_Vulnerabilidad_Provincias.ipynb) |
| **5** | **Resumen de Alertas ENFEN** | NLP (Procesamiento de Lenguaje) | **TextRank Graph-based con TF-IDF** ($76.4\%$ de compresión a ficha ejecutiva) | [`notebooks/05_NLP_Resumen_Comunicados_ENFEN.ipynb`](notebooks/05_NLP_Resumen_Comunicados_ENFEN.ipynb) |

---

## 🏛️ 3. Fuentes de Datos Oficiales y Auditables

El proyecto utiliza un pipeline transparente de datos:
* **Banco Central de Reserva del Perú (BCRP):** Series oficiales de producción mensual `PN01788AM` (Limón) y `PN01778AM` (Arroz) de 2007 a 2026.
* **NOAA Physical Sciences Laboratory (EE.UU.):** Temperatura marina mensual en la región **Niño 1+2** desde 1948 hasta 2026 (`data/raw/noaa_nino12.data`).
* **IGP / ENFEN:** Cálculo formal del **Índice Costero El Niño (ICEN)** según la metodología oficial (media móvil trimestral sobre anomalía climatológica 1981–2010).
* **INEI / SENAMHI / MIDAGRI:** Indicadores agroclimáticos de las 8 provincias de Piura (UBIGEO `2001` a `2008`).

---

## 📁 4. Estructura del Repositorio

```
Proyecto_IA/
│
├── README.md                                  # Documentación principal de GitHub
├── .gitignore                                 # Reglas de exclusión para Git
├── requirements.txt                           # Dependencias oficiales de Python
├── app.py                                     # Interfaz gráfica interactiva (Streamlit)
│
├── data/                                      # Almacenamiento local de datos
│   ├── raw/                                   # Archivos crudos descargados del BCRP y NOAA
│   ├── tabla_maestra_agro_nino.csv            # Serie temporal mensual consolidada (474 registros)
│   ├── provincias_piura_riesgo.csv            # Indicadores de las 8 provincias de Piura
│   ├── sample_leaves/                         # Muestras foliares (Hoja sana, Tizón, Mancha marrón)
│   └── comunicados_enfen/                     # Textos oficiales del ENFEN y SENAMHI
│
├── models/                                    # Modelos serializados (.pkl) listos para inferencia
│   ├── rf_yield_limon.pkl & arroz.pkl
│   ├── ts_precio_limon.pkl & arroz.pkl
│   └── kmeans_provincias.pkl
│
├── notebooks/                                 # Cuadernos modulares de entrenamiento (Jupyter / Colab)
│   ├── 01_Prediccion_Supervisada_Rendimiento.ipynb
│   ├── 02_Series_Temporales_Pronostico_Precios.ipynb
│   ├── 03_Clasificacion_Imagenes_Patologias_Hojas.ipynb
│   ├── 04_Clustering_Vulnerabilidad_Provincias.ipynb
│   └── 05_NLP_Resumen_Comunicados_ENFEN.ipynb
│
├── src/                                       # Lógica y controladores del backend
│   ├── prediction.py
│   ├── timeseries.py
│   ├── vision.py
│   ├── clustering.py
│   ├── summarizer.py
│   └── data_generator.py
│
└── Documentos/                                # Entregables académicos formales
    ├── Enunciado Trabajo Final.pdf
    ├── Informe_Tecnico_Avance_50.md           # Informe de desarrollo según pauta
    ├── Manual_Instalacion_y_Usuario.md        # Manual de instalación y operación
    └── graficos_evaluacion/                   # Gráficos en alta resolución
```

---

## ⚙️ 5. Instalación y Ejecución Rápida

### Requisitos:
* Python 3.10, 3.11 o 3.12 instalado.

### Pasos:
```bash
# 1. Clonar el repositorio
git clone <URL_DEL_REPOSITORIO>
cd Proyecto_IA

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar la aplicación web
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`.

---

## 👥 6. Entregables Académicos

* 📄 **[Informe Técnico de Desarrollo (Avance 50%)](Documentos/Informe_Tecnico_Avance_50.md)**
* 📖 **[Manual de Instalación y de Usuario](Documentos/Manual_Instalacion_y_Usuario.md)**
