# -*- coding: utf-8 -*-
"""
AgroSmart Perú: Sistema Inteligente para la Mitigación del Impacto de El Niño Costero en el Agro
Aplicación Principal en Streamlit con 5 Módulos Integrados.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Importación de módulos del proyecto
from src.prediction import CropYieldPredictor
from src.timeseries import AgroTimeSeriesForecaster
from src.vision import PlantDiseaseClassifier
from src.clustering import VulnerabilityClusterer
from src.summarizer import EnfenReportSummarizer

# Configuración de página
st.set_page_config(
    page_title="AgroSmart Perú | El Niño Costero",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1b5e20;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #455a64;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        border-left: 5px solid #2e7d32;
        margin-bottom: 1rem;
    }
    .alert-banner {
        background: linear-gradient(90deg, #d32f2f, #f57c00);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Instanciación de componentes
@st.cache_resource
def get_components():
    return {
        'predictor': CropYieldPredictor(),
        'forecaster': AgroTimeSeriesForecaster(),
        'vision': PlantDiseaseClassifier(),
        'clusterer': VulnerabilityClusterer(),
        'summarizer': EnfenReportSummarizer()
    }

components = get_components()

# Barra lateral de navegación
st.sidebar.image("https://images.unsplash.com/photo-1586771107445-d3ca888129ff?w=400&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🌾 AgroSmart Perú")
st.sidebar.markdown("**Monitoreo & Mitigación del Fenómeno El Niño**")
st.sidebar.markdown("---")

opcion = st.sidebar.radio(
    "Seleccione un Módulo:",
    [
        "🏠 Panel General y Diagnóstico El Niño",
        "🔮 1. Predicción de Rendimiento (Supervisado)",
        "📈 2. Series Temporales de Precios (Forecasting)",
        "🍃 3. Diagnóstico Fitosanitario de Hojas (Visión)",
        "🗺️ 4. Clustering de Vulnerabilidad Provincial (K-Means)",
        "📄 5. Resumen de Comunicados ENFEN / SENAMHI (NLP)"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Área de Estudio:** Región Piura (Costa Norte)  
**Cultivos Foco:** Arroz Cáscara y Limón Sutil  
**Fuentes Integradas:** MIDAGRI (SIEA), BCRP, IGP (ICEN), NOAA (Niño 1+2), SENAMHI y ENFEN.
""")

# ==============================================================================
# VISTA 0: PANEL GENERAL
# ==============================================================================
if opcion == "🏠 Panel General y Diagnóstico El Niño":
    st.markdown('<div class="main-title">🌾 AgroSmart Perú: Plataforma de Inteligencia Agrícola</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Monitoreo Predictivo y Mitigación del Impacto de El Niño Costero 2026 - 2027</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-banner">
        ⚠️ ESTADO OFICIAL ENFEN: ALERTA DE EL NIÑO COSTERO (MAGNITUD EXTRAORDINARIA) — ICEN +2.6 °C | TSM NIÑO 1+2: +2.9 °C
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Índice ICEN Actual", "+2.58 °C", "Anomalía Cálida")
    with col2:
        st.metric("Lluvias en Piura (Mes)", "385 mm", "+240% vs Climatología")
    with col3:
        st.metric("Pérdida Proyectada Limón", "-38.5%", "Caída de Floración")
    with col4:
        st.metric("Precio Limón en Chacra", "S/ 14.80 / kg", "+320% pico de escasez")

    st.markdown("### 📊 Evolución Histórica de El Niño y Rendimientos en Piura (2007 - 2026)")
    df_data = pd.read_csv(os.path.join("data", "tabla_maestra_agro_nino.csv"))
    df_data['fecha'] = pd.to_datetime(df_data['fecha'])
    
    cultivo_sel = st.selectbox("Seleccione Cultivo para Exploración:", ["Limon", "Arroz"])
    df_sub = df_data[df_data['cultivo'] == cultivo_sel].copy()

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax2 = ax1.twinx()

    ax1.plot(df_sub['fecha'], df_sub['rendimiento_tha'], color='#2e7d32', label=f'Rendimiento {cultivo_sel} (t/ha)', lw=2)
    ax2.plot(df_sub['fecha'], df_sub['icen'], color='#d32f2f', linestyle='--', label='Índice Costero El Niño (ICEN °C)', alpha=0.8)
    ax2.axhline(1.5, color='orange', linestyle=':', label='Umbral Niño Moderado')
    ax2.axhline(2.0, color='red', linestyle=':', label='Umbral Niño Fuerte')

    ax1.set_ylabel('Rendimiento (t/ha)', color='#2e7d32')
    ax2.set_ylabel('ICEN (°C)', color='#d32f2f')
    ax1.set_title(f'Impacto de las Ondas Cálidas de El Niño en el Rendimiento de {cultivo_sel} (Piura)')
    ax1.grid(True, alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    st.pyplot(fig)

    st.markdown("""
    #### 💡 Hallazgos Principales:
    - **Años de Impacto Severo:** Se observa una caída drástica en el rendimiento agrícola durante los eventos de **2017 (Niño Costero)**, **2023** y el actual **2026**.
    - **Comportamiento del Limón:** El exceso de humedad asfixia las raíces del limonero en Chulucanas y Tambogrande, derrumbando la producción hasta en 40%.
    - **Comportamiento del Arroz:** En el Bajo Piura, las inundaciones anegan parcelas en plena maduración y disparan plagas bacterianas (*Xanthomonas*).
    """)

# ==============================================================================
# VISTA 1: PREDICCIÓN SUPERVISADA
# ==============================================================================
elif opcion == "🔮 1. Predicción de Rendimiento (Supervisado)":
    st.markdown("## 🔮 Módulo 1: Predicción de Rendimiento Agrícola ante El Niño")
    st.markdown("Modelo de Machine Learning (**Random Forest Regressor**) calibrado con datos del BCRP y MIDAGRI.")

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("⚙️ Variables Climáticas y del Entorno")
        cultivo = st.selectbox("Cultivo a Evaluar:", ["Limon", "Arroz"])
        mes = st.slider("Mes del Año:", 1, 12, 3, help="Mes crítico de lluvias en Piura: Febrero a Abril")
        icen = st.slider("Índice Costero El Niño (ICEN °C):", -1.0, 3.5, 2.2, 0.1, help="> 1.5°C Niño Moderado, > 2.0°C Niño Fuerte/Extraordinario")
        anomalia_tsm = st.slider("Anomalía Temperatura del Mar Niño 1+2 (°C):", -1.5, 4.0, 2.5, 0.1)
        lluvia_mm = st.number_input("Precipitación Acumulada Mensual (mm):", min_value=0.0, max_value=800.0, value=250.0, step=10.0)
        temp_max = st.slider("Temperatura Máxima Promedio (°C):", 28.0, 38.0, 34.5, 0.5)

        btn_predict = st.button("🚀 Ejecutar Predicción de Rendimiento", use_container_width=True)

    with col_result:
        st.subheader("🎯 Resultado de la Predicción")
        if btn_predict:
            res = components['predictor'].predict(cultivo, icen, anomalia_tsm, lluvia_mm, temp_max, mes)
            
            st.metric(
                label=f"Rendimiento Estimado ({cultivo})",
                value=f"{res['rendimiento_estimado_tha']} t/ha",
                delta=f"-{res['perdida_estimada_pct']}% vs Promedio Normal ({res['rendimiento_historico_base']} t/ha)",
                delta_color="inverse"
            )

            if res['color'] == 'red':
                st.error(f"🚨 **Nivel de Alerta:** {res['alerta_riesgo']}")
                st.warning("⚠️ **Medidas Urgentes:** Evacuar drenes parcelarios, postergar fertilización nitrogenada y habilitar motobombas.")
            elif res['color'] == 'orange':
                st.warning(f"⚠️ **Nivel de Alerta:** {res['alerta_riesgo']}")
            else:
                st.success(f"✅ **Nivel de Alerta:** {res['alerta_riesgo']}")

            st.markdown("#### 🔍 Importancia de Variables en el Modelo:")
            imp_df = pd.DataFrame(list(res['importancia_variables'].items()), columns=['Variable', 'Importancia']).sort_values('Importancia', ascending=True)
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.barh(imp_df['Variable'], imp_df['Importancia'], color='#2e7d32')
            ax.set_xlabel('Peso Relativo en la Decisión')
            st.pyplot(fig)
        else:
            st.info("Ajuste los valores climáticos en el panel izquierdo y presione el botón para calcular el rendimiento.")

# ==============================================================================
# VISTA 2: SERIES TEMPORALES
# ==============================================================================
elif opcion == "📈 2. Series Temporales de Precios (Forecasting)":
    st.markdown("## 📈 Módulo 2: Pronóstico Temporal de Precios en Chacra")
    st.markdown("Modelo Autorregresivo Multivariado con el **Índice ICEN de El Niño como variable exógena**.")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("⚙️ Configuración del Pronóstico")
        cultivo_ts = st.selectbox("Seleccione Producto:", ["Limon", "Arroz"])
        horizonte = st.slider("Horizonte de Pronóstico (Meses):", 3, 12, 6)
        escenario = st.selectbox(
            "Escenario Climático Futuro (ENFEN):",
            [
                "Extraordinario (ICEN +2.6°C - Calor Extremo)",
                "Fuerte (ICEN +2.1°C)",
                "Moderado (ICEN +1.4°C)",
                "Normal / Neutro (ICEN +0.2°C)"
            ]
        )
        btn_forecast = st.button("📊 Proyectar Precios Futuros", use_container_width=True)

    with col2:
        st.subheader("📈 Proyección de Precios (S/ por Kilogramo)")
        if btn_forecast:
            hist_df, fut_df = components['forecaster'].forecast_future(cultivo_ts, horizonte, escenario)
            
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(hist_df['fecha'], hist_df['precio_chacra_soles_kg'], 'o-', color='#1565c0', label='Histórico Reciente (Soles/kg)')
            ax.plot(fut_df['fecha'], fut_df['precio_pronosticado_soles_kg'], 's--', color='#d32f2f', lw=2, label=f'Pronóstico {escenario.split()[0]}')
            
            ax.set_ylabel('Precio en Chacra (S/ por kg)')
            ax.set_title(f'Tendencia y Proyección de Precios de {cultivo_ts} ante El Niño')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)

            st.write("📋 **Valores Pronosticados para los Próximos Meses:**")
            st.dataframe(fut_df, use_container_width=True)
        else:
            st.info("Configure el escenario climático y presione 'Proyectar Precios Futuros' para visualizar la curva.")

# ==============================================================================
# VISTA 3: CLASIFICACIÓN DE IMÁGENES
# ==============================================================================
elif opcion == "🍃 3. Diagnóstico Fitosanitario de Hojas (Visión)":
    st.markdown("## 🍃 Módulo 3: Clasificación de Patologías en Hojas")
    st.markdown("Detección de plagas fúngicas y bacterianas proliferadas por el calor y humedad de El Niño.")

    vision_clf = components['vision']
    if vision_clf.model is None:
        vision_clf.load_model()
    m_vis = vision_clf.metrics or {}
    if m_vis:
        st.caption(
            f"Modelo: HistGradientBoosting sobre 123 descriptores de color y textura, entrenado con "
            f"{m_vis['n_train']} fotos reales de campo de arroz. Evaluación en {m_vis['n_test']} fotos no vistas: "
            f"accuracy = {m_vis['accuracy']*100:.1f}%, F1 macro = {m_vis['f1_macro']*100:.1f}%."
        )
    st.warning(
        "⚠️ Alcance: el modelo fue entrenado solo con **fotografías de hojas de arroz** (en campo o "
        "sobre fondo blanco). No es válido para limón ni para dibujos. El tizón bacteriano solo aparece "
        "en fotos de campo, así que una hoja con tizón fotografiada sobre fondo blanco puede clasificarse mal. "
        "Fuera de ese alcance el resultado no es confiable aunque la probabilidad mostrada sea alta."
    )

    col_img, col_diag = st.columns([1, 1])

    with col_img:
        st.subheader("📸 Imagen de la Hoja a Diagnosticar")
        modo = st.radio("Método de entrada:", ["Usar foto de muestra (conjunto de prueba)", "Subir mi propia foto"])

        selected_img = None
        if modo == "Usar foto de muestra (conjunto de prueba)":
            muestras_disp = sorted(
                f for f in os.listdir(os.path.join("data", "sample_leaves"))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            muestra = st.selectbox("Seleccione una foto real de campo (no usada en el entrenamiento):", muestras_disp)
            img_path = os.path.join("data", "sample_leaves", muestra)
            if os.path.exists(img_path):
                selected_img = Image.open(img_path)
                st.image(selected_img, caption=f"Muestra: {muestra}", use_container_width=True)
        else:
            uploaded = st.file_uploader("Suba una imagen foliar (JPG/PNG):", type=['jpg', 'jpeg', 'png'])
            if uploaded:
                selected_img = Image.open(uploaded)
                st.image(selected_img, caption="Imagen cargada", use_container_width=True)

    with col_diag:
        st.subheader("🔬 Diagnóstico y Recomendación Fitosanitaria")
        if selected_img is not None:
            res_vis = vision_clf.predict_image(selected_img)

            if res_vis['color'] == 'green':
                st.success(f"### Diagnóstico: {res_vis['clase_detectada']} ({res_vis['confianza_pct']}%)")
            elif res_vis['color'] == 'orange':
                st.warning(f"### Diagnóstico: {res_vis['clase_detectada']} ({res_vis['confianza_pct']}%)")
            else:
                st.error(f"### Diagnóstico: {res_vis['clase_detectada']} ({res_vis['confianza_pct']}%)")
            if res_vis['confianza_baja']:
                st.warning("Confianza baja: el modelo no distingue con claridad la clase. Se recomienda confirmar con un técnico de SENASA.")

            st.write(f"**Severidad:** {res_vis['severidad']}")
            st.write(f"**Descripción Patológica:** {res_vis['diagnostico']}")
            st.info(f"**Tratamiento y Acción SENASA:** {res_vis['accion_recomendada']}")

            st.markdown("#### Probabilidades por Clase:")
            prob_df = pd.DataFrame(list(res_vis['probabilidades'].items()), columns=['Enfermedad', 'Probabilidad (%)'])
            st.bar_chart(prob_df.set_index('Enfermedad'))
        else:
            st.info("Seleccione o suba una imagen en el panel izquierdo para procesar.")

# ==============================================================================
# VISTA 4: CLUSTERING DE VULNERABILIDAD
# ==============================================================================
elif opcion == "🗺️ 4. Clustering de Vulnerabilidad Provincial (K-Means)":
    st.markdown("## 🗺️ Módulo 4: Zonificación de Vulnerabilidad en Piura (K-Means)")
    st.markdown("Segmentación de las 8 provincias de Piura según su riesgo y daño histórico por El Niño.")

    df_cluster = components['clusterer'].get_clustered_data()

    col_map, col_table = st.columns([1, 1])

    with col_map:
        st.subheader("🎯 Espacio de Dispersión PCA (2 Componentes Principales)")
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for name, group in df_cluster.groupby('nivel_vulnerabilidad'):
            color = group['color_alerta'].iloc[0]
            ax.scatter(group['pca_x'], group['pca_y'], s=150, label=name, color=color, edgecolors='black')
            for _, row in group.iterrows():
                ax.annotate(row['provincia'], (row['pca_x'] + 0.1, row['pca_y'] + 0.1), fontsize=10, fontweight='bold')

        ax.set_xlabel('Componente Principal 1 (Exposición a Lluvias y Daño Histórico)')
        ax.set_ylabel('Componente Principal 2 (Déficit de Drenaje Parcelario)')
        ax.set_title('Zonificación de las Provincias de Piura por Clusters de Riesgo')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower left')
        st.pyplot(fig)

    with col_table:
        st.subheader("📋 Clasificación Oficial por Provincia")
        st.dataframe(
            df_cluster[['provincia', 'cultivo_predominante', 'nivel_vulnerabilidad', 'perdida_cosecha_nino_pct', 'pct_area_inundable']],
            use_container_width=True
        )

        # Los grupos se arman a partir de la predicción del modelo exportado (models/kmeans_provincias.pkl)
        def _provs(prefijo):
            sel = df_cluster[df_cluster['nivel_vulnerabilidad'].str.startswith(prefijo)]['provincia']
            return ", ".join(sel) if len(sel) else "—"

        st.markdown(f"""
        #### 🏛️ Asignación de Recursos para el GORE Piura y MIDAGRI:
        - **Grupo Rojo ({_provs('Vulnerabilidad Extrema')}):** Prioridad inmediata para traslado de motobombas pesadas de 12 pulgadas y descolmatación de drenes troncales.
        - **Grupo Naranja ({_provs('Vulnerabilidad Alta')}):** Monitoreo continuo de deslizamientos en quebradas y vigilancia de las zonas altas de la cuenca.
        - **Grupo Verde ({_provs('Vulnerabilidad Moderada')}):** Menor anegamiento de tierras de cultivo, habilitación como centros de acopio temporal.
        """)

# ==============================================================================
# VISTA 5: RESUMEN DE TEXTO NLP
# ==============================================================================
elif opcion == "📄 5. Resumen de Comunicados ENFEN / SENAMHI (NLP)":
    st.markdown("## 📄 Módulo 5: Resumen Inteligente de Comunicados Oficiales (NLP)")
    st.markdown("Procesamiento de Lenguaje Natural con **TextRank y TF-IDF** para sintetizar documentos técnicos extensos.")

    comunicados_disponibles = {
        "Comunicado Oficial ENFEN N° 04-2026 (Alerta El Niño Extraordinario)": "data/comunicados_enfen/comunicado_enfen_n04_2026.txt",
        "Aviso Meteorológico SENAMHI N° 182-2026 (Lluvias Extremas Costa Norte)": "data/comunicados_enfen/aviso_senamhi_lluvias_norte.txt"
    }

    sel_doc = st.selectbox("Seleccione un Documento Oficial precargado:", list(comunicados_disponibles.keys()))
    
    file_path = comunicados_disponibles[sel_doc]
    default_text = ""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            default_text = f.read()

    texto_input = st.text_area("Texto del Boletín o Comunicado:", value=default_text, height=220)
    
    col_param1, col_param2 = st.columns([1, 3])
    with col_param1:
        num_sent = st.slider("Número de Conclusiones Clave:", 2, 6, 4)
        btn_resumir = st.button("📝 Generar Ficha Ejecutiva", use_container_width=True)

    with col_param2:
        if btn_resumir and texto_input.strip():
            resultado = components['summarizer'].summarize(texto_input, num_sentences=num_sent)
            
            st.success(f"✅ Resumen Generado con Éxito (Compresión: {resultado['ratio_compresion_pct']}%)")
            st.write(f"**Oraciones evaluadas:** {resultado['total_oraciones_original']} ➡️ **Oraciones seleccionadas:** {resultado['oraciones_resumidas']}")
            
            st.markdown("### 📌 Ficha Ejecutiva para Agricultores y Técnicos:")
            for s in resultado['resumen']:
                st.markdown(f"- 🔸 **{s}**")
                
            st.markdown(f"**Palabras clave detectadas:** `{', '.join(resultado['palabras_clave'])}`")
        else:
            st.info("Presione 'Generar Ficha Ejecutiva' para extraer las conclusiones del texto oficial.")
