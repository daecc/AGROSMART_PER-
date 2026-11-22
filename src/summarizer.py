# -*- coding: utf-8 -*-
"""
Módulo 5: Resumen Inteligente de Comunicados Oficiales ENFEN y SENAMHI (NLP).
Aplica TextRank extractivo con TF-IDF y similitud coseno para transformar reportes técnicos
de 10-20 páginas en una ficha ejecutiva clara de 3 a 5 conclusiones clave para el productor.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EnfenReportSummarizer:
    def __init__(self):
        # Stopwords en español esenciales para procesamiento de texto oficial
        self.stop_words_es = [
            'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un',
            'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le',
            'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin', 'sobre',
            'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos', 'durante',
            'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e',
            'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él',
            'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'sea', 'poco',
            'ella', 'estar', 'haber', 'estas', 'estaba', 'estamos', 'fue', 'son', 'han', 'sido'
        ]

    def split_into_sentences(self, text):
        # Limpieza y división robusta en oraciones
        clean_text = re.sub(r'\s+', ' ', text).strip()
        sentence_candidates = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9ÁÉÍÓÚÑ¿¡])', clean_text)
        # Filtrar oraciones muy cortas o encabezados
        sentences = [s.strip() for s in sentence_candidates if len(s.strip()) > 35]
        return sentences

    def summarize(self, text, num_sentences=4):
        sentences = self.split_into_sentences(text)
        if len(sentences) <= num_sentences:
            return {
                'resumen': sentences,
                'total_oraciones_original': len(sentences),
                'oraciones_resumidas': len(sentences),
                'ratio_compresion_pct': 0.0,
                'palabras_clave': []
            }

        # Vectorización TF-IDF a nivel de oraciones
        vectorizer = TfidfVectorizer(stop_words=self.stop_words_es, min_df=1)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Matriz de similitud coseno entre oraciones (Grafo de TextRank)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Algoritmo PageRank simplificado sobre el grafo de oraciones
        n = sim_matrix.shape[0]
        # Poner en 0 la diagonal para evitar auto-referencias
        np.fill_diagonal(sim_matrix, 0)
        
        # Normalizar filas
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix_norm = np.divide(sim_matrix, row_sums[:, np.newaxis], where=row_sums[:, np.newaxis] != 0)
        
        # Iteración de PageRank (damping factor d = 0.85)
        scores = np.ones(n) / n
        d = 0.85
        for _ in range(30):
            scores = (1 - d) / n + d * sim_matrix_norm.T.dot(scores)

        # Ordenar y seleccionar las mejores oraciones
        top_indices = np.argsort(scores)[::-1][:num_sentences]
        # Reordenar cronológicamente para mantener coherencia narrativa
        top_indices_sorted = sorted(top_indices)
        summary_sentences = [sentences[idx] for idx in top_indices_sorted]

        # Extraer palabras clave más relevantes
        feature_names = vectorizer.get_feature_names_out()
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        top_kw_idx = np.argsort(mean_tfidf)[::-1][:8]
        top_keywords = [feature_names[i] for i in top_kw_idx if len(feature_names[i]) > 3]

        compresion = round((1.0 - (len(" ".join(summary_sentences)) / len(text))) * 100, 1)

        return {
            'resumen': summary_sentences,
            'total_oraciones_original': len(sentences),
            'oraciones_resumidas': len(summary_sentences),
            'ratio_compresion_pct': compresion,
            'palabras_clave': top_keywords
        }

if __name__ == '__main__':
    summarizer = EnfenReportSummarizer()
    sample_text = """La Comisión Multisectorial del ENFEN actualiza las perspectivas climáticas para el litoral peruano.
    Las anomalías térmicas en la región Niño 1+2 se mantienen en valores superiores a +2.6 °C, alcanzando niveles propios de un evento Niño Costero Extraordinario.
    Se prevé que las condiciones cálidas extraordinarias persistirán durante el verano e inicio del otoño de 2027.
    En la región Piura, las parcelas de los valles del Bajo Piura y San Lorenzo enfrentan saturación hídrica severa.
    En el cultivo de limón sutil, la floración se ha inhibido por encima del 40%, sumado a la caída de frutos y asfixia radicular.
    Se recomienda reforzar los drenes agrícolas troncales y habilitar motobombas en zonas críticas."""
    
    out = summarizer.summarize(sample_text, num_sentences=3)
    print("Resumen obtenido:")
    for s in out['resumen']:
        print("-", s)
