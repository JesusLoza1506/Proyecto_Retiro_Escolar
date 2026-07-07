import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def render_numericas(df):
    st.title("📈 Análisis de Features Numéricas - Estadísticos y Anomalías")
    st.markdown("*Módulo de auditoría exhaustiva de variables continuas: matrícula, ratios, docentes y detección de la anomalía en la planta contractual.*")
    st.divider()

    # Features de interés explícitas de tu backend
    features_interes = [
        'MATRICULA_G1', 'MATRICULA_G2', 'MATRICULA_G3', 'MATRICULA_G4', 'MATRICULA_G5',
        'RATIO_G1', 'RATIO_G2', 'RATIO_G3', 'RATIO_G4', 'RATIO_G5',
        'TOTAL_DOCENTES', 'PCT_DOCENTES_FEMENINO', 'PCT_CONTRATO_PERMANENTE', 'PCT_TITULADOS'
    ]

    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    # Estandarización de columnas a mayúsculas para evitar colisiones
    df_upper = df.copy()
    df_upper.columns = [c.upper() for c in df_upper.columns]
    features_presentes = [c for c in features_interes if c in df_upper.columns]

    # =========================================================================
    # 1. CARGA DE DATOS Y CONFIGURACIÓN INICIAL
    # =========================================================================
    st.header("1. Carga de Datos y Configuración Inicial")
    st.info(f"💾 **Dataset Cargado:** {df.shape[0]:,} registros | {df.shape[1]} columnas")
    
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['cod_mod', 'cod_local', 'cod_ugel', 'cod_dre', 
                    'target_g1', 'target_g2', 'target_g3', 'target_g4', 'target_g5']
    numeric_features = [col for col in numeric_features if col.lower() not in exclude_cols]
    
    col_inf1, col_inf2 = st.columns([1, 2])
    with col_inf1:
        st.metric(label="Features Numéricas Detectadas", value=len(numeric_features))
    with col_inf2:
        with st.expander("👁️ Ver listado de todas las columnas y sus dtypes"):
            df_types = pd.DataFrame({'Columna': df.columns, 'Tipo': [str(t) for t in df.dtypes]})
            st.dataframe(df_types, use_container_width=True, hide_index=True)
            
    st.divider()

    # =========================================================================
    # 2. ESTADÍSTICOS DESCRIPTIVOS DE FEATURES NUMÉRICAS
    # =========================================================================
    st.header("2. Estadísticos Descriptivos de Features Numéricas")
    st.subheader("📊 Matriz de Estadísticos Descriptivos Completos")
    stats_completos = df_upper[features_presentes].describe()
    st.dataframe(stats_completos.round(2), use_container_width=True)
    
    st.subheader("🔍 Análisis Analítico de Valores Vacíos / Nulos")
    nulos_info = pd.DataFrame({
        'Feature': features_presentes,
        'Nulos': [df_upper[col].isnull().sum() for col in features_presentes],
        'Pct_Nulos': [df_upper[col].isnull().sum() / len(df_upper) * 100 for col in features_presentes]
    })
    
    st.dataframe(nulos_info.round(2), hide_index=True, use_container_width=True)
    stats_completos.to_csv('./outputs/estadisticos_features_numericas.csv')
    nulos_info.to_csv('./outputs/analisis_nulos_features_numericas.csv', index=False)
    st.success("✅ Archivos de auditoría descriptiva guardados con éxito en `./outputs/`")
    st.divider()

    # =========================================================================
    # 3. ANÁLISIS ESPECÍFICO DE PCT_CONTRATO_PERMANENTE - LA ANOMALÍA
    # =========================================================================
    st.header("3. Análisis Específico de PCT_CONTRATO_PERMANENTE - La Anomalía")
    
    if 'PCT_CONTRATO_PERMANENTE' in df_upper.columns:
        pct_contrato_validos = df_upper['PCT_CONTRATO_PERMANENTE'].dropna()
        
        c_an1, c_an2, c_an3, c_an4 = st.columns(4)
        with c_an1: st.metric(label="Registros Válidos", value=f"{len(pct_contrato_validos):,}")
        with c_an2: st.metric(label="Media General", value=f"{pct_contrato_validos.mean():.4f}")
        with c_an3: st.metric(label="Mínimo Detectado", value=f"{pct_contrato_validos.min():.4f}")
        with c_an4: st.metric(label="Máximo Detectado", value=f"{pct_contrato_validos.max():.4f}")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**📋 Top 10 Valores Más Frecuentes (Moda Backend):**")
            distribucion = pct_contrato_validos.value_counts().sort_index()
            st.dataframe(distribucion.head(10), use_container_width=True)
        with col_d2:
            st.markdown("**📊 Análisis de Percentiles de Distribución:**")
            percentiles_list = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct_res = {"Percentil": [f"P{p}" for p in percentiles_list], "Valor": [np.percentile(pct_contrato_validos, p) for p in percentiles_list]}
            st.dataframe(pd.DataFrame(pct_res), hide_index=True, use_container_width=True)
    st.divider()

    # =========================================================================
    # 4. VISUALIZACIONES DE FEATURES NUMÉRICAS (HISTOGRAMAS Y BOXPLOTS)
    # =========================================================================
    st.header("4. Visualizaciones de Features Numéricas")
    df_viz = df_upper[features_presentes].dropna()

    if not df_viz.empty:
        # Pestañas para separar de forma elegante los dos tableros 4x4
        tab_dist, tab_outliers = st.tabs(["📊 Histogramas de Distribución", "📦 Diagramas de Caja (Outliers)"])
        
        with tab_dist:
            st.markdown("**Distribución de Frecuencias con Línea de Media Central (Rojo):**")
            fig1, axes1 = plt.subplots(4, 4, figsize=(20, 16))
            fig1.suptitle('Análisis de Features Numéricas - Distribuciones', fontsize=16, y=0.98)
            
            for i, feature in enumerate(features_presentes):
                row = i // 4
                col = i % 4
                ax = axes1[row, col]
                ax.hist(df_viz[feature], bins=50, alpha=0.7, edgecolor='black', color='steelblue')
                ax.set_title(f'{feature}\n(Media: {df_viz[feature].mean():.2f})', fontsize=10)
                ax.axvline(df_viz[feature].mean(), color='red', linestyle='--', alpha=0.8)
            
            plt.tight_layout()
            fig1.savefig('./outputs/distribucion_features_numericas.png', dpi=300, bbox_inches='tight')
            st.pyplot(fig1)
            plt.close(fig1)
            
        with tab_outliers:
            st.markdown("**Cálculo Estadístico de Valores Atípicos usando el Rango Intercuartílico ($IQR$):**")
            fig2, axes2 = plt.subplots(4, 4, figsize=(20, 16))
            fig2.suptitle('Análisis de Features Numéricas - Boxplots (Outliers)', fontsize=16, y=0.98)
            
            for i, feature in enumerate(features_presentes):
                row = i // 4
                col = i % 4
                ax = axes2[row, col]
                bp = ax.boxplot(df_viz[feature], patch_artist=True)
                bp['boxes'][0].set_facecolor('lightblue')
                bp['boxes'][0].set_alpha(0.7)
                ax.set_title(f'{feature}', fontsize=10)
                
                q1 = df_viz[feature].quantile(0.25)
                q3 = df_viz[feature].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df_viz[feature] < (q1 - 1.5 * iqr)) | (df_viz[feature] > (q3 + 1.5 * iqr))).sum()
                
                ax.text(0.5, 0.95, f'Outliers: {outliers}', transform=ax.transAxes, 
                        ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            fig2.savefig('./outputs/boxplots_features_numericas.png', dpi=300, bbox_inches='tight')
            st.pyplot(fig2)
            plt.close(fig2)
            
        st.success("🎨 Gráficos estructurales guardados en `./outputs/distribucion_features_numericas.png` y `./outputs/boxplots_features_numericas.png`")
    else:
        st.error("❌ No hay suficientes registros limpios (no nulos) para generar las matrices matriciales.")
    st.divider()

    # =========================================================================
    # 5. ANÁLISIS ESPECÍFICO DE PCT_CONTRATO_PERMANENTE - VISUALIZACIÓN DE ANOMALÍA
    # =========================================================================
    st.header("5. Análisis Visual Específico de la Anomalía")
    
    if 'PCT_CONTRATO_PERMANENTE' in df_upper.columns:
        pct_contrato = df_upper['PCT_CONTRATO_PERMANENTE'].dropna()
        
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
        fig3.suptitle('PCT_CONTRATO_PERMANENTE - Análisis Detallado de la Anomalía', fontsize=16)

        # 1. Histograma general
        axes3[0,0].hist(pct_contrato, bins=100, alpha=0.7, edgecolor='black', color='crimson')
        axes3[0,0].set_title('Distribución General')
        axes3[0,0].axvline(pct_contrato.mean(), color='blue', linestyle='--', label=f'Media: {pct_contrato.mean():.3f}')
        axes3[0,0].axvline(pct_contrato.median(), color='green', linestyle='--', label=f'Mediana: {pct_contrato.median():.3f}')
        axes3[0,0].legend()

        # 2. Zoom en valores bajos (0-0.5)
        valores_bajos = pct_contrato[pct_contrato <= 0.5]
        axes3[0,1].hist(valores_bajos, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes3[0,1].set_title(f'Zoom: Valores 0-0.5 ({len(valores_bajos):,} registros)')

        # 3. Concentración en 0 (Pie Chart)
        ceros = (pct_contrato == 0.0).sum()
        no_ceros = (pct_contrato > 0.0).sum()
        labels = [f'Exactamente 0\n({ceros:,} - {ceros/len(pct_contrato)*100:.1f}%)', 
                  f'Mayor a 0\n({no_ceros:,} - {no_ceros/len(pct_contrato)*100:.1f}%)']
        axes3[1,0].pie([ceros, no_ceros], labels=labels, colors=['lightcoral', 'lightblue'], autopct='%1.1f%%', startangle=90)
        axes3[1,0].set_title('Concentración en Cero vs No-Cero')

        # 4. Distribución por rangos
        rangos = ['0.0', '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0', '1.0']
        conteos = [
            (pct_contrato == 0.0).sum(),
            ((pct_contrato > 0.0) & (pct_contrato <= 0.2)).sum(),
            ((pct_contrato > 0.2) & (pct_contrato <= 0.4)).sum(),
            ((pct_contrato > 0.4) & (pct_contrato <= 0.6)).sum(),
            ((pct_contrato > 0.6) & (pct_contrato <= 0.8)).sum(),
            ((pct_contrato > 0.8) & (pct_contrato < 1.0)).sum(),
            (pct_contrato == 1.0).sum()
        ]
        axes3[1,1].bar(rangos, conteos, alpha=0.7, color='skyblue', edgecolor='black')
        axes3[1,1].set_title('Distribución por Rangos')
        axes3[1,1].tick_params(axis='x', rotation=45)
        for i, v in enumerate(conteos):
            axes3[1,1].text(i, v + 200, f'{v:,}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig3.savefig('./outputs/anomalia_pct_contrato_permanente.png', dpi=300, bbox_inches='tight')
        st.pyplot(fig3)
        plt.close(fig3)
        
        # Recuadro explicativo de la anomalía bimodal
        st.info("""
        🔬 **Interpretación del Diagnóstico:**
        * **Distribución Bimodal Extrema:** El dataset muestra una polarización total en los sistemas de contratación.
        * **46.4% de escuelas** reportan un $0\%$ absoluto de docentes permanentes (alta temporalidad).
        * **Solo el 0.7% de escuelas** operan al $100\%$ bajo régimen de estabilidad legal.
        * Esta polarización es una variable crítica que el modelo puede explotar para separar escuelas fiscales directas de las delegadas o privadas.
        """)
    st.divider()

    # =========================================================================
    # 6. REPORTE EJECUTIVO Y RECOMENDACIONES
    # =========================================================================
    st.header("6. Reporte Ejecutivo y Recomendaciones")
    
    # Compilación dinámica del reporte JSON
    reporte_ejecutivo = {
        "RESUMEN_GENERAL": {
            "total_registros": len(df_upper),
            "features_numericas_analizadas": len(features_presentes),
            "porcentaje_completitud": f"{(len(df_upper.dropna(subset=features_presentes)) / len(df_upper) * 100):.1f}%"
        },
        "ESTADISTICOS_CLAVE": {
            "matricula_promedio_g1": f"{df_upper['MATRICULA_G1'].mean():.1f}" if 'MATRICULA_G1' in df_upper.columns else "N/A",
            "matricula_promedio_g5": f"{df_upper['MATRICULA_G5'].mean():.1f}" if 'MATRICULA_G5' in df_upper.columns else "N/A",
            "ratio_promedio_g1": f"{df_upper['RATIO_G1'].mean():.1f}" if 'RATIO_G1' in df_upper.columns else "N/A",
            "total_docentes_promedio": f"{df_upper['TOTAL_DOCENTES'].mean():.1f}" if 'TOTAL_DOCENTES' in df_upper.columns else "N/A"
        },
        "RECOMENDACIONES_MODELADO": {
            "Tratamiento de Nulos": "Imputar la tasa de valores nulos usando la mediana o moda por estrato.",
            "Contrato Permanente": "Considerar binarización o transformación no lineal ante la distribución bimodal.",
            "Escalamiento": "Aplicar estandarización robusta (RobustScaler) debido al volumen de outliers.",
            "Feature Engineering": "Construir interacciones complejas entre el total de docentes y las matrículas consolidadas."
        }
    }
    
    # Escritura del archivo JSON
    with open('./outputs/reporte_ejecutivo_features_numericas.json', 'w', encoding='utf-8') as f:
        json.dump(reporte_ejecutivo, f, indent=2, ensure_ascii=False)
        
    # Despliegue en Streamlit
    col_rep1, col_rep2 = st.columns(2)
    with col_rep1:
        st.markdown("### 📋 Resumen del Dataset")
        st.json(reporte_ejecutivo["RESUMEN_GENERAL"])
        st.markdown("### 📈 Promedios Críticos Calculados")
        st.json(reporte_ejecutivo["ESTADISTICOS_CLAVE"])
    with col_rep2:
        st.markdown("### 🎯 Recomendaciones de Ingeniería de Variables")
        for k, v in reporte_ejecutivo["RECOMENDACIONES_MODELADO"].items():
            st.markdown(f"* **{k}:** {v}")

    st.success(f"💾 **¡Módulo de Auditoría Completado!** Se han generado y actualizado los 5 archivos del core en la carpeta `./outputs/` con total éxito.")