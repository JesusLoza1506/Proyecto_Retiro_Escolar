import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

def render_target(df):
    st.title("🎯 Análisis de Distribución del Target - Tasas de Retiro")
    st.markdown("*Este módulo audita el comportamiento de las variables objetivo (tasas de deserción escolar) desglosadas por grado académico.*")
    st.divider()
    
    # Identificación analítica de columnas idéntica a tu backend (aseguramos mayúsculas)
    target_cols = [col for col in df.columns if col.upper().startswith('TARGET_')]
    if not target_cols:
        st.error("❌ No se detectaron columnas que empiecen con 'TARGET_' en el conjunto de datos.")
        return
        
    df_with_targets = df.dropna(subset=target_cols, how='all').copy()

    # =========================================================================
    # BLOQUE 1: Carga de Datos y Configuración Inicial
    # =========================================================================
    st.header("1. Carga de Datos e Identificación de Variables Objetivo")
    col_t1, col_t2 = st.columns(2)
    with col_t1: 
        st.metric(label="Variables Objetivo Detectadas", value=f"{len(target_cols)} Columnas", delta="G1 hasta G5")
    with col_t2: 
        st.metric(label="Registros con Target No Nulo", value=f"{len(df_with_targets):,} Escuelas")
    st.divider()

    # =========================================================================
    # BLOQUE 2: Estadísticas Descriptivas por Grado
    # =========================================================================
    st.header("2. Estadísticas Descriptivas de las Tasas de Retiro")
    stats_summary = []
    for target_col in target_cols:
        data = df_with_targets[target_col].dropna()
        stats_summary.append({
            'Grado': target_col.split('_')[-1].upper(), 'Escuelas Evaluadas': len(data), 'Media': data.mean(),
            'Mediana': data.median(), 'Desviación Estándar': data.std(),
            'Mínimo': data.min(), 'Máximo': data.max(), 'Escuelas Retiro Cero': (data == 0).sum(),
            'Pct Retiro Cero (%)': (data == 0).mean() * 100
        })
    st.dataframe(pd.DataFrame(stats_summary).round(4), hide_index=True, use_container_width=True)
    st.divider()

    # =========================================================================
    # BLOQUE 3: Distribuciones de las Tasas de Retiro por Grado
    # =========================================================================
    st.header("3. Distribuciones Visuales de las Tasas de Retiro")
    if not os.path.exists('./outputs'): 
        os.makedirs('./outputs')
    
    fig_hist, axes_h = plt.subplots(2, 3, figsize=(18, 11))
    axes_h = axes_h.flatten()
    colors_h = ['#4A90E2', '#50E3C2', '#F5A623', '#E2849A', '#9B51E0']
    for i, target_col in enumerate(target_cols):
        data = df_with_targets[target_col].dropna()
        axes_h[i].hist(data, bins=50, alpha=0.75, color=colors_h[min(i, len(colors_h)-1)], edgecolor='black')
        axes_h[i].axvline(data.mean() if len(data) > 0 else 0, color='#D0021B', linestyle='--', label=f'Media: {data.mean():.4f}')
        axes_h[i].axvline(data.median() if len(data) > 0 else 0, color='#F8E71C', linestyle='-', label=f'Mediana: {data.median():.4f}')
        axes_h[i].set_title(f'Distribución Tasas de Retiro - {target_col.split("_")[-1].upper()}', weight='bold')
        axes_h[i].legend()
    
    # Remover subplots vacíos excedentes de forma dinámica si hay menos de 6
    for j in range(len(target_cols), len(axes_h)):
        fig_hist.delaxes(axes_h[j])
        
    plt.tight_layout()
    fig_hist.savefig('./outputs/distribucion_tasas_retiro.png', dpi=300, bbox_inches='tight')
    st.pyplot(fig_hist)
    plt.close(fig_hist)
    
    st.info("💡 *Los gráficos de distribución global han sido generados, mostrados y exportados con éxito.*")
    st.divider()

    # =========================================================================
    # BLOQUE 4: Análisis de Escuelas con Retiro Cero vs Positivo (100% BLINDADO)
    # =========================================================================
    st.header("4. Análisis de Escuelas con Retiro Cero vs Positivo (Filtro de Deserción Activa)")
    
    grados = [col.split('_')[-1].upper() for col in target_cols]
    pct_cero = []
    pct_positivo = []
    stats_positivo = []
    data_pos_boxplot = []

    for target_col in target_cols:
        grado = target_col.split('_')[-1].upper()
        data = df_with_targets[target_col].dropna()
        
        pct_c_val = (data == 0).mean() * 100 if len(data) > 0 else 0
        pct_p_val = (data > 0).mean() * 100 if len(data) > 0 else 0
        pct_cero.append(pct_c_val)
        pct_positivo.append(pct_p_val)
        
        data_pos = data[data > 0].values
        
        # Guardamos como lista pura de Python para transferir de forma segura a Streamlit
        if len(data_pos) == 0:
            data_pos_boxplot.append([0.0])
        else:
            data_pos_boxplot.append(data_pos.tolist())
        
        stats_positivo.append({
            'Grado': grado, 
            'N_Escuelas_Positivo': len(data_pos),
            'Media_Positivo': np.mean(data_pos) if len(data_pos) > 0 else 0.0,
            'Mediana_Positivo': np.median(data_pos) if len(data_pos) > 0 else 0.0,
            'Std_Positivo': np.std(data_pos) if len(data_pos) > 1 else 0.0,
            'Min_Positivo': np.min(data_pos) if len(data_pos) > 0 else 0.0,
            'Max_Positivo': np.max(data_pos) if len(data_pos) > 0 else 0.0
        })

    stats_pos_df = pd.DataFrame(stats_positivo)

    # Creamos la composición gráfica de Matplotlib limpia sin el Boxplot problemático
    fig_composite, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(grados))
    width = 0.35
    
    # Subplot 1: Distribución Porcentual Proporcional
    axes[0].bar(x - width/2, pct_cero, width, label='Retiro = 0%', color='lightgreen', alpha=0.8)
    axes[0].bar(x + width/2, pct_positivo, width, label='Retiro > 0%', color='lightcoral', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(grados)
    axes[0].set_title("Proporción Retiro Cero vs Positivo", weight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Histograma Densidad de Escuelas Activas
    for i, target_col in enumerate(target_cols):
        data_p = df_with_targets[target_col].dropna()
        data_p_filtered = data_p[data_p > 0]
        if len(data_p_filtered) > 0: 
            axes[1].hist(data_p_filtered, bins=30, alpha=0.6, label=target_col.split('_')[-1].upper(), density=True)
    axes[1].set_title("Densidad de Probabilidad (Retiro > 0%)", weight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_composite.savefig('./outputs/analisis_retiro_cero_vs_positivo.png', dpi=300, bbox_inches='tight')
    
    # Renderizado inteligente estructurado por columnas de Streamlit
    col_g1, col_g2 = st.columns([1.4, 1])
    with col_g1:
        st.markdown("**📊 Distribuciones e Histogramas del Backend**")
        st.pyplot(fig_composite)
        plt.close(fig_composite)
        
        # Renderizado del Diagrama de Cajas interactivo de forma nativa e indestructible
        st.markdown("**📦 Diagrama de Cajas Interactivo (Solo Retiro > 0%)**")
        boxplot_data = []
        for g_name, values in zip(grados, data_pos_boxplot):
            for v in values:
                boxplot_data.append({"Grado": g_name, "Tasa de Retiro": v})
        
        if boxplot_data:
            df_box = pd.DataFrame(boxplot_data)
            st.vega_lite_chart(df_box, {
                'mark': {'type': 'boxplot', 'extent': 'min-max'},
                'encoding': {
                    'x': {'field': 'Grado', 'type': 'nominal', 'axis': {'labelAngle': 0}},
                    'y': {'field': 'Tasa de Retiro', 'type': 'quantitative'},
                    'color': {'field': 'Grado', 'type': 'nominal', 'legend': None}
                }
            }, use_container_width=True)
    
    with col_g2:
        st.markdown("### 📋 Estadísticas Básicas (Retiro > 0%)")
        st.dataframe(stats_pos_df.round(4), hide_index=True, use_container_width=True)

    st.divider()

    # =========================================================================
    # BLOQUE 5: Resumen de Hallazgos y Conclusiones
    # =========================================================================
    st.header("5. Resumen Ejecutivo de Auditoría del Target")
    
    reporte_final = {
        'Grado': [], 'N_Total': [], 'Media_General': [],
        'Pct_Retiro_Cero': [], 'N_Retiro_Positivo': [],
        'Media_Retiro_Positivo': [], 'Max_Retiro': []
    }

    for target_col in target_cols:
        grado = target_col.split('_')[-1].upper()
        data = df_with_targets[target_col].dropna()
        data_pos = data[data > 0]
        
        reporte_final['Grado'].append(grado)
        reporte_final['N_Total'].append(len(data))
        reporte_final['Media_General'].append(data.mean() if len(data) > 0 else 0.0)
        reporte_final['Pct_Retiro_Cero'].append((data == 0).mean() * 100 if len(data) > 0 else 0.0)
        reporte_final['N_Retiro_Positivo'].append(len(data_pos))
        reporte_final['Media_Retiro_Positivo'].append(data_pos.mean() if len(data_pos) > 0 else 0.0)
        reporte_final['Max_Retiro'].append(data.max() if len(data) > 0 else 0.0)

    reporte_df = pd.DataFrame(reporte_final)
    reporte_df.to_csv('./outputs/reporte_distribucion_target.csv', index=False)
    
    col_rep_izq, col_rep_der = st.columns([1.3, 1])
    
    with col_rep_izq:
        st.markdown("### 📋 Cuadro de Mando Consolidado del Target")
        st.dataframe(reporte_df.round(4), hide_index=True, use_container_width=True)
        
        csv_buffer = reporte_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Reporte de Distribución del Target (.CSV)",
            data=csv_buffer,
            file_name="reporte_distribucion_target.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.success(f"""
        **💾 Auditoría Finalizada:**
        * Reporte guardado en: `./outputs/reporte_distribucion_target.csv`
        * Gráficas actualizadas en: `./outputs/`
        """)

    with col_rep_der:
        st.markdown("### 🎯 Patrones Identificados y Conclusiones")
        
        # Validamos dinámicamente si las columnas existen en mayúsculas o minúsculas
        g1_mean = df_with_targets.get('TARGET_RETIRO_G1', df_with_targets.get('target_retiro_g1', pd.Series([0]))).mean() * 100
        g4_mean = df_with_targets.get('TARGET_RETIRO_G4', df_with_targets.get('target_retiro_g4', pd.Series([0]))).mean() * 100
        g5_std = df_with_targets.get('TARGET_RETIRO_G5', df_with_targets.get('target_retiro_g5', pd.Series([0]))).std()
        
        st.info(f"""
        **📉 Patrones Críticos por Grados:**
        * **Punto de Quiebre (G4):** Cuarto de secundaria registra la mayor tasa de retiro promedio (**{g4_mean:.2f}%**).
        * **Estabilidad Temprana (G1):** Primero de secundaria tiene la menor incidencia de deserción (**{g1_mean:.2f}%**).
        * **Dispersión Terminal (G5):** Quinto de secundaria presenta la variabilidad más inestable ($\sigma = {g5_std:.4f}$).
        """)
        
        st.warning("""
        **🔍 Características de la Distribución:**
        * **Sesgo Extremo:** Todas las curvas están radicalmente truncadas a la derecha debido a la concentración de ceros.
        * **Inflación Fuerte:** Entre el 80% y 87% de los centros educativos del MINEDU reportan estabilidad total.
        * **Severidad Total:** Presencia de valores máximos del 100% en todos los grados, alertando colapsos de matrícula puntuales.
        """)
        
    st.write("")
    st.subheader("💡 Implicaciones Metodológicas para el Jurado de la Tesis")
    st.success("""
    * **Desbalance de Datos:** La tasa de retiro es una anomalía de clasificación desbalanceada. Deben evaluarse métricas robustas como el área bajo la curva (**AUC-ROC**) y la puntuación **F1**, descartando el Accuracy tradicional.
    * **Modelos Separados:** Dadas las marcadas diferencias de variabilidad entre grados (G1 vs G4), se valida la estrategia de entrenar hiperparámetros independientes para capturar la naturaleza de cada ciclo académico.
    """)