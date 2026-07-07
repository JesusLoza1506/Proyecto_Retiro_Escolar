import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import os

def render_target(df):
    st.title("🎯 Análisis de Distribución del Target - Tasas de Retiro")
    st.markdown("*Este módulo audita el comportamiento de las variables objetivo (tasas de deserción escolar) desglosadas por grado académico.*")
    st.divider()
    
    # Identificación analítica de columnas idéntica a tu backend
    target_cols = [col for col in df.columns if col.startswith('TARGET_')]
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
            'Grado': target_col.split('_')[-1], 'Escuelas Evaluadas': len(data), 'Media': data.mean(),
            'Mediana': data.median(), 'Desviación Estándar': data.std(),
            'Mínimo': data.min(), 'Máximo': data.max(), 'Escuelas Retiro Cero': (data == 0).sum(),
            'Pct Retiro Cero (%)': (data == 0).mean() * 100
        })
    st.dataframe(pd.DataFrame(stats_summary).round(4), hide_index=True, use_container_width=True)
    st.divider()

    # =========================================================================
    # BLOQUE 3: Distribuciones de las Tasas de Retiro por Grado (CORREGIDO)
    # =========================================================================
    st.header("3. Distribuciones Visuales de las Tasas de Retiro")
    if not os.path.exists('./outputs'): os.makedirs('./outputs')
    
    fig_hist, axes_h = plt.subplots(2, 3, figsize=(18, 11))
    axes_h = axes_h.flatten()
    colors_h = ['#4A90E2', '#50E3C2', '#F5A623', '#E2849A', '#9B51E0']
    for i, target_col in enumerate(target_cols):
        data = df_with_targets[target_col].dropna()
        axes_h[i].hist(data, bins=50, alpha=0.75, color=colors_h[i], edgecolor='black')
        axes_h[i].axvline(data.mean(), color='#D0021B', linestyle='--', label=f'Media: {data.mean():.4f}')
        axes_h[i].axvline(data.median(), color='#F8E71C', linestyle='-', label=f'Mediana: {data.median():.4f}')
        axes_h[i].set_title(f'Distribución Tasas de Retiro - {target_col.split("_")[-1]}', weight='bold')
        axes_h[i].legend()
    axes_h[5].remove()
    plt.tight_layout()
    
    # [CORRECCIÓN] 1. Guardar en disco local
    fig_hist.savefig('./outputs/distribucion_tasas_retiro.png', dpi=300, bbox_inches='tight')
    
    # [CORRECCIÓN] 2. Renderizar visualmente en Streamlit ANTES de cerrar
    st.pyplot(fig_hist)
    
    # [CORRECCIÓN] 3. Cerrar de forma segura para limpiar memoria RAM
    plt.close(fig_hist)
    
    st.info("💡 *Los gráficos de distribución global han sido generados, mostrados y exportados con éxito.*")
    st.divider()

    # =========================================================================
    # BLOQUE 4: Análisis de Escuelas con Retiro Cero vs Positivo (CORREGIDO)
    # =========================================================================
    st.header("4. Análisis de Escuelas con Retiro Cero vs Positivo (Filtro de Deserción Activa)")
    
    grados = [col.split('_')[-1] for col in target_cols]
    pct_cero = []
    pct_positivo = []
    stats_positivo = []
    data_pos_boxplot = []

    for target_col in target_cols:
        grado = target_col.split('_')[-1]
        data = df_with_targets[target_col].dropna()
        pct_cero.append((data == 0).mean() * 100)
        pct_positivo.append((data > 0).mean() * 100)
        data_pos = data[data > 0]
        data_pos_boxplot.append(data_pos)
        
        stats_positivo.append({
            'Grado': grado, 'N_Escuelas_Positivo': len(data_pos),
            'Media_Positivo': data_pos.mean() if len(data_pos) > 0 else np.nan,
            'Mediana_Positivo': data_pos.median() if len(data_pos) > 0 else np.nan,
            'Std_Positivo': data_pos.std() if len(data_pos) > 0 else np.nan,
            'Min_Positivo': data_pos.min() if len(data_pos) > 0 else np.nan,
            'Max_Positivo': data_pos.max() if len(data_pos) > 0 else np.nan
        })

    stats_pos_df = pd.DataFrame(stats_positivo)

    fig_composite, axes = plt.subplots(2, 2, figsize=(16, 12))
    x = np.arange(len(grados))
    width = 0.35
    axes[0,0].bar(x - width/2, pct_cero, width, label='Retiro = 0%', color='lightgreen', alpha=0.8)
    axes[0,0].bar(x + width/2, pct_positivo, width, label='Retiro > 0%', color='lightcoral', alpha=0.8)
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(grados)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    for i, target_col in enumerate(target_cols):
        data_p = df_with_targets[target_col].dropna()
        data_p_filtered = data_p[data_p > 0]
        if len(data_p_filtered) > 0: axes[0,1].hist(data_p_filtered, bins=30, alpha=0.6, label=target_col.split('_')[-1], density=True)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].axis('off')
    table = axes[1,0].table(cellText=stats_pos_df.round(4).values, colLabels=stats_pos_df.columns, cellLoc='center', loc='center')
    table.scale(1.2, 1.5)

    axes[1,1].boxplot(data_pos_boxplot, labels=grados, patch_artist=True, boxprops=dict(facecolor='lightcoral', alpha=0.7), medianprops=dict(color='darkred', linewidth=2))
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    
    # [CORRECCIÓN] 1. Guardar composición de 4 cuadrantes en disco
    fig_composite.savefig('./outputs/analisis_retiro_cero_vs_positivo.png', dpi=300, bbox_inches='tight')
    
    # [CORRECCIÓN] 2. Estructuración visual en Streamlit para mostrar Gráficas + Dataframe
    col_g1, col_g2 = st.columns([1.4, 1])
    with col_g1:
        st.markdown("**📊 Cuadrante Estadístico del Backend (Matplotlib)**")
        st.pyplot(fig_composite)
    
    with col_g2:
        st.markdown("### 📋 Estadísticas Básicas (Retiro > 0%)")
        st.dataframe(stats_pos_df.round(4), hide_index=True, use_container_width=True)

    # [CORRECCIÓN] 3. Cerrar la figura una vez pintada en el navegador
    plt.close(fig_composite)

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
        grado = target_col.split('_')[-1]
        data = df_with_targets[target_col].dropna()
        data_pos = data[data > 0]
        
        reporte_final['Grado'].append(grado)
        reporte_final['N_Total'].append(len(data))
        reporte_final['Media_General'].append(data.mean())
        reporte_final['Pct_Retiro_Cero'].append((data == 0).mean() * 100)
        reporte_final['N_Retiro_Positivo'].append(len(data_pos))
        reporte_final['Media_Retiro_Positivo'].append(data_pos.mean() if len(data_pos) > 0 else 0)
        reporte_final['Max_Retiro'].append(data.max())

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
        
        g1_mean = df_with_targets['TARGET_RETIRO_G1'].mean() * 100
        g4_mean = df_with_targets['TARGET_RETIRO_G4'].mean() * 100
        g5_std = df_with_targets['TARGET_RETIRO_G5'].std()
        
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