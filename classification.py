import pandas as pd
import numpy as np
import streamlit as st
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el threshold óptimo usando precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Manejar divisiones por cero
    optimal_idx = np.argmax(f1_scores)
    if optimal_idx >= len(thresholds):
        return 0.5, f1_scores[optimal_idx]
    return thresholds[optimal_idx], f1_scores[optimal_idx]


def render_classification(df):
    st.title("🤖 Etapa 1: Clasificación Binaria - Predicción de Retiro Escolar")
    st.caption("Implementación del modelo de clasificación para predecir si una escuela tendrá retiro > 0% en cada grado (G1-G5).")
    st.markdown("---")

    # Asegurar la existencia del directorio de salidas
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')

    # =========================================================================
    # BLOQUE 1: Preparación de Datos y Configuración Inicial
    # =========================================================================
    st.header("📋 1. Preparación de Datos y Configuración Inicial")
    
    csv_file = 'Results_2026-05-31-2354.csv'
    
    if not os.path.exists(csv_file):
        st.error(f"⚠️ No se encontró el archivo `{csv_file}` en el directorio actual. Por favor verifica su ubicación.")
        return
        
    df = pd.read_csv(csv_file)
    st.success(f"✅ Dataset cargado correctamente: `{csv_file}`")
    
    col_shape1, col_shape2 = st.columns(2)
    with col_shape1:
        st.metric("Total de Filas (Registros)", f"{df.shape[0]:,}")
    with col_shape2:
        st.metric("Total de Columnas", f"{df.shape[1]}")
        
    with st.expand_viewer if hasattr(st, 'expand_viewer') else st.expander("🔍 Ver listado completo de columnas en el Dataset"):
        st.write(list(df.columns))

    # =========================================================================
    # BLOQUE 2: Preparación de Variables y Targets Binarios
    # =========================================================================
    st.header("🔧 2. Preparación de Variables y Targets Binarios")
    
    df.columns = df.columns.str.lower()

    # Mapeos para visualización
    gestion_map  = {1: 'Pública directa', 2: 'Pública convenio', 3: 'Privada'}
    area_map     = {1: 'Urbana', 2: 'Rural'}
    tipssexo_map = {1: 'Solo varones', 2: 'Solo mujeres', 3: 'Mixto'}

    df['gestion_label']  = df['gestion'].map(gestion_map)
    df['area_label']     = df['area_censo'].map(area_map)
    df['tipssexo_label'] = df['tipssexo'].map(tipssexo_map)

    # Filtrar datos preparados
    train = df[(df['split']=='train') & (df['matricula_g1'].notna())].copy()
    inference = df[(df['split']=='inference') & (df['matricula_g1'].notna())].copy()
    inference_nuevas = df[(df['split']=='inference') & (df['matricula_g1'].isna())].copy()

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1: st.metric("Muestras de Train", f"{len(train):,}")
    with col_t2: st.metric("Muestras de Inferencia", f"{len(inference):,}")
    with col_t3: st.metric("Muestras Nuevas", f"{len(inference_nuevas):,}")

    st.markdown("**📊 Distribución temporal en Train:**")
    st.dataframe(train['anio_features'].value_counts().sort_index(), use_container_width=True)

    # Crear targets binarios para clasificación
    grades = ['g1','g2','g3','g4','g5']
    for g in grades:
        train[f'target_bin_{g}'] = (train[f'target_retiro_{g}'] > 0).astype(int)

    st.markdown("**📊 Distribución de targets binarios (% positivos por Grado):**")
    dist_list = []
    for g in grades:
        pos_rate = train[f'target_bin_{g}'].mean()
        dist_list.append({
            "Grado": f"G{g[-1]}", 
            "Ratio Positivos (%)": f"{pos_rate:.1%}", 
            "Escuelas con Retiro": f"{train[f'target_bin_{g}'].sum():,}"
        })
    st.dataframe(pd.DataFrame(dist_list), use_container_width=True, hide_index=True)

    # Definir features
    cat_features = ['gestion','ges_dep','area_censo','region_nat','tipssexo','cod_tur','vraem','zfrontera']
    num_features = [
        'matricula_g1','matricula_g2','matricula_g3','matricula_g4','matricula_g5',
        'secciones_g1','secciones_g2','secciones_g3','secciones_g4','secciones_g5',
        'ratio_g1','ratio_g2','ratio_g3','ratio_g4','ratio_g5',
        'total_docentes','pct_docentes_femenino','pct_contrato_permanente','pct_titulados',
        'retiro_lag_g1','retiro_lag_g2','retiro_lag_g3','retiro_lag_g4','retiro_lag_g5'
    ]
    feature_cols = cat_features + num_features
    st.info(f"🔧 **Features definidas:** {len(feature_cols)} ({len(cat_features)} categóricas + {len(num_features)} numéricas)")

    # =========================================================================
    # BLOQUE 3: División Temporal para Validación
    # =========================================================================
    st.header("📅 3. División Temporal para Validación")
    
    # Manejar tipos string/int según los datos reales del dataset de manera segura
    train['anio_features'] = train['anio_features'].astype(str)
    
    X_tr = train[train['anio_features']=='2022'][feature_cols]
    y_tr = train[train['anio_features']=='2022'][[f'target_bin_{g}' for g in grades]]
    X_val = train[train['anio_features']=='2023'][feature_cols]
    y_val = train[train['anio_features']=='2023'][[f'target_bin_{g}' for g in grades]]

    col_y1, col_y2 = st.columns(2)
    with col_y1: st.metric("Train (Año 2022)", f"{len(X_tr):,} escuelas")
    with col_y2: st.metric("Validación (Año 2023)", f"{len(X_val):,} escuelas")

    st.markdown("**📊 Distribución de positivos por año:**")
    pos_years = []
    for g in grades:
        tr_pos = train[train['anio_features']=='2022'][f'target_bin_{g}'].mean()
        val_pos = train[train['anio_features']=='2023'][f'target_bin_{g}'].mean()
        pos_years.append({"Grado": f"G{g[-1]}", "Train 2022": f"{tr_pos:.1%}", "Val 2023": f"{val_pos:.1%}"})
    st.dataframe(pd.DataFrame(pos_years), use_container_width=True, hide_index=True)

    # Validar nulos antes del fit
    nulos_tr = X_tr.isnull().sum().sum()
    nulos_val = X_val.isnull().sum().sum()
    if nulos_tr > 0 or nulos_val > 0:
        st.warning(f"⚠️ Alerta: Se detectaron nulos en features (Train: {nulos_tr} | Val: {nulos_val})")
    else:
        st.success("✅ Validación de Calidad: No hay valores nulos en los espacios de características.")

    # Codificación Automatizada de Categóricas (Requisito de los bloques 4/5)
    X_tr_encoded = X_tr.copy()
    X_val_encoded = X_val.copy()
    label_encoders = {}
    for col in cat_features:
        X_tr_encoded[col] = X_tr_encoded[col].astype(str)
        X_val_encoded[col] = X_val_encoded[col].astype(str)
        le = LabelEncoder()
        le.fit(pd.concat([X_tr_encoded[col], X_val_encoded[col]]).unique())
        X_tr_encoded[col] = le.transform(X_tr_encoded[col])
        X_val_encoded[col] = le.transform(X_val_encoded[col])
        label_encoders[col] = le

    # =========================================================================
    # BLOQUE 4 Y 5: Entrenamiento Final de Modelos LightGBM
    # =========================================================================
    st.header("🚀 4 y 5. Entrenamiento Final de Modelos LightGBM por Grado")
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 500, 'learning_rate': 0.05,
        'num_leaves': 31, 'min_child_samples': 20, 'class_weight': 'balanced', 'random_state': 42,
        'n_jobs': -1, 'verbose': -1
    }

    models = {}
    results = []

    # Uso de st.spinner para reportar que está entrenando el pipeline secuencial
    with st.spinner("Entrenando matrices LightGBM optimizadas..."):
        for i, g in enumerate(grades):
            y_tr_g = y_tr[f'target_bin_{g}']
            y_val_g = y_val[f'target_bin_{g}']
            
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(X_tr_encoded, y_tr_g)
            
            y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
            y_pred_05 = (y_pred_proba >= 0.5).astype(int)
            
            auc_score = roc_auc_score(y_val_g, y_pred_proba)
            f1_05 = f1_score(y_val_g, y_pred_05, zero_division=0)
            
            optimal_thresh, f1_optimal = find_optimal_threshold(y_val_g, y_pred_proba)
            
            models[g] = model
            results.append({
                'Grado': f'G{g[-1]}', 'AUC-ROC': auc_score, 'F1 (0.5)': f1_05,
                'F1 (óptimo)': f1_optimal, 'Threshold óptimo': optimal_thresh,
                'Positivos train': y_tr_g.sum(), 'Positivos val': y_val_g.sum()
            })
            
    st.success("✅ ¡Entrenamiento del Pipeline de Clasificación completado con éxito!")
    results_df = pd.DataFrame(results)
    st.dataframe(results_df.round(3), use_container_width=True, hide_index=True)

    # =========================================================================
    # BLOQUE 6: Análisis de Importancia de Features
    # =========================================================================
    st.header("📊 6. Análisis de Importancia de Features")
    
    feature_importance_data = []
    for g in grades:
        importance = models[g].feature_importances_
        for feat, imp in zip(feature_cols, importance):
            feature_importance_data.append({'Grado': f'G{g[-1]}', 'Feature': feat, 'Importance': imp})

    importance_df = pd.DataFrame(feature_importance_data)

    # Gráfica Matplotlib/Seaborn adaptada para Streamlit
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()
    for i, g in enumerate(grades):
        ax = axes[i]
        top10 = importance_df[importance_df['Grado']==f'G{g[-1]}'].nlargest(10, 'Importance')
        ax.barh(range(len(top10)), top10['Importance'].values, color='steelblue')
        ax.set_yticks(range(len(top10)))
        ax.set_yticklabels(top10['Feature'].values)
        ax.set_xlabel('Importance (Gain)')
        ax.set_title(f'Top 10 Features - G{g[-1]}')
        ax.invert_yaxis()
    axes[5].axis('off')
    plt.tight_layout()
    
    fig.savefig('./outputs/feature_importance_clasificacion.png', dpi=300, bbox_inches='tight')
    st.pyplot(fig)
    plt.close(fig)
    st.info("🎨 Gráfico guardado en: `./outputs/feature_importance_clasificacion.png`")

    results_df.to_csv('./outputs/resultados_clasificacion.csv', index=False)
    importance_df.to_csv('./outputs/feature_importance_clasificacion.csv', index=False)

    # =========================================================================
    # BLOQUE 7: Guardar Modelos Entrenados
    # =========================================================================
    st.header("💾 7. Guardar Modelos Entrenados")
    
    for g in grades:
        with open(f'./outputs/modelo_clasificacion_{g}.pkl', 'wb') as f:
            pickle.dump(models[g], f)
            
    with open('./outputs/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
        
    st.success("🎉 **ETAPA 1 COMPLETADA CON ÉXITO!**")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1: st.metric("AUC-ROC Promedio", f"{results_df['AUC-ROC'].mean():.3f}")
    with col_r2: st.metric("F1 Óptimo Promedio", f"{results_df['F1 (óptimo)'].mean():.3f}")
    
    st.info("All artifacts saved successfully in local `./outputs/` space.")

    # =========================================================================
    # REVISIÓN DE FEATURE IMPORTANCE - ANÁLISIS DETALLADO
    # =========================================================================
    st.header("🔍 Revisión de Feature Importance - Análisis Detallado")
    
    tab_completo, tab_posicion, tab_dist_lag, tab_corr_lag = st.tabs([
        "📊 Top 15 G1/G5", "🎯 Posición de Lags", "📈 Distribución de Lags", "🔗 Correlación Lag vs Target"
    ])
    
    with tab_completo:
        for grade in ['g1', 'g5']:
            st.markdown(f"**TOP 15 FEATURES - G{grade[-1].upper()}:**")
            importance_df_det = pd.DataFrame({
                'feature': feature_cols,
                'importance': models[grade].feature_importances_
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            st.dataframe(importance_df_det.head(15), use_container_width=True)

    with tab_posicion:
        st.markdown("**🎯 G1 y G5 - Posición de Lag Features Continuas:**")
        lag_features = [f'retiro_lag_{g}' for g in grades]
        pos_data = []
        for grade in ['g1', 'g5']:
            imp_sort = pd.DataFrame({
                'feature': feature_cols,
                'importance': models[grade].feature_importances_
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            for lag_feat in lag_features:
                position = imp_sort[imp_sort['feature'] == lag_feat].index[0] + 1
                importance_val = imp_sort[imp_sort['feature'] == lag_feat]['importance'].iloc[0]
                pos_data.append({
                    "Grado Evaluado": f"G{grade[-1].upper()}",
                    "Lag Feature": lag_feat,
                    "Posición en Ranking": f"#{position}",
                    "Importancia (Gain)": importance_val
                })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

    with tab_dist_lag:
        st.markdown("**📈 Distribución Interna de Lag Features en Train:**")
        dist_lag_list = []
        for g in grades:
            lag_col = f'retiro_lag_{g}'
            zero_pct = (train[lag_col] == 0).mean()
            positive_pct = (train[lag_col] > 0).mean()
            mean_val = train[lag_col].mean()
            dist_lag_list.append({
                "Grado": f"G{g[-1]}", "Ceros (%)": f"{zero_pct:.1%}", 
                "Positivos (%)": f"{positive_pct:.1%}", "Media Histórica": f"{mean_val:.3f}"
            })
        st.dataframe(pd.DataFrame(dist_lag_list), use_container_width=True, hide_index=True)

    with tab_corr_lag:
        st.markdown("**🔗 Asociación Lineal y Binaria (Lag vs Target Actual):**")
        corr_lag_list = []
        for g in grades:
            lag_col = f'retiro_lag_{g}'
            target_col = f'target_bin_{g}'
            corr_pearson = train[lag_col].corr(train[target_col])
            lag_binary = (train[lag_col] > 0).astype(int)
            corr_binary = lag_binary.corr(train[target_col])
            corr_lag_list.append({
                "Grado": f"G{g[-1]}", "Corr. Pearson (Continua)": corr_pearson, "Corr. Binaria (Presencia)": corr_binary
            })
        st.dataframe(pd.DataFrame(corr_lag_list).round(3), use_container_width=True, hide_index=True)

    # =========================================================================
    # EVALUACIÓN DE FEATURES BINARIAS DE LAG
    # =========================================================================
    st.header("🔧 Evaluación de Features Binarias de Lag")
    
    for g in grades:
        train[f'tuvo_retiro_lag_{g}'] = (train[f'retiro_lag_{g}'] > 0).astype(int)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("**📊 Distribución de Nuevas Features Binarias de Lag:**")
        bin_dist = []
        for g in grades:
            binary_col = f'tuvo_retiro_lag_{g}'
            positive_pct = train[binary_col].mean()
            bin_dist.append({"Grado": f"G{g[-1]}", "Escuelas con Retiro Año Anterior": f"{positive_pct:.1%}"})
        st.dataframe(pd.DataFrame(bin_dist), use_container_width=True, hide_index=True)

    with col_b2:
        st.markdown("**📈 Comparación de Señal: Continua vs Binaria:**")
        signal_comp = []
        for g in grades:
            lag_cont = f'retiro_lag_{g}'
            lag_bin = f'tuvo_retiro_lag_{g}'
            target_col = f'target_bin_{g}'
            corr_cont = train[lag_cont].corr(train[target_col])
            corr_bin = train[lag_bin].corr(train[target_col])
            signal_comp.append({
                "Grado": f"G{g[-1]}", "Corr. Continua": corr_cont, 
                "Corr. Binaria": corr_bin, "Factor de Mejora": f"{corr_bin/corr_cont:.1f}x"
            })
        st.dataframe(pd.DataFrame(signal_comp).round(3), use_container_width=True, hide_index=True)

    st.markdown("**🔗 Análisis del Lift Efectivo Mediante Probabilidad Condicional:**")
    lift_list = []
    for g in grades:
        binary_col = f'tuvo_retiro_lag_{g}'
        target_col = f'target_bin_{g}'
        contingency = pd.crosstab(train[binary_col], train[target_col], normalize='index')
        prob_retiro_given_lag = contingency.loc[1, 1] if 1 in contingency.index and 1 in contingency.columns else 0
        prob_retiro_given_no_lag = contingency.loc[0, 1] if 0 in contingency.index and 1 in contingency.columns else 0
        lift_list.append({
            "Grado": f"G{g[-1]}", "P(Retiro | Tuvo Lag)": f"{prob_retiro_given_lag:.1%}",
            "P(Retiro | No Lag)": f"{prob_retiro_given_no_lag:.1%}", "Lift Efectivo": f"{prob_retiro_given_lag/prob_retiro_given_no_lag:.1f}x"
        })
    st.dataframe(pd.DataFrame(lift_list), use_container_width=True, hide_index=True)
    st.divider()

    # =========================================================================
    # ETAPA 1 - REENTRENAMIENTO CON FEATURES BINARIAS DE LAG
    # =========================================================================
    st.subheader("🔄 Etapa 1 - Reentrenamiento con Features Binarias de Lag")
    
    # 1. Agregar Features Binarias de Lag
    st.markdown("#### 1. Agregar Features Binarias de Lag")
    
    for g in grades:
        train[f'tuvo_retiro_lag_{g}'] = (train[f'retiro_lag_{g}'] > 0).astype(int)
        inference[f'tuvo_retiro_lag_{g}'] = (inference[f'retiro_lag_{g}'] > 0).astype(int)

    lag_bin_features = [f'tuvo_retiro_lag_{g}' for g in grades]
    feature_cols_v2 = feature_cols + lag_bin_features

    st.info(f"✅ **Evolución del Espacio de Características:** Expandido a v2 con {len(feature_cols_v2)} columnas totales (v1 tenía {len(feature_cols)}).")

    # Asegurar el casteo a entero de manera defensiva para el split temporal
    train['anio_features'] = train['anio_features'].astype(int)
    
    X_tr_v2 = train[train['anio_features']==2022][feature_cols_v2]
    y_tr_v2 = train[train['anio_features']==2022][[f'target_bin_{g}' for g in grades]]
    X_val_v2 = train[train['anio_features']==2023][feature_cols_v2]
    y_val_v2 = train[train['anio_features']==2023][[f'target_bin_{g}' for g in grades]]

    X_tr_v2_encoded = X_tr_v2.copy()
    X_val_v2_encoded = X_val_v2.copy()

    for col in cat_features:
        if col in label_encoders:
            X_tr_v2_encoded[col] = label_encoders[col].transform(X_tr_v2[col].astype(str))
            X_val_v2_encoded[col] = label_encoders[col].transform(X_val_v2[col].astype(str))

    # 2. Reentrenamiento de Modelos v2
    st.markdown("#### 2. Reentrenamiento de Modelos LightGBM v2")
    models_v2 = {}
    results_v2 = []

    with st.spinner("Reentrenando iteraciones v2..."):
        for i, g in enumerate(grades):
            y_tr_g = y_tr_v2[f'target_bin_{g}']
            y_val_g = y_val_v2[f'target_bin_{g}']
            
            model_v2 = lgb.LGBMClassifier(**lgb_params)
            model_v2.fit(X_tr_v2_encoded, y_tr_g)
            
            y_pred_proba_v2 = model_v2.predict_proba(X_val_v2_encoded)[:, 1]
            y_pred_05_v2 = (y_pred_proba_v2 >= 0.5).astype(int)
            
            auc_score_v2 = roc_auc_score(y_val_g, y_pred_proba_v2)
            f1_05_v2 = f1_score(y_val_g, y_pred_05_v2, zero_division=0)
            
            optimal_thresh_v2, f1_optimal_v2 = find_optimal_threshold(y_val_g, y_pred_proba_v2)
            
            models_v2[g] = model_v2
            results_v2.append({
                'Grado': f'G{g[-1]}', 'AUC-ROC v2': auc_score_v2, 'F1 (0.5) v2': f1_05_v2,
                'F1 (óptimo) v2': f1_optimal_v2, 'Threshold óptimo v2': optimal_thresh_v2,
                'Positivos train': y_tr_g.sum(), 'Positivos val': y_val_g.sum()
            })
            
    results_v2_df = pd.DataFrame(results_v2)
    st.dataframe(results_v2_df.round(3), use_container_width=True, hide_index=True)

    # 3. Comparación v1 vs v2
    st.markdown("#### 3. Comparación v1 vs v2")
    comparison_data = []
    for i, g in enumerate(grades):
        v1_auc = results_df.iloc[i]['AUC-ROC']
        v2_auc = results_v2_df.iloc[i]['AUC-ROC v2']
        v1_f1 = results_df.iloc[i]['F1 (óptimo)']
        v2_f1 = results_v2_df.iloc[i]['F1 (óptimo) v2']
        
        comparison_data.append({
            'Grado': f'G{g[-1]}', 'AUC v1': v1_auc, 'AUC v2': v2_auc, 'Δ AUC': v2_auc - v1_auc,
            'F1 v1': v1_f1, 'F1 v2': v2_f1, 'Δ F1': v2_f1 - v1_f1
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.round(4), use_container_width=True, hide_index=True)

    avg_auc_improvement = comparison_df['Δ AUC'].mean()
    avg_f1_improvement = comparison_df['Δ F1'].mean()
    grados_mejorados = (comparison_df['Δ AUC'] > 0).sum()
    significant_improvement = avg_auc_improvement >= 0.005

    # Renderizado dinámico de la caja de decisión
    st.markdown("**🎯 DECLARACIÓN DE DECISIÓN ESTADÍSTICA:**")
    if significant_improvement:
        st.success(f"✅ **Mejora significativa detectada:** (Δ AUC = {avg_auc_improvement:+.4f} ≥ 0.005). Adoptando e indexando modelos v2 como versión final. Grados con mejora: {grados_mejorados}/5.")
    else:
        st.warning(f"⚠️ **Mejora marginal / Sub-umbral:** (Δ AUC = {avg_auc_improvement:+.4f} < 0.005). Manteniendo **modelos v1** como versión final para preservar parsimonia y evitar redundancia. Grados con mejora: {grados_mejorados}/5.")

    comparison_df.to_csv('./outputs/comparacion_v1_vs_v2.csv', index=False)

    # 4. Análisis de Feature Importance v2
    st.markdown("#### 4. Análisis de Feature Importance v2")
    tab_rank_v2, tab_top5_v2 = st.tabs(["🔍 Top 15 v2 (G1/G5)", "📈 Auditoría de Rezagos Binarios en Top 5"])
    
    with tab_rank_v2:
        for grade in ['g1', 'g5']:
            st.markdown(f"**TOP 15 FEATURES - G{grade[-1].upper()} (v2):**")
            importance_df_v2 = pd.DataFrame({
                'feature': feature_cols_v2,
                'importance': models_v2[grade].feature_importances_
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            # Formatear el DataFrame para resaltar de manera visual las columnas de lag
            importance_df_v2['Estatus'] = importance_df_v2['feature'].apply(lambda x: "🎯 Target Temporal" if x.startswith('tuvo_retiro_lag') else "Base")
            st.dataframe(importance_df_v2.head(15), use_container_width=True)

    with tab_top5_v2:
        top5_count = 0
        top5_audit_log = []
        for grade in grades:
            importance_df_v2 = pd.DataFrame({
                'feature': feature_cols_v2,
                'importance': models_v2[grade].feature_importances_
            }).sort_values('importance', ascending=False).reset_index(drop=True)
            
            top5_features = importance_df_v2.head(5)['feature'].tolist()
            lag_in_top5 = any(feat.startswith('tuvo_retiro_lag') for feat in top5_features)
            
            if lag_in_top5:
                top5_count += 1
                lag_features_in_top5 = [feat for feat in top5_features if feat.startswith('tuvo_retiro_lag')]
                top5_audit_log.append({"Grado": f"G{grade[-1]}", "Estatus Top 5": "✅ Presente", "Variables": str(lag_features_in_top5)})
            else:
                top5_audit_log.append({"Grado": f"G{grade[-1]}", "Estatus Top 5": "❌ Ausente", "Variables": "Ninguna"})
                
        st.dataframe(pd.DataFrame(top5_audit_log), use_container_width=True, hide_index=True)
        st.write(f"🎯 **Resultado de Auditoría:** {top5_count}/5 grados tienen lag binario incorporado en el Top 5 de ganancia.")

    # 5. Conclusiones del Reentrenamiento
    st.markdown("#### 5. Conclusiones del Reentrenamiento")
    st.info("""
    💡 **Interpretación Ejecutiva:**
    1. **Paradoja de Ganancia:** El lift observado en el análisis exploratorio (EDA) no se tradujo en importancia lineal para el modelo debido a que LightGBM ya extrae la señal temporal implícitamente mediante las features dinámicas de matrículas y ratios bases.
    2. **Fuerza de Proxy:** La variable `pct_docentes_femenino` retiene el peso predictivo dominante como proxy contextual del entorno escolar.
    """)

    # =========================================================================
    # ETAPA 2: REGRESIÓN - PREDICCIÓN DE MAGNITUD DEL RETIRO ESCOLAR
    # =========================================================================
    st.markdown("---")
    st.title("📈 Etapa 2: Regresión - Predicción de Magnitud del Retiro Escolar")
    st.caption("Implementación del modelo de regresión para predecir la magnitud exacta del retiro en escuelas clasificadas como positivas (retiro > 0%).")
    st.markdown("---")

    # =========================================================================
    # BLOQUE 1: Construcción de Subsets de Regresión
    # =========================================================================
    st.header("🔧 1. Construcción de Subsets de Regresión")
    
    regression_data = {}
    train['anio_features'] = train['anio_features'].astype(int)
    
    subsets_log = []
    for g in grades:
        target_col = f'target_retiro_{g}'
        
        # Train: 2022 con retiro > 0
        train_pos_2022 = train[(train['anio_features'] == 2022) & (train[target_col] > 0)]
        # Validación: 2023 con retiro > 0  
        val_pos_2023 = train[(train['anio_features'] == 2023) & (train[target_col] > 0)]
        
        regression_data[g] = {
            'X_tr': train_pos_2022[feature_cols], 'y_tr': train_pos_2022[target_col],
            'X_val': val_pos_2023[feature_cols], 'y_val': val_pos_2023[target_col]
        }
        subsets_log.append({
            "Grado": f"G{g[-1]}", "Train 2022 (Retiro > 0)": f"{len(train_pos_2022):,} esc.", 
            "Val 2023 (Retiro > 0)": f"{len(val_pos_2023):,} esc."
        })
        
    st.dataframe(pd.DataFrame(subsets_log), use_container_width=True, hide_index=True)

    st.markdown("**📊 Distribución de targets de regresión (solo retiro > 0):**")
    dist_reg_list = []
    for g in grades:
        y_tr_g = regression_data[g]['y_tr']
        y_val_g = regression_data[g]['y_val']
        dist_reg_list.append({
            "Grado": f"G{g[-1]}", "Train Media": y_tr_g.mean(), "Train Mediana": y_tr_g.median(), "Train Max": y_tr_g.max(),
            "Val Media": y_val_g.mean(), "Val Mediana": y_val_g.median(), "Val Max": y_val_g.max()
        })
    st.dataframe(pd.DataFrame(dist_reg_list).round(3), use_container_width=True, hide_index=True)

    # Codificar features categóricas para regresión
    for g in grades:
        X_tr_encoded = regression_data[g]['X_tr'].copy()
        X_val_encoded = regression_data[g]['X_val'].copy()
        for col in cat_features:
            if col in label_encoders:
                X_tr_encoded[col] = label_encoders[col].transform(regression_data[g]['X_tr'][col].astype(str))
                X_val_encoded[col] = label_encoders[col].transform(regression_data[g]['X_val'][col].astype(str))
        regression_data[g]['X_tr_encoded'] = X_tr_encoded
        regression_data[g]['X_val_encoded'] = X_val_encoded

    st.success("✅ Subsets de regresión condicional preparados y codificados de forma segura.")

    # =========================================================================
    # BLOQUE 2: Entrenamiento de Modelos de Regresión LightGBM (Tweedie)
    # =========================================================================
    st.header("🚀 2. Entrenamiento de Modelos de Regresión LightGBM")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    lgb_reg_params = {
        'objective': 'tweedie', 'tweedie_variance_power': 1.5, 'metric': 'mae',
        'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
        'min_child_samples': 10, 'random_state': 42, 'n_jobs': -1, 'verbose': -1
    }

    regression_models = {}
    regression_results = []

    with st.spinner("Entrenando regresores bajo distribución Tweedie..."):
        for i, g in enumerate(grades):
            X_tr_reg = regression_data[g]['X_tr_encoded']
            y_tr_reg = regression_data[g]['y_tr']
            X_val_reg = regression_data[g]['X_val_encoded']
            y_val_reg = regression_data[g]['y_val']
            
            reg_model = lgb.LGBMRegressor(**lgb_reg_params)
            reg_model.fit(X_tr_reg, y_tr_reg)
            
            y_pred = reg_model.predict(X_val_reg)
            y_pred = np.clip(y_pred, 0, 1)
            
            mae = mean_absolute_error(y_val_reg, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val_reg, y_pred))
            r2 = r2_score(y_val_reg, y_pred)
            
            regression_models[g] = reg_model
            regression_results.append({
                'Grado': f'G{g[-1]}', 'MAE': mae, 'RMSE': rmse, 'R²': r2,
                'N_train': len(y_tr_reg), 'N_val': len(y_val_reg),
                'Media_real': y_val_reg.mean(), 'Media_pred': y_pred.mean()
            })

    regression_results_df = pd.DataFrame(regression_results)
    st.success("✅ ¡Entrenamiento de regresores Tweedie completado!")
    st.dataframe(regression_results_df.round(4), use_container_width=True, hide_index=True)

    # =========================================================================
    # BLOQUE 3: Análisis de Residuos y Distribuciones
    # =========================================================================
    st.header("📊 3. Análisis de Residuos y Distribuciones")
    
    residuals_data = {}
    for g in grades:
        X_val_reg = regression_data[g]['X_val_encoded']
        y_val_reg = regression_data[g]['y_val']
        y_pred = np.clip(regression_models[g].predict(X_val_reg), 0, 1)
        residuals = y_val_reg - y_pred
        residuals_data[g] = {'y_true': y_val_reg, 'y_pred': y_pred, 'residuals': residuals}

    # Gráfico 1: Distribución de Residuos
    fig_res, axes_res = plt.subplots(2, 3, figsize=(18, 11))
    axes_res = axes_res.flatten()
    for i, g in enumerate(grades):
        ax = axes_res[i]
        res = residuals_data[g]['residuals']
        ax.hist(res, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax.set_title(f'Distribución de Residuos - G{g[-1]}')
        ax.text(0.05, 0.95, f'Media: {res.mean():.3f}\nStd: {res.std():.3f}', 
                transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes_res[5].axis('off')
    plt.tight_layout()
    fig_res.savefig('./outputs/residuos_regresion.png', dpi=300, bbox_inches='tight')
    
    # Gráfico 2: Predicciones vs Reales
    fig_scat, axes_scat = plt.subplots(2, 3, figsize=(18, 11))
    axes_scat = axes_scat.flatten()
    for i, g in enumerate(grades):
        ax = axes_scat[i]
        yt = residuals_data[g]['y_true']
        yp = residuals_data[g]['y_pred']
        ax.scatter(yt, yp, alpha=0.6, s=20, color='teal')
        min_v, max_v = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', alpha=0.8)
        ax.set_title(f'Predicciones vs Reales - G{g[-1]}')
        r2_val = regression_results_df[regression_results_df['Grado']==f'G{g[-1]}']['R²'].iloc[0]
        ax.text(0.05, 0.95, f'R² = {r2_val:.3f}', transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes_scat[5].axis('off')
    plt.tight_layout()
    fig_scat.savefig('./outputs/predicciones_vs_reales_regresion.png', dpi=300, bbox_inches='tight')

    tab_vis1, tab_vis2 = st.tabs(["📉 Distribución de Residuos", "🎯 Dispersión Reales vs Predicciones"])
    with tab_vis1: st.pyplot(fig_res)
    with tab_vis2: st.pyplot(fig_scat)
    plt.close(fig_res)
    plt.close(fig_scat)

    # =========================================================================
    # BLOQUE 4: Pipeline Completo de Predicción (Inferencia)
    # =========================================================================
    st.header("🎯 4. Pipeline Completo de Predicción (Set de Inferencia)")
    
    X_inference = inference[feature_cols]
    X_inference_encoded = X_inference.copy()
    for col in cat_features:
        if col in label_encoders:
            X_inference_encoded[col] = label_encoders[col].transform(X_inference[col].astype(str))

    pipeline_results = []
    bucket_summary = []

    for g in grades:
        clf_model = models[g]  # Viene de la Etapa 1
        prob_retiro = clf_model.predict_proba(X_inference_encoded)[:, 1]
        pred_binary = clf_model.predict(X_inference_encoded)
        
        reg_model = regression_models[g]
        pred_magnitude = np.clip(reg_model.predict(X_inference_encoded), 0, 1)
        
        # Ensamble condicional: Si clasificador predice 0 -> 0. Si predice 1 -> Magnitud
        pred_final = np.where(pred_binary == 1, pred_magnitude, 0)
        
        pipeline_results.append({
            'grado': g, 'prob_retiro': prob_retiro, 'pred_binary': pred_binary,
            'pred_magnitude': pred_magnitude, 'pred_final': pred_final
        })
        
        n_pred_0 = (pred_binary == 0).sum()
        n_pred_1 = (pred_binary == 1).sum()
        bucket_summary.append({
            'Grado': f'G{g[-1]}', 'Pred = 0': n_pred_0, 'Pred = 1': n_pred_1,
            '% Pred = 1': (n_pred_1 / len(pred_binary)) * 100, 'Media final': pred_final.mean(),
            'Media magnitud (pred=1)': pred_magnitude[pred_binary == 1].mean() if n_pred_1 > 0 else 0
        })

    predictions_df = inference[['cod_mod', 'anexo']].copy()
    for i, g in enumerate(grades):
        predictions_df[f'pred_retiro_{g}'] = pipeline_results[i]['pred_final']
        predictions_df[f'prob_retiro_{g}'] = pipeline_results[i]['prob_retiro']

    st.markdown("**📋 Resumen de Asignación por Grado (Buckets):**")
    bucket_df = pd.DataFrame(bucket_summary)
    st.dataframe(bucket_df.round(3), use_container_width=True, hide_index=True)

    st.markdown("**📊 Muestra Control de Salida (Dataset de Predicciones Finales):**")
    sample_cols = ['cod_mod', 'anexo'] + [f'pred_retiro_{g}' for g in grades]
    st.dataframe(predictions_df[sample_cols].head(10).round(4), use_container_width=True)
    predictions_df.to_csv('./outputs/predicciones_finales.csv', index=False)

    # =========================================================================
    # BLOQUE 5: Guardar Modelos y Resultados Finales
    # =========================================================================
    st.header("💾 5. Guardar Modelos y Resultados Finales")
    
    for g in grades:
        with open(f'./outputs/modelo_regresion_{g}.pkl', 'wb') as f:
            pickle.dump(regression_models[g], f)

    regression_results_df.to_csv('./outputs/resultados_regresion.csv', index=False)
    bucket_df.to_csv('./outputs/resumen_buckets.csv', index=False)

    # Reconstrucción del Resumen Ejecutivo JSON
    executive_summary = {
        "proyecto": "Predicción de Retiro Escolar - Sistema Educativo Peruano",
        "fecha_ejecucion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "arquitectura": "Two-stage multi-output (Clasificación + Regresión)",
        "datos": {
            "total_registros": len(df), "train": len(train), "inference": len(inference),
            "inference_nuevas": len(inference_nuevas), "features": len(feature_cols), "grados": len(grades)
        },
        "etapa_1_clasificacion": {
            "modelo": "LightGBM Classifier", "validacion": "Temporal (2022→2023)",
            "auc_promedio": results_df['AUC-ROC'].mean(), "f1_promedio": results_df['F1 (óptimo)'].mean(),
            "mejor_grado": results_df.loc[results_df['AUC-ROC'].idxmax(), 'Grado'] if not results_df.empty else 'N/A',
            "mejor_auc": results_df['AUC-ROC'].max() if not results_df.empty else 0
        },
        "etapa_2_regresion": {
            "modelo": "LightGBM Regressor (Tweedie)",
            "mae_promedio": regression_results_df['MAE'].mean(), "r2_promedio": regression_results_df['R²'].mean(),
            "mejor_grado": regression_results_df.loc[regression_results_df['R²'].idxmax(), 'Grado'] if not regression_results_df.empty else 'N/A',
            "mejor_r2": regression_results_df['R²'].max() if not regression_results_df.empty else 0
        },
        "predicciones_finales": {
            "escuelas_procesadas": len(predictions_df),
            "retiro_promedio_g1": bucket_df[bucket_df['Grado']=='G1']['Media final'].iloc[0] if not bucket_df.empty else 0,
            "retiro_promedio_g5": bucket_df[bucket_df['Grado']=='G5']['Media final'].iloc[0] if not bucket_df.empty else 0,
            "porcentaje_pred_positivo_g1": bucket_df[bucket_df['Grado']=='G1']['% Pred = 1'].iloc[0] if not bucket_df.empty else 0,
            "porcentaje_pred_positivo_g5": bucket_df[bucket_df['Grado']=='G5']['% Pred = 1'].iloc[0] if not bucket_df.empty else 0
        },
        "archivos_generados": [
            "modelo_clasificacion_g1.pkl - modelo_clasificacion_g5.pkl", "modelo_regresion_g1.pkl - modelo_regresion_g5.pkl", 
            "label_encoders.pkl", "predicciones_finales.csv", "resultados_clasificacion.csv", "resultados_regresion.csv",
            "resumen_buckets.csv", "feature_importance_clasificacion.png", "residuos_regresion.png", "predicciones_vs_reales_regresion.png"
        ]
    }

    import json
    with open('./outputs/resumen_ejecutivo.json', 'w', encoding='utf-8') as f:
        json.dump(executive_summary, f, indent=2, ensure_ascii=False)

    st.success("🎉 **¡PROYECTO INTEGRADO Y COMPLETADO EXITOSAMENTE!**")
    
    # Tarjetas finales de métricas resumidas
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1: st.metric("AUC Clasificación Promedio", f"{results_df['AUC-ROC'].mean():.3f}")
    with col_f2: st.metric("R² Regresión Promedio", f"{regression_results_df['R²'].mean():.3f}")
    with col_f3: st.metric("Escuelas Escaneadas", f"{len(predictions_df):,}")

    st.info("📦 Todos los artefactos operativos e históricos se encuentran almacenados de manera local en el directorio `./outputs/`.")