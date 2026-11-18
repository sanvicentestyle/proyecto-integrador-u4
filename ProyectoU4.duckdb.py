import duckdb
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configuración general de la página Streamlit (Fase 5)
st.set_page_config(
    page_title="Optimización Inferencial Agrícola",
    layout="wide"
)

# ==============================================================================
# HITO 1 / FASE 1: CARGA Y PREPARACIÓN DE TELEMETRÍA EN DUCKDB
# ==============================================================================
@st.cache_data
def generar_datos_duckdb() -> pd.DataFrame:
    # Conexión en memoria
    con = duckdb.connect(database=':memory:')
    
    # Paso 1.1: Fijar la semilla determinista global en DuckDB
    con.execute("SELECT setseed(0.42);")
    
    # Paso 1.2: Simulación de las N=100 parcelas respetando el orden y parámetros exactos
    # Nota: Se incluye temperatura_c para soportar la Gráfica 1 (Heatmap) solicitada en el Hito 4
    query_creacion = """
    CREATE TABLE parcelas_invernadero AS
    WITH base AS (
        SELECT 
            i AS parcela_id,
            ROUND(100.0 + random() * (300.0 - 100.0), 2) AS volumen_riego,
            ROUND(50.0 + random() * (150.0 - 50.0), 2) AS fertilizante_kg,
            CASE 
                WHEN random() < 0.50 THEN 'Tratamiento_A' 
                ELSE 'Tratamiento_B' 
            END AS tipo_tratamiento,
            ROUND(20.0 + random() * (30.0 - 20.0), 2) AS temperatura_c,
            (-0.8 + random() * (0.8 - (-0.8))) AS ruido_aleatorio
        FROM generate_series(1, 100) AS t(i)
    )
    SELECT 
        parcela_id,
        volumen_riego,
        fertilizante_kg,
        tipo_tratamiento,
        temperatura_c,
        ROUND(
            5.0 
            + (0.035 * volumen_riego) 
            + (0.020 * fertilizante_kg) 
            + (CASE WHEN tipo_tratamiento = 'Tratamiento_B' THEN 1.8 ELSE 0.0 END) 
            + ruido_aleatorio, 
            2
        ) AS rendimiento_ton
    FROM base;
    """
    con.execute(query_creacion)
    
    # Paso 1.3: Limpieza y extracción SQL hacia DataFrame de Pandas
    df = con.execute("""
        SELECT 
            parcela_id,
            volumen_riego,
            fertilizante_kg,
            tipo_tratamiento,
            temperatura_c,
            rendimiento_ton
        FROM parcelas_invernadero
        WHERE rendimiento_ton IS NOT NULL
        ORDER BY parcela_id ASC;
    """).df()
    
    con.close()
    return df

# Cargar dataset base
df_completo = generar_datos_duckdb()

# ==============================================================================
# CONTROLES INTERACTIVOS (FASE 5)
# ==============================================================================
st.title("🌱 Optimización Inferencial del Rendimiento Agrícola y Eficiencia de Riego")
st.markdown("Monitor de Telemetría Agroindustrial en Invernaderos Inteligentes (DuckDB + Python)")

st.sidebar.header("Filtros de Telemetría")
tratamientos_disponibles = df_completo["tipo_tratamiento"].unique().tolist()
tratamiento_sel = st.sidebar.multiselect(
    "Seleccionar Tratamiento:",
    options=tratamientos_disponibles,
    default=tratamientos_disponibles
)

rango_riego = st.sidebar.slider(
    "Rango de Volumen de Riego (m³/ha):",
    min_value=float(df_completo["volumen_riego"].min()),
    max_value=float(df_completo["volumen_riego"].max()),
    value=(float(df_completo["volumen_riego"].min()), float(df_completo["volumen_riego"].max()))
)

# Filtrado dinámico del DataFrame
df_filtrado = df_completo[
    (df_completo["tipo_tratamiento"].isin(tratamiento_sel)) &
    (df_completo["volumen_riego"].between(rango_riego[0], rango_riego[1]))
].copy()

if df_filtrado.empty:
    st.warning("No hay parcelas que coincidan con los filtros seleccionados.")
    st.stop()

# ==============================================================================
# HITO 2 / FASE 2: CORRELACIÓN Y REGRESIÓN LINEAL MÚLTIPLE (OLS)
# ==============================================================================
# Variables predictoras (X) y dependiente (Y)
X = df_filtrado[["volumen_riego", "fertilizante_kg"]]
X_const = sm.add_constant(X)  # Agregar intercepto Beta_0
Y = df_filtrado["rendimiento_ton"]

# Ajuste del modelo OLS con statsmodels
modelo_ols = sm.OLS(Y, X_const).fit()
y_pred = modelo_ols.predict(X_const)
residuos = Y - y_pred

# Métricas de validación
r2_ajustado = modelo_ols.rsquared_adj
rmse = np.sqrt(np.mean(residuos ** 2))

# Matriz de correlación de Pearson
cols_corr = ["volumen_riego", "fertilizante_kg", "temperatura_c", "rendimiento_ton"]
matriz_corr = df_filtrado[cols_corr].corr(method="pearson")

# ==============================================================================
# HITO 3 / FASE 3: INFERENCIA DE TRATAMIENTOS (PRUEBA t-STUDENT E IC 95%)
# ==============================================================================
# Se evalúa sobre el dataset completo para mantener rigor inferencial entre ambos grupos
grupo_a = df_completo[df_completo["tipo_tratamiento"] == "Tratamiento_A"]["rendimiento_ton"]
grupo_b = df_completo[df_completo["tipo_tratamiento"] == "Tratamiento_B"]["rendimiento_ton"]

# Prueba t de dos muestras independientes (H1: mu_B - mu_A > 0 -> alternative='greater')
t_stat, p_valor = stats.ttest_ind(grupo_b, grupo_a, equal_var=False, alternative="greater")

# Cálculo del Intervalo de Confianza al 95% para la diferencia de medias (mu_B - mu_A)
diff_medias = grupo_b.mean() - grupo_a.mean()
se_diff = np.sqrt((grupo_b.var(ddof=1) / len(grupo_b)) + (grupo_a.var(ddof=1) / len(grupo_a)))

# Grados de libertad de Welch-Satterthwaite
num_gl = ((grupo_b.var(ddof=1) / len(grupo_b)) + (grupo_a.var(ddof=1) / len(grupo_a))) ** 2
den_gl = ((grupo_b.var(ddof=1) / len(grupo_b)) ** 2 / (len(grupo_b) - 1)) + \
         ((grupo_a.var(ddof=1) / len(grupo_a)) ** 2 / (len(grupo_a) - 1))
gl = num_gl / den_gl

t_critico = stats.t.ppf(0.975, df=gl)
ic_95_inf = diff_medias - t_critico * se_diff
ic_95_sup = diff_medias + t_critico * se_diff

# ==============================================================================
# FASE 5.1: TARJETAS KPI DE RESUMEN
# ==============================================================================
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric(
    label="Rendimiento Promedio Global",
    value=f"{df_filtrado['rendimiento_ton'].mean():.2f} ton/ha"
)
kpi2.metric(
    label="R² Ajustado del Modelo OLS",
    value=f"{r2_ajustado * 100:.2f}%",
    delta=f"RMSE: {rmse:.2f} ton/ha",
    delta_color="off"
)
kpi3.metric(
    label="P-Valor Prueba t (Trat. B > Trat. A)",
    value=f"{p_valor:.4e}",
    delta="Significativo (α = 0.05)" if p_valor < 0.05 else "No significativo",
    delta_color="normal" if p_valor < 0.05 else "inverse"
)

st.divider()

# ==============================================================================
# HITO 4 / FASE 4 & 5.2: CUADRÍCULA 2x2 DE GRÁFICAS
# ==============================================================================
sns.set_theme(style="whitegrid")
col1, col2 = st.columns(2)

# --- GRÁFICA 1: Mapa de Calor de Correlación (Heatmap de Pearson) ---
with col1:
    st.subheader("1. Mapa de Calor de Correlación (Pearson)")
    fig1, ax1 = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=-1,
        vmax=1,
        ax=ax1
    )
    ax1.set_title("Correlación entre Variables Ambientales, Insumos y Rendimiento")
    st.pyplot(fig1)

# --- GRÁFICA 2: Dispersión con Línea de Regresión OLS ---
with col2:
    st.subheader("2. Dispersión y Ajuste Lineal OLS (Riego vs Rendimiento)")
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    sns.regplot(
        data=df_filtrado,
        x="volumen_riego",
        y="rendimiento_ton",
        scatter_kws={"alpha": 0.7, "color": "#2b8cbe"},
        line_kws={"color": "#e34a33", "label": "Ajuste OLS"},
        ax=ax2
    )
    ax2.set_xlabel("Volumen de Riego (m³/ha)")
    ax2.set_ylabel("Rendimiento (ton/ha)")
    ax2.set_title("Tendencia Lineal y Banda de Confianza (95%)")
    ax2.legend()
    st.pyplot(fig2)

col3, col4 = st.columns(2)

# --- GRÁFICA 3: Diagrama de Cajas Comparativo (Boxplot de Tratamientos) ---
with col3:
    st.subheader("3. Comparativa de Tratamientos (Boxplot)")
    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    sns.boxplot(
        data=df_completo,
        x="tipo_tratamiento",
        y="rendimiento_ton",
        palette=["#a6bddb", "#31a354"],
        ax=ax3
    )
    # Superponer medias para apoyo visual de la prueba t
    medias = df_completo.groupby("tipo_tratamiento")["rendimiento_ton"].mean()
    ax3.scatter([0, 1], [medias["Tratamiento_A"], medias["Tratamiento_B"]], 
                color="red", zorder=5, label="Media muestral")
    ax3.set_xlabel("Tipo de Tratamiento")
    ax3.set_ylabel("Rendimiento (ton/ha)")
    ax3.set_title("Distribución de Rendimiento: Estándar (A) vs Bio-estimulante (B)")
    ax3.legend()
    st.pyplot(fig3)

# --- GRÁFICA 4: Histograma y Densidad de Residuos ---
with col4:
    st.subheader("4. Diagnóstico de Residuos del Modelo")
    fig4, ax4 = plt.subplots(figsize=(6, 4.5))
    sns.histplot(residuos, kde=True, color="#756bb1", bins=15, ax=ax4)
    ax4.axvline(0, color="red", linestyle="--", linewidth=1.2, label="Media ideal (0)")
    ax4.set_xlabel("Residuos (Real - Predicho) [ton/ha]")
    ax4.set_ylabel("Frecuencia")
    ax4.set_title("Distribución de Errores (Validación de Normalidad)")
    ax4.legend()
    st.pyplot(fig4)

st.divider()

# ==============================================================================
# SECCIÓN DE RESULTADOS ESTADÍSTICOS Y VEREDICTO DE INGENIERÍA
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Resumen Regresión OLS (Hito 2)", 
    "🧪 Prueba de Hipótesis y Veredicto (Hito 3)", 
    "🗄️ Datos Crudos DuckDB (Hito 1)"
])

with tab1:
    st.markdown("### Resultados del Modelo de Regresión Lineal Múltiple")
    st.code(modelo_ols.summary().as_text(), language="text")
    st.info(
        f"**Interpretación:** El modelo explica el **{r2_ajustado*100:.2f}%** de la variabilidad "
        f"del rendimiento agrícola ($R^2$ ajustado) con un error cuadrático medio (RMSE) de **{rmse:.2f} ton/ha**."
    )

with tab2:
    st.markdown("### Evaluación Inferencial del Bio-estimulante (`Tratamiento_B`)")
    st.write(f"- **Media Tratamiento A (Estándar):** `{grupo_a.mean():.2f} ton/ha` (n = {len(grupo_a)})")
    st.write(f"- **Media Tratamiento B (Bio-estimulante):** `{grupo_b.mean():.2f} ton/ha` (n = {len(grupo_b)})")
    st.write(f"- **Diferencia de medias ($\mu_B - \mu_A$):** `+{diff_medias:.2f} ton/ha`")
    st.write(f"- **Intervalo de Confianza al 95% de la diferencia:** `[{ic_95_inf:.2f}, {ic_95_sup:.2f}] ton/ha`")
    st.write(f"- **Estadístico t:** `{t_stat:.4f}` | **P-valor unilateral:** `{p_valor:.6f}`")
    
    if p_valor < 0.05:
        st.success(
            "**Veredicto de Ingeniería:** Se rechaza $H_0$ ($p < 0.05$). Existe evidencia estadística "
            "suficiente para afirmar que el **Tratamiento_B incrementa significativamente** el rendimiento promedio. "
            "**Recomendación:** Autorizar la inversión para escalar el bio-estimulante a todo el invernadero."
        )
    else:
        st.warning(
            "**Veredicto de Ingeniería:** No se rechaza $H_0$ ($p \ge 0.05$). No hay evidencia suficiente "
            "para justificar el cambio al Tratamiento_B."
        )

with tab3:
    st.markdown("### Vista previa de las 100 parcelas generadas en DuckDB")
    st.dataframe(df_completo, use_container_width=True)