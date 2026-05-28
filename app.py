import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURACION GENERAL
# ==============================================================================

st.set_page_config(
    page_title="Trabajo Final — ML Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 10px; }
    .block-container { padding-top: 1rem; }
    h1 { color: #FFD700; }
    h2 { color: #4fc3f7; border-bottom: 2px solid #4fc3f7; padding-bottom: 6px; }
    h3 { color: #81c784; }
    .stAlert { border-radius: 10px; }
    .css-1d391kg { background-color: #1e2130; }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
    }
    .story-box {
        background-color: #1e2130;
        border-left: 4px solid #FFD700;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 10px 0;
    }
    .fase-header {
        background: linear-gradient(90deg, #1a237e, #0d47a1);
        color: white;
        padding: 12px 20px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">',
    unsafe_allow_html=True
)

# ==============================================================================
# CARGA Y PREPARACION DE DATOS (cache)
# ==============================================================================

@st.cache_data
def cargar_datos():
    df = sns.load_dataset('taxis')
    df = df.rename(columns={
        'pickup'         : 'fecha_recogida',
        'dropoff'        : 'fecha_destino',
        'passengers'     : 'pasajeros',
        'distance'       : 'distancia',
        'fare'           : 'tarifa',
        'tip'            : 'propina',
        'tolls'          : 'peajes',
        'total'          : 'total',
        'color'          : 'color',
        'payment'        : 'metodo_pago',
        'pickup_zone'    : 'zona_recogida',
        'dropoff_zone'   : 'zona_destino',
        'pickup_borough' : 'municipio_recogida',
        'dropoff_borough': 'municipio_destino'
    })
    df['hora']          = df['fecha_recogida'].dt.hour
    df['dia_semana']    = df['fecha_recogida'].dt.day_name()
    df['es_fin_semana'] = df['fecha_recogida'].dt.dayofweek >= 5
    return df

@st.cache_data
def preparar_modelo(df):
    df_prep = df.copy()
    df_prep['propina'].fillna(0, inplace=True)
    df_prep['tiene_peaje'] = (~df['peajes'].isnull()).astype(int)
    df_prep.drop('peajes', axis=1, inplace=True)
    df_prep = df_prep[df_prep['tarifa'] > 0]
    df_prep.drop(['zona_recogida', 'zona_destino'], axis=1, inplace=True)
    df_prep.drop(['fecha_destino', 'total', 'fecha_recogida', 'dia_semana'], axis=1, inplace=True)

    vars_cat = ['color', 'metodo_pago', 'municipio_recogida', 'municipio_destino']
    df_prep  = pd.get_dummies(df_prep, columns=vars_cat, drop_first=True, dtype=int)

    vars_num = ['distancia', 'propina', 'hora']
    scaler   = StandardScaler()
    df_prep[vars_num] = scaler.fit_transform(df_prep[vars_num])

    X = df_prep.drop('tarifa', axis=1)
    y = df_prep['tarifa']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo_rl = LinearRegression()
    modelo_rl.fit(X_train, y_train)

    modelo_ad = DecisionTreeRegressor(max_depth=6, random_state=42)
    modelo_ad.fit(X_train, y_train)

    y_pred_rl = modelo_rl.predict(X_test)
    y_pred_ad = modelo_ad.predict(X_test)

    return {
        'df_prep'   : df_prep,
        'X_train'   : X_train,
        'X_test'    : X_test,
        'y_train'   : y_train,
        'y_test'    : y_test,
        'modelo_rl' : modelo_rl,
        'modelo_ad' : modelo_ad,
        'y_pred_rl' : y_pred_rl,
        'y_pred_ad' : y_pred_ad,
        'r2_rl'     : r2_score(y_test, y_pred_rl),
        'mae_rl'    : mean_absolute_error(y_test, y_pred_rl),
        'rmse_rl'   : mean_squared_error(y_test, y_pred_rl) ** 0.5,
        'r2_ad'     : r2_score(y_test, y_pred_ad),
        'mae_ad'    : mean_absolute_error(y_test, y_pred_ad),
        'rmse_ad'   : mean_squared_error(y_test, y_pred_ad) ** 0.5,
    }

df  = cargar_datos()
res = preparar_modelo(df)

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown('<h2><i class="fa-solid fa-taxi"></i> Dataset Taxi ML</h2>', unsafe_allow_html=True)
    st.markdown("---")

    pagina = st.radio("Navegar por fases:", [
        "Portada",
        "Business Understanding",
        "Data Understanding",
        "Data Preparation",
        "Modeling",
        "Evaluation",
        "Etica y Sesgos",
        "Deployment"
    ])

    st.markdown("---")
    st.markdown("**Filtros globales**")

    municipios_disponibles = sorted(df['municipio_recogida'].dropna().unique())
    municipios_sel = st.multiselect(
        "Municipio de recogida",
        municipios_disponibles,
        default=municipios_disponibles
    )

    metodos_disponibles = sorted(df['metodo_pago'].dropna().unique())
    metodo_sel = st.multiselect(
        "Metodo de pago",
        metodos_disponibles,
        default=metodos_disponibles
    )

    rango_tarifa = st.slider(
        "Rango de tarifa (USD)",
        float(df['tarifa'].min()),
        float(df['tarifa'].quantile(0.99)),
        (float(df['tarifa'].min()), float(df['tarifa'].quantile(0.99)))
    )

    st.markdown("---")
    st.markdown("**Integrantes**")
    integrantes = [
        "Jaime Alberto Alzate",
        "Jhon Stiven Cortes",
        "Sebastian Henao",
        "Juan Camilo Herrera",
        "Juan Pablo Quintero"
    ]
    for i in integrantes:
        st.markdown(f"- {i}")
    st.markdown("*Pascual Bravo | 2026-1*")

# Aplicar filtros
df_filtrado = df[
    (df['municipio_recogida'].isin(municipios_sel)) &
    (df['metodo_pago'].isin(metodo_sel)) &
    (df['tarifa'] >= rango_tarifa[0]) &
    (df['tarifa'] <= rango_tarifa[1])
]

# ==============================================================================
# PAGINA: PORTADA
# ==============================================================================

if pagina == "Portada":
    st.markdown("""
    <div style='text-align:center; padding: 30px 0'>
        <img src='https://thumbs.dreamstime.com/b/taxi-36488171.jpg' width='180' style='border-radius: 16px; margin-bottom: 16px;'>
        <h1 style='font-size: 48px; color: #FFD700; margin: 0'>Dataset Taxis</h1>
        <p style='font-size: 20px; color: #aaa'>Prediccion de Tarifas con Machine Learning</p>
        <p style='color: #888'>Metodologia CRISP-DM | Ingenieria de Software | Pascual Bravo 2026-1</p>
        <p style='color:#aaa; margin-top:32px'>Jaime Alberto Alzate · Jhon Stiven Cortes · Sebastian Henao</p>
        <p style='color:#aaa; margin-top:4px'>Juan Camilo Herrera · Juan Pablo Quintero</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Viajes", f"{len(df):,}")
    col2.metric("Tarifa Promedio", f"${df['tarifa'].mean():.2f}")
    col3.metric("Distancia Promedio", f"{df['distancia'].mean():.2f} mi")
    col4.metric("Periodo", "Feb–Jun 2019")

    st.markdown("---")
    st.markdown("### ¿De qué trata este proyecto?")
    st.markdown("""
    <div class='story-box'>
    Este dashboard cuenta la historia completa de un proyecto de Machine Learning aplicado
    a datos reales de taxis en Nueva York (NYC).<br><br>
    <b>Objetivo:</b> Predecir la tarifa de un viaje de taxi a partir de variables como
    la distancia recorrida, la ubicacion geografica, la hora del dia y el tipo de servicio.<br><br>
    Seguimos la metodologia <b>CRISP-DM</b> de principio a fin:
    desde entender el problema hasta evaluar y desplegar el modelo.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Fases del proyecto")
        fases = {
            "1. Business Understanding": "Definir el problema y los criterios de exito",
            "2. Data Understanding": "Explorar, visualizar y entender los datos",
            "3. Data Preparation": "Limpiar, transformar y preparar el dataset",
            "4. Modeling": "Entrenar Regresion Lineal y Arbol de Decision",
            "5. Evaluation": "Medir R², MAE y RMSE — elegir el mejor modelo",
            "6. Deployment": "Conclusiones y pasos hacia produccion"
        }
        for fase, desc in fases.items():
            st.markdown(f"**{fase}**  \n{desc}")

    with col2:
        st.markdown("### Dataset")
        st.code("seaborn.load_dataset('taxis')", language='python')
        st.markdown(f"""
        | Campo | Valor |
        |---|---|
        | Registros | {len(df):,} |
        | Variables originales | 14 |
        | Variable objetivo | `tarifa` (USD) |
        | Tipo de problema | Regresion supervisada |
        | Periodo | Feb – Jun 2019 |
        """)

# ==============================================================================
# PAGINA: BUSINESS UNDERSTANDING
# ==============================================================================

elif pagina == "Business Understanding":
    st.markdown('<h2><i class="fa-solid fa-briefcase"></i> Fase 1: Business Understanding</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    Antes de tocar un solo dato, CRISP-DM exige entender <b>por qué</b> estamos
    construyendo el modelo y <b>qué significa el éxito</b>. Esta fase define el
    norte del proyecto.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ¿Por que este dataset?")
        razones = [
            ("La predicción de tarifas es un problema real y vigente en plataformas de movilidad como Uber, servicios de taxi tradicionales y sistemas de transporte público."),
            ("Variables ricas", "Combina datos numericos, categoricos y temporales"),
            ("Tamano manejable", "6,433 registros — ideal para modelos supervisados en clase"),
            ("Regresion clara", "La variable objetivo `tarifa` es continua con relacion logica a los predictores"),
        ]
        for titulo, desc in razones:
            st.markdown(f"**{titulo}:** {desc}")

        st.markdown("### Definicion del Problema")
        st.info("Predecir la tarifa de un viaje de taxi en NYC a partir de distancia, ubicacion, hora y tipo de servicio.")

        st.markdown("### Impacto del negocio")
        impactos = {
            "Transparencia": "Estimar el costo antes del viaje genera confianza",
            "Precios dinamicos": "Ajustar tarifas segun hora, zona y demanda",
            "Deteccion de fraudes": "Tarifas anomalas respecto al modelo = posible irregularidad"
        }
        for k, v in impactos.items():
            st.markdown(f"- **{k}:** {v}")

    with col2:
        st.markdown("### Criterios de Exito")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("R²", ">= 0.65", help="Varianza explicada por el modelo")
        col_b.metric("MAE", "< $3.50", help="Error promedio en dolares")
        col_c.metric("RMSE", "Minimizar", help="Penaliza errores grandes")

        st.markdown("### Tipo de Problema")
        st.markdown("""
        | Caracteristica | Valor |
        |---|---|
        | Categoria | Aprendizaje Supervisado |
        | Tipo | Regresion |
        | Modelos | Regresion Lineal, Arbol de Decision |
        | Division | 80% train / 20% test |
        """)

        st.markdown("### Variable Objetivo")
        fig = px.histogram(df_filtrado, x='tarifa', nbins=60,
                           title="Distribucion de la variable objetivo: tarifa (USD)",
                           color_discrete_sequence=['#FFD700'])
        fig.update_layout(
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGINA: DATA UNDERSTANDING
# ==============================================================================

elif pagina == "Data Understanding":
    st.markdown('<h2><i class="fa-solid fa-magnifying-glass-chart"></i> Fase 2: Data Understanding</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    Con el problema definido, exploramos el dataset en profundidad.
    Buscamos entender su estructura, detectar problemas de calidad
    y descubrir relaciones entre variables.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "2.1 Vista General",
        "2.2 Valores Nulos",
        "2.3 Distancia vs Tarifa",
        "2.4 Analisis Temporal"
    ])

    # --------------------------------------------------------------------------
    with tab1:
        st.markdown("### Estructura del dataset")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Registros", f"{len(df_filtrado):,}")
        col2.metric("Variables", str(df.shape[1]))
        col3.metric("Tarifa media", f"${df_filtrado['tarifa'].mean():.2f}")
        col4.metric("Distancia media", f"{df_filtrado['distancia'].mean():.2f} mi")

        st.markdown("### Primeros registros")
        cols_mostrar = ['fecha_recogida', 'pasajeros', 'distancia', 'tarifa',
                        'propina', 'color', 'metodo_pago', 'municipio_recogida']
        st.dataframe(df_filtrado[cols_mostrar].head(20), use_container_width=True)

        st.markdown("### Estadisticas descriptivas")
        st.dataframe(df_filtrado[['distancia', 'tarifa', 'propina', 'pasajeros']].describe().round(2),
                     use_container_width=True)

        st.markdown("### Distribucion por municipio")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df_filtrado, names='municipio_recogida',
                         title="Recogidas por Municipio",
                         color_discrete_sequence=px.colors.qualitative.Set3)
            fig.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.pie(df_filtrado, names='metodo_pago',
                         title="Metodo de Pago",
                         color_discrete_sequence=['#2ca02c', '#d62728'])
            fig.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    with tab2:
        st.markdown("### Analisis de Valores Nulos")

        nulos     = df.isnull().sum()
        nulos_pct = (nulos / len(df) * 100).round(2)
        nulos_df  = pd.DataFrame({
            'Variable'      : nulos.index,
            'Nulos'         : nulos.values,
            'Porcentaje (%)': nulos_pct.values
        }).query('Nulos > 0').sort_values('Porcentaje (%)', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            colores_nulos = ['#d62728' if x > 50 else '#ff7f0e' if x > 5 else '#2ca02c'
                             for x in nulos_df['Porcentaje (%)']]
            fig = go.Figure(go.Bar(
                x=nulos_df['Porcentaje (%)'],
                y=nulos_df['Variable'],
                orientation='h',
                marker_color=colores_nulos,
                text=[f"{v:,} ({p:.1f}%)" for v, p in zip(nulos_df['Nulos'], nulos_df['Porcentaje (%)'])],
                textposition='outside'
            ))
            fig.update_layout(
                title="Valores Nulos por Variable (%)",
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                font_color='white', xaxis_title="Porcentaje (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### Decisiones de tratamiento")
            st.markdown("""
            | Variable | % Nulos | Decision |
            |---|---|---|
            | `peajes` | 92.5% | Eliminar + crear `tiene_peaje` (binaria) |
            | `propina` | 4.6% | Imputar con **0** |
            """)
            st.markdown("""
            <div class='story-box'>
            <b>propina → 0:</b> Los nulos en propina corresponden a pagos en
            efectivo donde el sistema no registra la propina.
            Ausencia de registro = sin propina = 0.<br><br>
            <b>peajes → tiene_peaje:</b> Con 92.5% de nulos no es viable imputar.
            La variable binaria conserva si el viaje incluyo peaje o no.
            </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    with tab3:
        st.markdown("### Relacion Distancia vs Tarifa")

        col1, col2 = st.columns([2, 1])
        with col1:
            color_var = st.selectbox("Colorear por:", ['pasajeros', 'color', 'metodo_pago', 'municipio_recogida'])
            df_sample = df_filtrado.sample(min(2000, len(df_filtrado)))
            fig = px.scatter(
                df_sample,
                x='distancia', y='tarifa',
                color=color_var,
                title="Distancia vs Tarifa con tendencia lineal",
                labels={'distancia': 'Distancia (millas)', 'tarifa': 'Tarifa (USD)'},
                opacity=0.5
            )
            _x = df_filtrado['distancia'].dropna()
            _y = df_filtrado['tarifa'].dropna()
            _idx = _x.index.intersection(_y.index)
            _m, _b = np.polyfit(_x[_idx], _y[_idx], 1)
            _xr = np.linspace(_x.min(), _x.max(), 100)
            fig.add_trace(go.Scatter(
                x=_xr, y=_m * _xr + _b,
                mode='lines', name='Tendencia',
                line=dict(color='red', width=2, dash='dash')
            ))
            fig.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            corr = df_filtrado[['distancia', 'tarifa']].corr().iloc[0, 1]
            st.metric("Correlacion de Pearson", f"{corr:.3f}")
            st.markdown("""
            <div class='story-box'>
            Correlacion fuerte y positiva entre distancia y tarifa.<br><br>
            A mayor distancia recorrida, mayor es la tarifa — exactamente
            lo que esperariamos en un sistema de taxi real.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Tarifa por municipio")
            fig2 = px.box(df_filtrado, x='municipio_recogida', y='tarifa',
                          color='municipio_recogida',
                          title="Boxplot tarifa por municipio")
            fig2.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
                               font_color='white', showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

    # --------------------------------------------------------------------------
    with tab4:
        st.markdown("### Patrones Temporales")

        col1, col2 = st.columns(2)
        with col1:
            por_hora = df_filtrado.groupby('hora')['tarifa'].mean().reset_index()
            fig = px.line(por_hora, x='hora', y='tarifa', markers=True,
                          title="Tarifa Promedio por Hora del Dia",
                          labels={'hora': 'Hora', 'tarifa': 'Tarifa Promedio (USD)'})
            fig.update_traces(line_color='#FFD700', line_width=3)
            fig.update_layout(plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            demanda_hora = df_filtrado.groupby('hora').size().reset_index(name='viajes')
            colores_hora = ['#d62728' if (h < 6 or h >= 22)
                            else '#ff7f0e' if (6 <= h < 9 or 17 <= h < 20)
                            else '#2ca02c' for h in demanda_hora['hora']]
            fig = go.Figure(go.Bar(
                x=demanda_hora['hora'], y=demanda_hora['viajes'],
                marker_color=colores_hora
            ))
            fig.update_layout(
                title="Demanda por Hora del Dia",
                xaxis_title="Hora", yaxis_title="Numero de Viajes",
                plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        orden_dias   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        nombres_dias = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab', 'Dom']
        por_dia = df_filtrado.groupby('dia_semana')['tarifa'].mean().reindex(orden_dias).reset_index()
        por_dia['dia_es'] = nombres_dias
        colores_dia = ['#3498db'] * 5 + ['#e74c3c'] * 2
        fig = go.Figure(go.Bar(
            x=por_dia['dia_es'], y=por_dia['tarifa'],
            marker_color=colores_dia,
            text=por_dia['tarifa'].round(2), textposition='outside'
        ))
        fig.update_layout(
            title="Tarifa Promedio por Dia de la Semana",
            xaxis_title="Dia", yaxis_title="Tarifa Promedio (USD)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGINA: DATA PREPARATION
# ==============================================================================

elif pagina == "Data Preparation":
    st.markdown('<h2><i class="fa-solid fa-screwdriver-wrench"></i> Fase 3: Data Preparation</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    Con el dataset entendido, preparamos los datos para el modelado.
    Cada decision tiene una razon tecnica clara que se documenta aqui.
    </div>
    """, unsafe_allow_html=True)

    pasos = {
        "3.1 Imputacion de Nulos": {
            "color": "#dce8f5",
            "items": [
                ("propina", "Imputar con 0",
                 "Se imputa en 0 porque Pagos en tarjeta no registran propina. NaN = sin propina = 0."),
                ("peajes", "Eliminar + crear tiene_peaje",
                 "92.5% nulos. Variable binaria conserva la informacion clave.")
            ]
        },
        "3.2 Eliminacion de Errores": {
            "color": "#fde8e8",
            "items": [
                ("tarifa < 0", "Eliminar registros",
                 "Una tarifa negativa es fisicamente imposible. Son errores de entrada.")
            ]
        },
        "3.3 Limpieza Categoricas": {
            "color": "#e8f5e9",
            "items": [
                ("zona_recogida / zona_destino", "Eliminar (240+ categorias)",
                 "Alta cardinalidad genera maldicion de dimensionalidad. Se usa municipio (5 cat)."),
                ("color, metodo_pago, municipio", "One-Hot Encoding",
                 "Se codifican 4 variables con drop_first=True para evitar multicolinealidad.")
            ]
        },
        "3.4 Features Temporales": {
            "color": "#fff9db",
            "items": [
                ("fecha_recogida", "Extraer hora + es_fin_semana",
                 "Los modelos no leen datetime. Se extraen features numericas utiles."),
                ("fecha_recogida", "Eliminar columna original",
                 "Ya no aporta informacion una vez extraidas las features.")
            ]
        },
        "3.5 Colinealidad": {
            "color": "#f3e8ff",
            "items": [
                ("total", "Eliminar (data leakage)",
                 "total = tarifa + propina + peajes. Usar total para predecir tarifa es trampa."),
                ("fecha_destino", "Eliminar",
                 "Correlacionada con fecha_recogida. Informacion redundante.")
            ]
        }
    }

    for paso, contenido in pasos.items():
        with st.expander(paso, expanded=True):
            for var, decision, razon in contenido['items']:
                col1, col2, col3 = st.columns([2, 2, 4])
                col1.markdown(f"`{var}`")
                col2.markdown(f"**{decision}**")
                col3.markdown(razon)

    st.markdown("---")
    st.markdown("### Dataset antes vs despues")
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros originales", f"{len(df):,}")
    col2.metric("Registros finales", f"{len(res['df_prep']):,}")
    col3.metric("Variables finales", str(res['X_train'].shape[1]))

    st.markdown("### Variables predictoras finales (X)")
    cols = res['X_train'].columns.tolist()
    col1, col2 = st.columns(2)
    mitad = len(cols) // 2
    with col1:
        for c in cols[:mitad]:
            st.markdown(f"- `{c}`")
    with col2:
        for c in cols[mitad:]:
            st.markdown(f"- `{c}`")

# ==============================================================================
# PAGINA: MODELING
# ==============================================================================

elif pagina == "Modeling":
    st.markdown('<h2><i class="fa-solid fa-robot"></i> Fase 4: Modeling</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    Se entrenan dos modelos supervisados con los mismos datos para comparacion justa.
    Ambos usan el 80% de los datos para entrenamiento y el 20% restante para evaluacion.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Modelo 1: Regresion Lineal")
        st.markdown("""
        Modelo lineal que asume una relacion proporcional entre las variables
        predictoras y la tarifa. Simple, rapido e interpretable.
        """)

        coef_data = sorted(
            zip(res['X_train'].columns, res['modelo_rl'].coef_),
            key=lambda x: x[1]
        )
        vars_coef = [x[0] for x in coef_data]
        vals_coef = [x[1] for x in coef_data]
        colores_c = ['#d62728' if v < 0 else '#2ca02c' for v in vals_coef]

        fig = go.Figure(go.Bar(
            x=vals_coef, y=vars_coef, orientation='h',
            marker_color=colores_c,
            text=[f"{v:.2f}" for v in vals_coef],
            textposition='outside'
        ))
        fig.add_vline(x=0, line_dash='dash', line_color='white')
        fig.update_layout(
            title="Coeficientes (verde=sube tarifa | rojo=baja tarifa)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Modelo 2: Arbol de Decision")
        st.markdown("""
        Modelo no lineal que divide los datos en reglas de decision.
        Captura relaciones complejas que un modelo lineal no puede.
        """)

        imp_data = sorted(
            zip(res['X_train'].columns, res['modelo_ad'].feature_importances_),
            key=lambda x: x[1]
        )
        vars_imp = [x[0] for x in imp_data]
        vals_imp = [x[1] for x in imp_data]
        colores_i = ['#ff7f0e' if v >= 0.5 else '#1f77b4' for v in vals_imp]

        fig = go.Figure(go.Bar(
            x=vals_imp, y=vars_imp, orientation='h',
            marker_color=colores_i,
            text=[f"{v*100:.1f}%" for v in vals_imp],
            textposition='outside'
        ))
        fig.update_layout(
            title="Importancia de Variables (naranja=dominante)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font_color='white', height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("""
    <div class='story-box'>
    <b>Conclusion clave:</b> Ambos modelos coinciden en que <b>distancia</b> es el predictor
    principal. El Arbol de Decision le asigna el 93.95% de su capacidad predictiva.
    La Regresion Lineal distribuye el peso entre mas variables.
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGINA: EVALUATION
# ==============================================================================

elif pagina == "Evaluation":
    st.markdown('<h2><i class="fa-solid fa-chart-bar"></i> Fase 5: Evaluation</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    Evaluamos ambos modelos sobre el conjunto de prueba (20% de los datos)
    usando las metricas definidas en el Business Understanding.
    </div>
    """, unsafe_allow_html=True)

    # Metricas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Regresion Lineal")
        ca, cb, cc = st.columns(3)
        ca.metric("R²",   f"{res['r2_rl']:.4f}",
                  "CUMPLE" if res['r2_rl'] >= 0.65 else "NO CUMPLE")
        cb.metric("MAE",  f"${res['mae_rl']:.4f}",
                  "CUMPLE" if res['mae_rl'] < 3.5 else "NO CUMPLE")
        cc.metric("RMSE", f"${res['rmse_rl']:.4f}")

    with col2:
        st.markdown("### Arbol de Decision")
        ca, cb, cc = st.columns(3)
        ca.metric("R²",   f"{res['r2_ad']:.4f}",
                  "CUMPLE" if res['r2_ad'] >= 0.65 else "NO CUMPLE")
        cb.metric("MAE",  f"${res['mae_ad']:.4f}",
                  "CUMPLE" if res['mae_ad'] < 3.5 else "NO CUMPLE")
        cc.metric("RMSE", f"${res['rmse_ad']:.4f}")

    st.markdown("---")

    # Graficas prediccion vs real
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res['y_test'], y=res['y_pred_rl'],
            mode='markers', name='Predicciones',
            marker=dict(color='steelblue', opacity=0.4, size=5)
        ))
        lim = [res['y_test'].min(), res['y_test'].max()]
        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode='lines', name='Prediccion perfecta',
            line=dict(color='red', dash='dash', width=2)
        ))
        fig.update_layout(
            title=f"Regresion Lineal: Real vs Predicho (R²={res['r2_rl']:.3f})",
            xaxis_title="Valor Real (USD)", yaxis_title="Valor Predicho (USD)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=res['y_test'], y=res['y_pred_ad'],
            mode='markers', name='Predicciones',
            marker=dict(color='#2ca02c', opacity=0.4, size=5)
        ))
        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode='lines', name='Prediccion perfecta',
            line=dict(color='red', dash='dash', width=2)
        ))
        fig.update_layout(
            title=f"Arbol de Decision: Real vs Predicho (R²={res['r2_ad']:.3f})",
            xaxis_title="Valor Real (USD)", yaxis_title="Valor Predicho (USD)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparacion de metricas
    st.markdown("### Comparacion de Metricas")
    fig = go.Figure()
    metricas = ['R²', 'MAE', 'RMSE']
    vals_rl  = [res['r2_rl'], res['mae_rl'], res['rmse_rl']]
    vals_ad  = [res['r2_ad'], res['mae_ad'], res['rmse_ad']]

    fig.add_trace(go.Bar(name='Regresion Lineal', x=metricas, y=vals_rl,
                         marker_color='steelblue',
                         text=[f"{v:.3f}" for v in vals_rl], textposition='outside'))
    fig.add_trace(go.Bar(name='Arbol de Decision', x=metricas, y=vals_ad,
                         marker_color='#2ca02c',
                         text=[f"{v:.3f}" for v in vals_ad], textposition='outside'))
    fig.update_layout(
        barmode='group',
        plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Modelo ganador
    mejor = "Arbol de Decision" if res['r2_ad'] > res['r2_rl'] else "Regresion Lineal"
    st.success(f"**Modelo seleccionado: {mejor}** — mayor R² y menor error en conjunto de prueba.")

# ==============================================================================
# PAGINA: ETICA Y SESGOS
# ==============================================================================

elif pagina == "Etica y Sesgos":
    st.markdown('<h2><i class="fa-solid fa-scale-balanced"></i> Fase 5: Etica y Sesgos</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    Un modelo tecnicamente bueno puede ser eticamente problematico si
    comete errores sistematicamente mayores en ciertos grupos de personas.
    Analizamos si el modelo trata de forma equitativa a todos los usuarios.
    </div>
    """, unsafe_allow_html=True)

    y_pred_rl = res['y_pred_rl']
    y_pred_ad = res['y_pred_ad']
    y_test    = res['y_test']
    X_test    = res['X_test'].copy()

    X_test['tarifa_real'] = y_test.values
    X_test['error_rl']    = abs(y_test.values - y_pred_rl)
    X_test['error_ad']    = abs(y_test.values - y_pred_ad)

    def recuperar_municipio(row):
        for c in ['municipio_recogida_Brooklyn', 'municipio_recogida_Manhattan',
                  'municipio_recogida_Queens']:
            if c in row.index and row[c] == 1:
                return c.replace('municipio_recogida_', '')
        return 'Bronx'

    X_test['municipio'] = X_test.apply(recuperar_municipio, axis=1)
    X_test['pago']      = X_test['metodo_pago_credit card'].map({1: 'Tarjeta', 0: 'Efectivo'})
    X_test['taxi']      = X_test['color_yellow'].map({1: 'Amarillo', 0: 'Verde'}) \
                          if 'color_yellow' in X_test.columns else 'N/A'

    tab1, tab2, tab3 = st.tabs([
        "Sesgo Geografico",
        "Sesgo por Pago",
        "Sesgo por Tipo de Taxi"
    ])

    with tab1:
        err_geo = X_test.groupby('municipio')[['error_rl', 'error_ad']].mean().round(3)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Regresion Lineal', x=err_geo.index, y=err_geo['error_rl'],
                             marker_color='steelblue'))
        fig.add_trace(go.Bar(name='Arbol de Decision', x=err_geo.index, y=err_geo['error_ad'],
                             marker_color='#2ca02c'))
        fig.update_layout(
            title="MAE promedio por Municipio de Recogida",
            barmode='group', yaxis_title="Error Promedio (USD)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        dif = err_geo['error_rl'].max() - err_geo['error_rl'].min()
        if dif > 2:
            st.warning(f"Diferencia de ${dif:.2f} entre municipios — posible sesgo geografico.")
        else:
            st.success(f"Diferencia de ${dif:.2f} entre municipios — error relativamente uniforme.")

    with tab2:
        err_pago = X_test.groupby('pago')[['error_rl', 'error_ad']].mean().round(3)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Regresion Lineal', x=err_pago.index, y=err_pago['error_rl'],
                             marker_color='steelblue'))
        fig.add_trace(go.Bar(name='Arbol de Decision', x=err_pago.index, y=err_pago['error_ad'],
                             marker_color='#2ca02c'))
        fig.update_layout(
            title="MAE promedio por Metodo de Pago",
            barmode='group', yaxis_title="Error Promedio (USD)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        dif_p = abs(err_pago['error_rl'].max() - err_pago['error_rl'].min())
        if dif_p > 1:
            st.warning(f"Diferencia de ${dif_p:.2f} entre metodos de pago — revisar sesgo.")
        else:
            st.success(f"Diferencia de ${dif_p:.2f} — sin sesgo significativo por metodo de pago.")

    with tab3:
        err_taxi = X_test.groupby('taxi')[['error_rl', 'error_ad']].mean().round(3)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Regresion Lineal', x=err_taxi.index, y=err_taxi['error_rl'],
                             marker_color='steelblue'))
        fig.add_trace(go.Bar(name='Arbol de Decision', x=err_taxi.index, y=err_taxi['error_ad'],
                             marker_color='#2ca02c'))
        fig.update_layout(
            title="MAE promedio por Tipo de Taxi",
            barmode='group', yaxis_title="Error Promedio (USD)",
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130', font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        dif_t = abs(err_taxi['error_rl'].max() - err_taxi['error_rl'].min())
        if dif_t > 1:
            st.warning(f"Diferencia de ${dif_t:.2f} entre tipos de taxi — revisar sesgo.")
        else:
            st.success(f"Diferencia de ${dif_t:.2f} — sin sesgo significativo por tipo de taxi.")

# ==============================================================================
# PAGINA: DEPLOYMENT
# ==============================================================================

elif pagina == "Deployment":
    st.markdown('<h2><i class="fa-solid fa-rocket"></i> Fase 6: Deployment</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class='story-box'>
    El ciclo CRISP-DM culmina con las conclusiones del proyecto y los
    pasos conceptuales para llevar el modelo a produccion real.
    </div>
    """, unsafe_allow_html=True)

    # Resumen de metricas
    mejor    = "Arbol de Decision" if res['r2_ad'] > res['r2_rl'] else "Regresion Lineal"
    r2_mejor = max(res['r2_rl'], res['r2_ad'])
    mae_mejor = min(res['mae_rl'], res['mae_ad'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Modelo seleccionado", mejor)
    col2.metric("R²", f"{r2_mejor:.4f}", "CUMPLE criterio >= 0.65" if r2_mejor >= 0.65 else "NO CUMPLE")
    col3.metric("MAE", f"${mae_mejor:.4f}", "CUMPLE criterio < $3.50" if mae_mejor < 3.5 else "NO CUMPLE")
    col4.metric("Predictor principal", "distancia (93.95%)")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Conclusiones del proyecto")
        conclusiones = [
            ("Business Understanding",
             "Se definio como objetivo predecir la tarifa con R² >= 0.65 y MAE < $3.50."),
            ("Data Understanding",
             "Distancia tiene correlacion alta con tarifa (r > 0.8). Se detectaron nulos en propina y peajes."),
            ("Data Preparation",
             "Se eliminaron variables colineales, se aplico OHE y StandardScaler. Dataset final: 15 variables."),
            ("Modeling",
             "Ambos modelos identificaron 'distancia' como predictor dominante del 94%."),
            ("Evaluation",
             f"El {mejor} supero al otro en R² y MAE sobre el conjunto de prueba."),
            ("Etica",
             "No se detectaron sesgos significativos por municipio, pago o tipo de taxi.")
        ]
        for fase, desc in conclusiones:
            st.markdown(f"**{fase}:** {desc}")

    with col2:
        st.markdown("### Limitaciones")
        limitaciones = [
            "Solo 6,433 registros de un periodo especifico (Feb-Jun 2019).",
            "El 94% de la capacidad predictiva depende de 'distancia'. Si falla, el modelo falla.",
            "No se cuenta con trafico en tiempo real, clima ni eventos especiales.",
            "El modelo no se adapta a cambios futuros sin reentrenamiento.",
        ]
        for lim in limitaciones:
            st.markdown(f"- {lim}")

        st.markdown("### Pasos hacia produccion")
        pasos_prod = [
            "Serializar el modelo con `joblib`",
            "Crear una API REST con FastAPI o Flask",
            "Validar datos de entrada antes de predecir",
            "Monitorear predicciones reales vs predichas (data drift)",
            "Reentrenar trimestralmente con datos nuevos",
        ]
        for i, paso in enumerate(pasos_prod, 1):
            st.markdown(f"{i}. {paso}")

    st.markdown("---")

    # Simulador de prediccion
    st.markdown("### Simulador de Prediccion")
    st.markdown("Ingresa los datos de un viaje para estimar la tarifa:")

    col1, col2, col3 = st.columns(3)
    with col1:
        dist_input  = st.slider("Distancia (millas)", 0.5, 20.0, 3.0, 0.5)
        pas_input   = st.slider("Pasajeros", 1, 6, 1)
    with col2:
        hora_input  = st.slider("Hora del dia", 0, 23, 12)
        fin_sem     = st.checkbox("Fin de semana")
    with col3:
        mun_rec     = st.selectbox("Municipio recogida", ['Manhattan', 'Brooklyn', 'Queens', 'Bronx'])
        tipo_pago   = st.selectbox("Metodo de pago", ['Tarjeta', 'Efectivo'])

    if st.button("Estimar Tarifa", type="primary"):
        scaler_sim  = StandardScaler()
        df_sim_fit  = df[df['tarifa'] > 0].copy()
        df_sim_fit['propina'].fillna(0, inplace=True)
        scaler_sim.fit(df_sim_fit[['distancia', 'propina', 'hora']])

        dist_sc  = scaler_sim.transform([[dist_input, 0, hora_input]])[0][0]
        prop_sc  = scaler_sim.transform([[dist_input, 0, hora_input]])[0][1]
        hora_sc  = scaler_sim.transform([[dist_input, 0, hora_input]])[0][2]

        fila = {col: 0 for col in res['X_train'].columns}
        fila['distancia']   = dist_sc
        fila['pasajeros']   = pas_input
        fila['propina']     = prop_sc
        fila['hora']        = hora_sc
        fila['es_fin_semana'] = int(fin_sem)

        if mun_rec != 'Bronx':
            col_mun = f'municipio_recogida_{mun_rec}'
            if col_mun in fila:
                fila[col_mun] = 1

        if tipo_pago == 'Tarjeta' and 'metodo_pago_credit card' in fila:
            fila['metodo_pago_credit card'] = 1

        X_sim = pd.DataFrame([fila])
        pred_rl = res['modelo_rl'].predict(X_sim)[0]
        pred_ad = res['modelo_ad'].predict(X_sim)[0]

        col1, col2 = st.columns(2)
        col1.metric("Regresion Lineal", f"${max(0, pred_rl):.2f}")
        col2.metric("Arbol de Decision", f"${max(0, pred_ad):.2f}")