import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pandas as pd
import joblib
import json
import os

# ============================================
# 1. DATOS PARA LAS GRÁFICAS DE ANÁLISIS
# ============================================
# Si existe data/cardio_clean.csv (generado por entrenar_modelos.py con el
# dataset real), se usa ese. Si no, se usa una muestra sintética de respaldo
# para que el dashboard nunca se rompa por falta de datos.

RUTA_DATOS_REALES = 'data/cardio_clean.csv'

if os.path.exists(RUTA_DATOS_REALES):
    print("📊 Cargando dataset real (data/cardio_clean.csv)...")
    df = pd.read_csv(RUTA_DATOS_REALES)
    # Quitar las unidades de los nombres de columna para mantener el resto del código igual
    df = df.rename(columns={
        'height(cm)': 'height', 'weight(kg)': 'weight',
        'ap_hi(mmHg)': 'ap_hi', 'ap_lo(mmHg)': 'ap_lo'
    })
    print(f"✅ Dataset real cargado ({len(df)} pacientes)")
else:
    print("⚠️ data/cardio_clean.csv no encontrado — usando datos de ejemplo (sintéticos)")
    data = {
        'id': list(range(100)),
        'age': [50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
                49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
                50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
                49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
                50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
                49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
                50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
                49, 56, 44, 61, 42, 57, 46, 59, 54, 43,
                50, 55, 51, 48, 47, 52, 60, 45, 58, 53,
                49, 56, 44, 61, 42, 57, 46, 59, 54, 43],
        'gender': [2, 1, 1, 2, 1, 2, 1, 2, 1, 2] * 10,
        'height': [168, 156, 165, 169, 156, 160, 170, 155, 175, 165] * 10,
        'weight': [62, 85, 64, 82, 56, 70, 80, 55, 90, 65] * 10,
        'ap_hi': [110, 140, 130, 150, 100, 120, 130, 110, 140, 120] * 10,
        'ap_lo': [80, 90, 70, 100, 60, 80, 85, 70, 90, 80] * 10,
        'cholesterol': [1, 3, 3, 1, 1, 2, 3, 1, 2, 1] * 10,
        'gluc': [1, 1, 1, 1, 1, 2, 1, 1, 2, 1] * 10,
        'smoke': [0, 0, 0, 0, 0, 1, 0, 0, 1, 0] * 10,
        'alco': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] * 10,
        'active': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0] * 10,
        'cardio': [0, 1, 1, 1, 0, 1, 0, 0, 1, 0] * 10
    }
    df = pd.DataFrame(data)

# ============================================
# 2. CARGAR MODELOS (si existen)
# ============================================

try:
    modelo_clasificacion = joblib.load('models/clasificador_cardio.pkl')
    print("✅ Modelo de clasificación cargado")
except Exception:
    modelo_clasificacion = None
    print("⚠️ Modelo de clasificación NO encontrado")

# El escalador del clasificador es OPCIONAL: solo existe si el modelo ganador
# fue SVM o KNN (Random Forest no necesita datos escalados).
try:
    scaler_clasificador = joblib.load('models/scaler_clasificador.pkl')
    print("✅ Escalador del clasificador cargado (el modelo ganador lo requiere)")
except Exception:
    scaler_clasificador = None
    print("ℹ️ Sin escalador de clasificación (normal si el ganador fue Random Forest)")

try:
    modelo_regresion = joblib.load('models/regresor_presion.pkl')
    print("✅ Modelo de regresión cargado")
except Exception:
    modelo_regresion = None
    print("⚠️ Modelo de regresión NO encontrado")

try:
    scaler_regresor = joblib.load('models/scaler_regresor.pkl')
    print("✅ Escalador de regresión cargado")
except Exception:
    scaler_regresor = None
    print("⚠️ Escalador de regresión NO encontrado")

# ============================================
# 2.1 CARGAR DATOS DEL MAPA DE PANAMÁ
# ============================================

try:
    with open('data/panama_geo.json', encoding='utf-8') as f:
        panama_geojson = json.load(f)
    df_panama = pd.read_csv('data/panama_ingresos.csv')
    print(f"✅ Mapa de Panamá cargado ({len(df_panama)} distritos)")
except Exception as e:
    panama_geojson = None
    df_panama = None
    print(f"⚠️ Datos del mapa de Panamá NO encontrados: {e}")

# ============================================
# 3. PALETA Y TEMA VISUAL
# ============================================

COLORS = {
    'bg': '#f4f6fa',
    'card': '#ffffff',
    'primary': '#3b5bdb',
    'primary_dark': '#2c46b0',
    'accent': '#0ca678',
    'danger': '#e03131',
    'text': '#212529',
    'muted': '#6c757d',
    'sano': '#12b886',
    'riesgo': '#e03131',
}

PLOTLY_TEMPLATE = 'plotly_white'
CARDIO_COLOR_MAP = {0: COLORS['sano'], 1: COLORS['riesgo']}

CARD_STYLE = {
    'backgroundColor': COLORS['card'],
    'borderRadius': '16px',
    'padding': '20px',
    'margin': '14px',
    'boxShadow': '0 4px 16px rgba(0,0,0,0.06)',
}

# ============================================
# 4. GRÁFICAS DE ANÁLISIS EXPLORATORIO
# ============================================

corr_matrix = df.select_dtypes(include=['number']).drop(columns=['id']).corr()
fig_correlacion = ff.create_annotated_heatmap(
    z=corr_matrix.values,
    x=list(corr_matrix.columns),
    y=list(corr_matrix.index),
    colorscale='RdBu',
    zmid=0,
    showscale=True,
    annotation_text=corr_matrix.round(2).values
)
fig_correlacion.update_layout(
    title='Matriz de Correlación entre Variables',
    height=600,
    font=dict(size=11, family='Inter, sans-serif'),
    margin=dict(l=40, r=40, t=60, b=40),
)

fig_edad = px.histogram(
    df, x='age', color='cardio', barmode='overlay',
    title='Distribución de Edad por Estado Cardiovascular',
    labels={'age': 'Edad (años)', 'count': 'Pacientes', 'cardio': 'Cardio'},
    color_discrete_map=CARDIO_COLOR_MAP, nbins=25, template=PLOTLY_TEMPLATE,
)
fig_edad.update_layout(bargap=0.1, font_family='Inter, sans-serif', legend_title_text='Riesgo')

fig_peso = px.histogram(
    df, x='weight', color='cardio', barmode='overlay',
    title='Distribución de Peso por Estado Cardiovascular',
    labels={'weight': 'Peso (kg)', 'count': 'Pacientes', 'cardio': 'Cardio'},
    color_discrete_map=CARDIO_COLOR_MAP, nbins=25, template=PLOTLY_TEMPLATE,
)
fig_peso.update_layout(bargap=0.1, font_family='Inter, sans-serif', legend_title_text='Riesgo')

fig_presion = px.scatter(
    df, x='ap_hi', y='ap_lo', color='cardio',
    title='Presión Sistólica vs. Diastólica',
    labels={'ap_hi': 'Presión Sistólica (mmHg)', 'ap_lo': 'Presión Diastólica (mmHg)', 'cardio': 'Cardio'},
    color_discrete_map=CARDIO_COLOR_MAP, opacity=0.55, hover_data=['age', 'weight'],
    template=PLOTLY_TEMPLATE,
)
fig_presion.update_layout(font_family='Inter, sans-serif', legend_title_text='Riesgo')

for fig in (fig_edad, fig_peso, fig_presion):
    fig.update_layout(title_font_size=16, title_x=0.02)


def crear_mapa_panama():
    if panama_geojson is None or df_panama is None:
        fig = go.Figure()
        fig.update_layout(
            annotations=[{
                'text': '⚠️ No se encontraron los archivos data/panama_geo.json y data/panama_ingresos.csv',
                'showarrow': False, 'font': {'size': 14, 'color': COLORS['muted']}
            }],
            height=550, xaxis={'visible': False}, yaxis={'visible': False},
            font_family='Inter, sans-serif'
        )
        return fig

    fig = px.choropleth_mapbox(
        df_panama,
        geojson=panama_geojson,
        locations='shapeName',
        featureidkey='properties.shapeName',
        color='ingreso',
        color_continuous_scale='YlGnBu',
        range_color=(df_panama['ingreso'].min(), df_panama['ingreso'].max()),
        mapbox_style='carto-positron',
        zoom=6.1,
        center={'lat': 8.6, 'lon': -80.3},
        opacity=0.78,
        hover_name='shapeName',
        hover_data={'provincia': True, 'ingreso': ':.0f', 'shapeName': False},
        labels={'ingreso': 'Ingreso promedio (USD/mes)', 'provincia': 'Provincia'},
    )
    fig.update_layout(
        title='Ingreso Promedio Mensual por Persona, por Distrito (INEC 2023)',
        title_font_size=16, title_x=0.02,
        margin=dict(l=0, r=0, t=50, b=0),
        height=580,
        font_family='Inter, sans-serif',
        coloraxis_colorbar=dict(title='USD/mes'),
    )
    return fig


fig_mapa_panama = crear_mapa_panama()


def grafico_vacio(mensaje):
    """Gauge/placeholder inicial antes de predecir."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        number={'suffix': '%', 'font': {'size': 36, 'color': COLORS['muted']}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': COLORS['muted']},
            'steps': [
                {'range': [0, 40], 'color': '#e6fcf5'},
                {'range': [40, 70], 'color': '#fff9db'},
                {'range': [70, 100], 'color': '#ffe3e3'},
            ],
        },
        title={'text': mensaje, 'font': {'size': 14}}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), font_family='Inter, sans-serif')
    return fig


def grafico_gauge(probabilidad_pct, es_riesgo):
    color_barra = COLORS['riesgo'] if es_riesgo else COLORS['sano']
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probabilidad_pct,
        number={'suffix': '%', 'font': {'size': 40, 'color': color_barra}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color_barra, 'thickness': 0.3},
            'steps': [
                {'range': [0, 40], 'color': '#e6fcf5'},
                {'range': [40, 70], 'color': '#fff9db'},
                {'range': [70, 100], 'color': '#ffe3e3'},
            ],
            'threshold': {
                'line': {'color': color_barra, 'width': 4},
                'thickness': 0.85,
                'value': probabilidad_pct
            }
        },
        title={'text': 'Probabilidad de Riesgo Cardiovascular', 'font': {'size': 15}}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), font_family='Inter, sans-serif')
    return fig


def grafico_vacio_presion(mensaje):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        number={'suffix': ' mmHg', 'font': {'size': 32, 'color': COLORS['muted']}},
        gauge={
            'axis': {'range': [60, 220], 'tickwidth': 1},
            'bar': {'color': COLORS['muted']},
            'steps': [
                {'range': [60, 120], 'color': '#e6fcf5'},
                {'range': [120, 130], 'color': '#fff9db'},
                {'range': [130, 140], 'color': '#ffe8cc'},
                {'range': [140, 180], 'color': '#ffd8a8'},
                {'range': [180, 220], 'color': '#ffc9c9'},
            ],
        },
        title={'text': mensaje, 'font': {'size': 14}}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), font_family='Inter, sans-serif')
    return fig


def grafico_gauge_presion(valor_ap_hi, categoria_color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor_ap_hi,
        number={'suffix': ' mmHg', 'font': {'size': 36, 'color': categoria_color}},
        gauge={
            'axis': {'range': [60, 220], 'tickwidth': 1},
            'bar': {'color': categoria_color, 'thickness': 0.3},
            'steps': [
                {'range': [60, 120], 'color': '#e6fcf5'},
                {'range': [120, 130], 'color': '#fff9db'},
                {'range': [130, 140], 'color': '#ffe8cc'},
                {'range': [140, 180], 'color': '#ffd8a8'},
                {'range': [180, 220], 'color': '#ffc9c9'},
            ],
            'threshold': {
                'line': {'color': categoria_color, 'width': 4},
                'thickness': 0.85,
                'value': valor_ap_hi
            }
        },
        title={'text': 'Presión Sistólica Predicha (ap_hi)', 'font': {'size': 15}}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10), font_family='Inter, sans-serif')
    return fig


def clasificar_presion(ap_hi):
    """Misma lógica de categorización usada en el notebook (Sección 4)."""
    if ap_hi < 120:
        return "Normal", COLORS['sano']
    elif ap_hi <= 129:
        return "Elevada", '#f08c00'
    elif ap_hi <= 139:
        return "Hipertensión Grado 1", '#e8590c'
    elif ap_hi <= 180:
        return "Hipertensión Grado 2", '#e03131'
    else:
        return "Crisis Hipertensiva", '#9c0000'


# ============================================
# 5. HELPERS DE UI
# ============================================

def campo_slider(id_, etiqueta, min_, max_, valor, step=1, marks=None):
    return html.Div([
        html.Label(etiqueta, style={'fontWeight': 600, 'color': COLORS['text'], 'fontSize': '14px'}),
        dcc.Slider(
            id=id_, min=min_, max=max_, step=step, value=valor, marks=marks,
            tooltip={'placement': 'bottom', 'always_visible': True}
        ),
    ], style={'margin': '18px 6px'})


def campo_dropdown(id_, etiqueta, opciones, valor):
    return html.Div([
        html.Label(etiqueta, style={'fontWeight': 600, 'color': COLORS['text'], 'fontSize': '14px'}),
        dcc.Dropdown(id=id_, options=opciones, value=valor, clearable=False, style={'marginTop': '6px'}),
    ], style={'margin': '14px 6px'})


def campo_toggle(id_, etiqueta, valor):
    return html.Div([
        html.Label(etiqueta, style={'fontWeight': 600, 'color': COLORS['text'], 'fontSize': '14px'}),
        dcc.RadioItems(
            id=id_,
            options=[{'label': ' No', 'value': 0}, {'label': ' Sí', 'value': 1}],
            value=valor, inline=True,
            style={'marginTop': '8px'},
            inputStyle={'marginRight': '6px', 'marginLeft': '14px'}
        ),
    ], style={'margin': '14px 6px'})


# ============================================
# 6. APP
# ============================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    ],
)
app.title = "Predictor Cardiovascular"
server = app.server

_index_template = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { font-family: 'Inter', sans-serif; background-color: __BG__; margin: 0; }
            ::-webkit-scrollbar { width: 8px; }
            ::-webkit-scrollbar-thumb { background: #c1c9d6; border-radius: 8px; }
            .tab-container .tab {
                border: none !important;
                font-weight: 600 !important;
                color: __MUTED__ !important;
            }
            .tab-container .tab--selected {
                border-bottom: 3px solid __PRIMARY__ !important;
                color: __PRIMARY__ !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.index_string = (
    _index_template
    .replace('__BG__', COLORS['bg'])
    .replace('__MUTED__', COLORS['muted'])
    .replace('__PRIMARY__', COLORS['primary'])
)

# ---------- HEADER ----------
header = html.Div([
    html.Div([
        html.Span("🫀", style={'fontSize': '38px', 'marginRight': '14px'}),
        html.Div([
            html.H1("Predictor de Riesgo Cardiovascular",
                    style={'margin': 0, 'color': 'white', 'fontSize': '28px', 'fontWeight': 800}),
            html.P("Análisis exploratorio y predicción basada en datos clínicos",
                   style={'margin': 0, 'color': 'rgba(255,255,255,0.85)', 'fontSize': '14px'}),
        ])
    ], style={'display': 'flex', 'alignItems': 'center'})
], style={
    'background': f"linear-gradient(120deg, {COLORS['primary']}, {COLORS['primary_dark']})",
    'padding': '28px 36px',
    'borderRadius': '0 0 24px 24px',
    'boxShadow': '0 4px 20px rgba(59,91,219,0.25)',
    'marginBottom': '10px',
})

# ---------- TAB: ANÁLISIS EXPLORATORIO ----------
tab_analisis = html.Div([
    html.Div([dcc.Graph(figure=fig_correlacion, config={'displayModeBar': False})], style=CARD_STYLE),
    html.Div([
        html.Div([dcc.Graph(figure=fig_edad, config={'displayModeBar': False})],
                  style={**CARD_STYLE, 'flex': '1', 'minWidth': '320px'}),
        html.Div([dcc.Graph(figure=fig_peso, config={'displayModeBar': False})],
                  style={**CARD_STYLE, 'flex': '1', 'minWidth': '320px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    html.Div([dcc.Graph(figure=fig_presion, config={'displayModeBar': False})], style=CARD_STYLE),
], style={'padding': '10px'})

# ---------- TAB: PREDICTOR ----------
tab_predictor = html.Div([
    html.Div([

        # Columna izquierda: formulario
        html.Div([
            html.H3("Datos del Paciente", style={'color': COLORS['text'], 'marginTop': 0}),

            html.Div([
                html.Div([
                    campo_slider('input-edad', 'Edad (años)', 18, 100, 50),
                    campo_slider('input-altura', 'Altura (cm)', 130, 210, 165),
                    campo_slider('input-peso', 'Peso (kg)', 35, 180, 70),
                ], style={'flex': '1', 'minWidth': '260px'}),

                html.Div([
                    campo_slider('input-ap_hi', 'Presión Sistólica (mmHg)', 80, 220, 120),
                    campo_slider('input-ap_lo', 'Presión Diastólica (mmHg)', 40, 150, 80),
                    campo_dropdown('input-genero', 'Género', [
                        {'label': 'Mujer', 'value': 1}, {'label': 'Hombre', 'value': 2}
                    ], 1),
                ], style={'flex': '1', 'minWidth': '260px'}),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),

            html.Hr(style={'margin': '10px 0 20px', 'borderColor': '#eee'}),

            html.Div([
                campo_dropdown('input-colesterol', 'Colesterol', [
                    {'label': 'Normal', 'value': 1}, {'label': 'Alto', 'value': 2}, {'label': 'Muy alto', 'value': 3}
                ], 1),
                campo_dropdown('input-glucosa', 'Glucosa', [
                    {'label': 'Normal', 'value': 1}, {'label': 'Alto', 'value': 2}, {'label': 'Muy alto', 'value': 3}
                ], 1),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),

            html.Div([
                campo_toggle('input-fuma', 'Fuma', 0),
                campo_toggle('input-alcohol', 'Consume alcohol', 0),
                campo_toggle('input-activo', 'Actividad física', 1),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginTop': '6px'}),

            html.Div([
                html.Button(
                    '🔍  Predecir Riesgo',
                    id='btn-predecir', n_clicks=0,
                    style={
                        'padding': '14px 36px', 'fontSize': '16px', 'fontWeight': 700,
                        'background': COLORS['primary'], 'color': 'white', 'border': 'none',
                        'borderRadius': '10px', 'cursor': 'pointer', 'marginTop': '24px',
                        'boxShadow': '0 4px 14px rgba(59,91,219,0.35)', 'transition': 'transform 0.1s',
                    }
                ),
            ], style={'textAlign': 'center'}),

        ], style={**CARD_STYLE, 'flex': '1.3', 'minWidth': '380px'}),

        # Columna derecha: resultado
        html.Div([
            html.H3("Resultado", style={'color': COLORS['text'], 'marginTop': 0}),
            dcc.Loading(
                id='loading-resultado',
                type='circle',
                color=COLORS['primary'],
                children=[
                    dcc.Graph(id='gauge-resultado', figure=grafico_vacio("Esperando datos"),
                              config={'displayModeBar': False}),
                    html.Div(
                        id='resultado-texto',
                        children="Completa los datos y presiona 'Predecir Riesgo'.",
                        style={'textAlign': 'center', 'fontSize': '16px', 'color': COLORS['muted'],
                               'padding': '10px 6px'}
                    ),
                ]
            ),
        ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '320px'}),

    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'padding': '10px'}),
])

# ---------- TAB: MAPA DE PANAMÁ ----------
tab_mapa = html.Div([
    html.Div([
        html.H3("Ingreso Promedio por Distrito", style={'color': COLORS['text'], 'marginTop': 0}),
        html.P(
            "Mapa interactivo a nivel de distritos de Panamá. Pasa el mouse sobre cada distrito "
            "para ver su provincia e ingreso promedio mensual por persona (INEC 2023).",
            style={'color': COLORS['muted'], 'fontSize': '14px'}
        ),
        dcc.Graph(figure=fig_mapa_panama, config={'scrollZoom': True}),
    ], style=CARD_STYLE)
], style={'padding': '10px'})

# ---------- TAB: PREDICTOR DE REGRESIÓN (PRESIÓN SISTÓLICA) ----------
tab_regresion = html.Div([
    html.Div([

        html.Div([
            html.H3("Datos del Paciente", style={'color': COLORS['text'], 'marginTop': 0}),
            html.P(
                "Modelo ganador: MLP Regressor (R² = 0.601), entrenado para predecir la "
                "presión sistólica (ap_hi) a partir de estas variables.",
                style={'color': COLORS['muted'], 'fontSize': '13px'}
            ),

            html.Div([
                html.Div([
                    campo_slider('reg-edad', 'Edad (años)', 18, 100, 50),
                    campo_slider('reg-altura', 'Altura (cm)', 130, 210, 165),
                    campo_slider('reg-peso', 'Peso (kg)', 35, 180, 70),
                    campo_slider('reg-ap_lo', 'Presión Diastólica (mmHg)', 40, 150, 80),
                ], style={'flex': '1', 'minWidth': '260px'}),

                html.Div([
                    campo_dropdown('reg-genero', 'Género', [
                        {'label': 'Mujer', 'value': 1}, {'label': 'Hombre', 'value': 2}
                    ], 1),
                    campo_dropdown('reg-colesterol', 'Colesterol', [
                        {'label': 'Normal', 'value': 1}, {'label': 'Alto', 'value': 2}, {'label': 'Muy alto', 'value': 3}
                    ], 1),
                    campo_dropdown('reg-glucosa', 'Glucosa', [
                        {'label': 'Normal', 'value': 1}, {'label': 'Alto', 'value': 2}, {'label': 'Muy alto', 'value': 3}
                    ], 1),
                    campo_dropdown('reg-cardio', 'Diagnóstico cardiovascular previo', [
                        {'label': 'Sano', 'value': 0}, {'label': 'Enfermo', 'value': 1}
                    ], 0),
                ], style={'flex': '1', 'minWidth': '260px'}),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}),

            html.Div([
                campo_toggle('reg-fuma', 'Fuma', 0),
                campo_toggle('reg-alcohol', 'Consume alcohol', 0),
                campo_toggle('reg-activo', 'Actividad física', 1),
            ], style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap', 'marginTop': '6px'}),

            html.Div([
                html.Button(
                    '📈  Predecir Presión Sistólica',
                    id='btn-predecir-reg', n_clicks=0,
                    style={
                        'padding': '14px 36px', 'fontSize': '16px', 'fontWeight': 700,
                        'background': COLORS['accent'], 'color': 'white', 'border': 'none',
                        'borderRadius': '10px', 'cursor': 'pointer', 'marginTop': '24px',
                        'boxShadow': '0 4px 14px rgba(12,166,120,0.35)',
                    }
                ),
            ], style={'textAlign': 'center'}),

        ], style={**CARD_STYLE, 'flex': '1.3', 'minWidth': '380px'}),

        html.Div([
            html.H3("Resultado", style={'color': COLORS['text'], 'marginTop': 0}),
            dcc.Loading(
                id='loading-resultado-reg',
                type='circle',
                color=COLORS['accent'],
                children=[
                    dcc.Graph(id='gauge-resultado-reg', figure=grafico_vacio_presion("Esperando datos"),
                              config={'displayModeBar': False}),
                    html.Div(
                        id='resultado-texto-reg',
                        children="Completa los datos y presiona 'Predecir Presión Sistólica'.",
                        style={'textAlign': 'center', 'fontSize': '16px', 'color': COLORS['muted'],
                               'padding': '10px 6px'}
                    ),
                ]
            ),
        ], style={**CARD_STYLE, 'flex': '1', 'minWidth': '320px'}),

    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'padding': '10px'}),
])

# ---------- LAYOUT PRINCIPAL ----------
app.layout = html.Div([
    header,

    html.Div([
        dcc.Tabs(id='tabs-principales', value='tab-predictor', className='tab-container', children=[
            dcc.Tab(label='🔮 Predictor de Diagnóstico', value='tab-predictor'),
            dcc.Tab(label='📈 Predictor de Presión', value='tab-regresion'),
            dcc.Tab(label='📊 Análisis Exploratorio', value='tab-analisis'),
            dcc.Tab(label='🗺️ Mapa de Panamá', value='tab-mapa'),
        ]),
        html.Div(id='contenido-tabs'),
    ], style={'maxWidth': '1200px', 'margin': '0 auto', 'padding': '0 16px'}),

    html.P(
        "Dashboard creado con Dash · Análisis y predicción de enfermedades cardiovasculares",
        style={'textAlign': 'center', 'color': COLORS['muted'], 'padding': '30px', 'fontSize': '13px'}
    )
], style={'backgroundColor': COLORS['bg'], 'minHeight': '100vh', 'paddingBottom': '20px'})


# ============================================
# 7. CALLBACKS
# ============================================

@app.callback(Output('contenido-tabs', 'children'), Input('tabs-principales', 'value'))
def render_tab(tab):
    if tab == 'tab-predictor':
        return tab_predictor
    elif tab == 'tab-regresion':
        return tab_regresion
    elif tab == 'tab-mapa':
        return tab_mapa
    return tab_analisis


@app.callback(
    Output('gauge-resultado', 'figure'),
    Output('resultado-texto', 'children'),
    Output('resultado-texto', 'style'),
    Input('btn-predecir', 'n_clicks'),
    State('input-edad', 'value'),
    State('input-genero', 'value'),
    State('input-altura', 'value'),
    State('input-peso', 'value'),
    State('input-ap_hi', 'value'),
    State('input-ap_lo', 'value'),
    State('input-colesterol', 'value'),
    State('input-glucosa', 'value'),
    State('input-fuma', 'value'),
    State('input-alcohol', 'value'),
    State('input-activo', 'value'),
    prevent_initial_call=True,
)
def predecir(n_clicks, edad, genero, altura, peso, ap_hi, ap_lo,
             colesterol, glucosa, fuma, alcohol, activo):

    texto_base_style = {'textAlign': 'center', 'fontSize': '16px', 'padding': '10px 6px'}

    if modelo_clasificacion is None:
        return (
            grafico_vacio("Modelo no disponible"),
            "⚠️ No se encontró el modelo en 'models/clasificador_cardio.pkl'. Entrénalo y guárdalo primero.",
            {**texto_base_style, 'color': '#e67e22', 'fontWeight': 600}
        )

    try:
        datos = [[edad, genero, altura, peso, ap_hi, ap_lo, colesterol, glucosa, fuma, alcohol, activo]]

        # Si el modelo ganador fue SVM o KNN, necesita los datos escalados
        if scaler_clasificador is not None:
            datos = scaler_clasificador.transform(datos)

        if hasattr(modelo_clasificacion, "predict_proba"):
            proba = modelo_clasificacion.predict_proba(datos)[0]
            prob_riesgo = float(proba[1]) * 100
        else:
            pred = modelo_clasificacion.predict(datos)[0]
            prob_riesgo = 85.0 if pred == 1 else 15.0

        es_riesgo = prob_riesgo >= 50

        if es_riesgo:
            mensaje = f"⚠️ Riesgo ALTO estimado: {prob_riesgo:.1f}%"
            color = COLORS['riesgo']
        else:
            mensaje = f"✅ Riesgo BAJO estimado: {prob_riesgo:.1f}%"
            color = COLORS['sano']

        return (
            grafico_gauge(prob_riesgo, es_riesgo),
            mensaje,
            {**texto_base_style, 'color': color, 'fontWeight': 700, 'fontSize': '19px'}
        )

    except Exception as e:
        return (
            grafico_vacio("Error"),
            f"❌ Error en la predicción: {str(e)}",
            {**texto_base_style, 'color': COLORS['riesgo'], 'fontWeight': 600}
        )


@app.callback(
    Output('gauge-resultado-reg', 'figure'),
    Output('resultado-texto-reg', 'children'),
    Output('resultado-texto-reg', 'style'),
    Input('btn-predecir-reg', 'n_clicks'),
    State('reg-edad', 'value'),
    State('reg-genero', 'value'),
    State('reg-altura', 'value'),
    State('reg-peso', 'value'),
    State('reg-ap_lo', 'value'),
    State('reg-colesterol', 'value'),
    State('reg-glucosa', 'value'),
    State('reg-fuma', 'value'),
    State('reg-alcohol', 'value'),
    State('reg-activo', 'value'),
    State('reg-cardio', 'value'),
    prevent_initial_call=True,
)
def predecir_presion(n_clicks, edad, genero, altura, peso, ap_lo,
                      colesterol, glucosa, fuma, alcohol, activo, cardio):

    texto_base_style = {'textAlign': 'center', 'fontSize': '16px', 'padding': '10px 6px'}

    if modelo_regresion is None or scaler_regresor is None:
        faltante = "models/regresor_presion.pkl" if modelo_regresion is None else "models/scaler_regresor.pkl"
        return (
            grafico_vacio_presion("Modelo no disponible"),
            f"⚠️ No se encontró '{faltante}'. Genera y guarda ese archivo primero.",
            {**texto_base_style, 'color': '#e67e22', 'fontWeight': 600}
        )

    try:
        # Mismo orden de variables usado al entrenar el MLP Regressor en el notebook:
        # age, gender, height, weight, ap_lo, cholesterol, gluc, smoke, alco, active, cardio
        datos = [[edad, genero, altura, peso, ap_lo, colesterol, glucosa, fuma, alcohol, activo, cardio]]
        datos_escalados = scaler_regresor.transform(datos)

        valor_predicho = float(modelo_regresion.predict(datos_escalados)[0])
        categoria, color = clasificar_presion(valor_predicho)

        mensaje = f"Presión sistólica estimada: {valor_predicho:.1f} mmHg — {categoria}"

        return (
            grafico_gauge_presion(valor_predicho, color),
            mensaje,
            {**texto_base_style, 'color': color, 'fontWeight': 700, 'fontSize': '18px'}
        )

    except Exception as e:
        return (
            grafico_vacio_presion("Error"),
            f"❌ Error en la predicción: {str(e)}",
            {**texto_base_style, 'color': COLORS['riesgo'], 'fontWeight': 600}
        )


# ============================================
# 8. EJECUTAR
# ============================================

if __name__ == '__main__':
    app.run(debug=True)
 

