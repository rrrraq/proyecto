"""
entrenar_modelos.py
====================
Script NO interactivo para correr en Google Colab (usa tu mismo Drive y
dataset). Reemplaza las Secciones 0.2, 1, 2, 3, 4 y 6 de tu notebook
(quitando los input() manuales) y al final te deja listos, en /content:

    models/clasificador_cardio.pkl      (RF, SVM o KNN — el que gane)
    models/scaler_clasificador.pkl      (solo si el ganador fue SVM o KNN)
    models/regresor_presion.pkl         (RFR o MLP — el que gane)
    models/scaler_regresor.pkl
    data/cardio_clean.csv               (dataset limpio, para las 4 gráficas
                                          del dashboard con datos REALES)

Descarga las carpetas 'models/' y 'data/' completas y súbelas a tu repo de
GitHub, respetando esa misma estructura de carpetas junto a tu app.py.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from google.colab import drive
drive.mount('/content/drive')

SEMILLA = 42
TEST_SIZE = 0.2

# ============================================
# 0.2 FUNCIONES DE USO GENERAL (sin input(), listas para producción)
# ============================================

def error_score(modelo, x_test, y_test):
    """Precisión (score) del modelo sobre datos de prueba."""
    return modelo.score(x_test, y_test)


def calc_overfitting(modelo, x_train, x_test, y_train, y_test, nombre_modelo, umbral=0.05):
    acc_train = error_score(modelo, x_train, y_train)
    acc_test = error_score(modelo, x_test, y_test)
    diferencia = acc_train - acc_test
    print(f"\n--- {nombre_modelo} ---")
    print(f"Score Train: {acc_train:.4f}")
    print(f"Score Test:  {acc_test:.4f}")
    print(f"Diferencia:  {diferencia:.4f}")
    print("⚠️ Posible overfitting" if diferencia > umbral else "✅ El modelo generaliza correctamente")
    return diferencia


def escalar(x_train, x_test):
    """Ajusta un StandardScaler con x_train y transforma ambos conjuntos."""
    scaler = StandardScaler()
    x_train_esc = scaler.fit_transform(x_train)
    x_test_esc = scaler.transform(x_test)
    return x_train_esc, x_test_esc, scaler


def clasificar_presion(ap_hi):
    """Misma lógica de tu notebook (Sección 4) para categorizar ap_hi."""
    if ap_hi < 120:
        return "Normal"
    elif ap_hi <= 129:
        return "Elevada"
    elif ap_hi <= 139:
        return "Hipertensión Grado 1"
    elif ap_hi <= 180:
        return "Hipertensión Grado 2"
    else:
        return "Crisis Hipertensiva"


# ---------- FUNCIONES PARA GRAFICAR (Sección 0.2) ----------

def matriz_confusion(modelo, x_train, x_test, y_train, y_test, nombre_modelo):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cm_train = confusion_matrix(y_train, modelo.predict(x_train), normalize="true")
    ConfusionMatrixDisplay(cm_train, display_labels=['Sano', 'Enfermo']).plot(ax=axes[0])
    axes[0].set_title(f"{nombre_modelo} — Train")
    cm_test = confusion_matrix(y_test, modelo.predict(x_test), normalize="true")
    ConfusionMatrixDisplay(cm_test, display_labels=['Sano', 'Enfermo']).plot(ax=axes[1])
    axes[1].set_title(f"{nombre_modelo} — Test")
    plt.tight_layout()
    plt.show()


# ============================================
# SECCIÓN 1: INGESTA, LIMPIEZA Y PREPARACIÓN
# (idéntico a tu notebook)
# ============================================

ruta_archivo = "/content/drive/MyDrive/AnDa_ProyectoFinal_Data/cardio_train.csv"
df_cardios = pd.read_csv(ruta_archivo, sep=';')

mask_pressure = (df_cardios['ap_hi'] > df_cardios['ap_lo']) & \
                (df_cardios['ap_hi'] < 250) & (df_cardios['ap_hi'] > 60) & \
                (df_cardios['ap_lo'] < 150) & (df_cardios['ap_lo'] > 40)
mask_height = (df_cardios['height'] >= 100) & (df_cardios['height'] <= 200)
mask_weight = (df_cardios['weight'] >= 40) & (df_cardios['weight'] <= 200)

df_cardiosclean = df_cardios[mask_pressure & mask_height & mask_weight].copy()
df_cardiosclean['age'] = (df_cardiosclean['age'] / 365.25).astype(int)
df_cardiosclean.drop_duplicates(inplace=True)

columnas_unidades = {
    'height': 'height(cm)', 'weight': 'weight(kg)',
    'ap_hi': 'ap_hi(mmHg)', 'ap_lo': 'ap_lo(mmHg)'
}
df_cardiosclean.rename(columns=columnas_unidades, inplace=True)
print(f"✅ Limpieza completada. Registros finales: {len(df_cardiosclean)}")

# ============================================
# SECCIÓN 2: EDA — SE EXPORTA EL DATASET LIMPIO
# PARA QUE EL DASHBOARD GRAFIQUE CON DATOS REALES
# ============================================

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

df_cardiosclean.to_csv('data/cardio_clean.csv', index=False)
print("✅ data/cardio_clean.csv guardado — el dashboard usará esto para sus 4 gráficas")

# Vistazo rápido de referencia (igual que en tu notebook, Sección 2)
plt.figure(figsize=(6, 4))
sns.histplot(data=df_cardiosclean, x='age', hue='cardio', bins=25)
plt.title("Distribución de Edad por Estado Cardiovascular")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(
    df_cardiosclean.select_dtypes('number').drop(columns=['id']).corr(),
    annot=True, cmap='RdBu', center=0
)
plt.title("Matriz de Correlación")
plt.show()

# ============================================
# SECCIÓN 3: CLASIFICACIÓN — RANDOM FOREST vs SVM vs KNN
# (mismo orden de variables que usaste: NO cambiar el orden)
# ============================================

FEATURES_CLASIF = ['age', 'gender', 'height(cm)', 'weight(kg)', 'ap_hi(mmHg)',
                    'ap_lo(mmHg)', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
TARGET_CLASIF = 'cardio'

X = df_cardiosclean[FEATURES_CLASIF]
y = df_cardiosclean[TARGET_CLASIF]
xvar_train, xvar_test, yvar_train, yvar_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEMILLA
)

# --- 3.1 Random Forest (SIN escalado, igual que en tu notebook) ---
modelo_rf = RandomForestClassifier(
    random_state=SEMILLA, n_estimators=100, max_depth=10,
    min_samples_split=5, min_samples_leaf=2
)
modelo_rf.fit(xvar_train, yvar_train)
precision_rf = round(error_score(modelo_rf, xvar_test, yvar_test), 3)
print(f"Precisión Random Forest: {precision_rf}")
calc_overfitting(modelo_rf, xvar_train, xvar_test, yvar_train, yvar_test, "Random Forest")

# --- Escalado compartido para SVM y KNN ---
xvar_train_esc, xvar_test_esc, scaler_clasificador = escalar(xvar_train, xvar_test)

# --- 3.2 SVM (con escalado; probability=True para poder dar % de riesgo) ---
modelo_svm = SVC(kernel="rbf", random_state=SEMILLA, probability=True)
modelo_svm.fit(xvar_train_esc, yvar_train)
precision_svm = round(error_score(modelo_svm, xvar_test_esc, yvar_test), 3)
print(f"Precisión SVM: {precision_svm}")
calc_overfitting(modelo_svm, xvar_train_esc, xvar_test_esc, yvar_train, yvar_test, "SVM")

# --- 3.3 KNN (con escalado; k=15 como valor por defecto razonable) ---
modelo_knn = KNeighborsClassifier(n_neighbors=15)
modelo_knn.fit(xvar_train_esc, yvar_train)
precision_knn = round(error_score(modelo_knn, xvar_test_esc, yvar_test), 3)
print(f"Precisión KNN: {precision_knn}")
calc_overfitting(modelo_knn, xvar_train_esc, xvar_test_esc, yvar_train, yvar_test, "KNN")

# --- Selección automática del modelo ganador ---
resultados_clasif = {
    'Random Forest': (modelo_rf, precision_rf, False),
    'SVM': (modelo_svm, precision_svm, True),
    'KNN': (modelo_knn, precision_knn, True),
}
nombre_ganador = max(resultados_clasif, key=lambda k: resultados_clasif[k][1])
modelo_clasificador_ganador, precision_ganadora, necesita_escalado = resultados_clasif[nombre_ganador]
print(f"\n🏆 Modelo de clasificación ganador: {nombre_ganador} (precisión={precision_ganadora})")

matriz_confusion(
    modelo_clasificador_ganador,
    xvar_train_esc if necesita_escalado else xvar_train,
    xvar_test_esc if necesita_escalado else xvar_test,
    yvar_train, yvar_test, nombre_ganador
)

# ============================================
# SECCIÓN 4: REGRESIÓN — RANDOM FOREST REGRESSOR vs MLP
# (predice ap_hi a partir de las otras variables, incluyendo cardio)
# ============================================

FEATURES_REG = ['age', 'gender', 'height(cm)', 'weight(kg)', 'ap_lo(mmHg)',
                 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
TARGET_REG = 'ap_hi(mmHg)'

Xr = df_cardiosclean[FEATURES_REG]
yr = df_cardiosclean[TARGET_REG]
xr_train, xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=TEST_SIZE, random_state=SEMILLA
)
xr_train_esc, xr_test_esc, scaler_regresor = escalar(xr_train, xr_test)

modelo_rfr = RandomForestRegressor(random_state=SEMILLA)
modelo_rfr.fit(xr_train_esc, yr_train)
r2_rfr = round(error_score(modelo_rfr, xr_test_esc, yr_test), 3)
print(f"R² Random Forest Regressor: {r2_rfr}")

modelo_mlp = MLPRegressor(random_state=SEMILLA, max_iter=500)
modelo_mlp.fit(xr_train_esc, yr_train)
r2_mlp = round(error_score(modelo_mlp, xr_test_esc, yr_test), 3)
print(f"R² MLP Regressor: {r2_mlp}")

resultados_reg = {
    'Random Forest Regressor': (modelo_rfr, r2_rfr),
    'MLP Regressor': (modelo_mlp, r2_mlp),
}
nombre_ganador_reg = max(resultados_reg, key=lambda k: resultados_reg[k][1])
modelo_regresor_ganador, r2_ganador = resultados_reg[nombre_ganador_reg]
print(f"\n🏆 Modelo de regresión ganador: {nombre_ganador_reg} (R²={r2_ganador})")

# ============================================
# SECCIÓN 6: CONGELAR Y EXPORTAR LOS MODELOS GANADORES
# ============================================

joblib.dump(modelo_clasificador_ganador, 'models/clasificador_cardio.pkl')
if necesita_escalado:
    joblib.dump(scaler_clasificador, 'models/scaler_clasificador.pkl')

joblib.dump(modelo_regresor_ganador, 'models/regresor_presion.pkl')
joblib.dump(scaler_regresor, 'models/scaler_regresor.pkl')

print("\n🎯 Archivos generados en /content/models/ y /content/data/:")
print(" - models/clasificador_cardio.pkl   (ganador:", nombre_ganador, ")")
if necesita_escalado:
    print(" - models/scaler_clasificador.pkl   (requerido porque el ganador NO fue Random Forest)")
print(" - models/regresor_presion.pkl      (ganador:", nombre_ganador_reg, ")")
print(" - models/scaler_regresor.pkl")
print(" - data/cardio_clean.csv            (dataset real para las gráficas del dashboard)")
print("\n👉 Descarga las carpetas 'models/' y 'data/' completas y súbelas a tu repo de GitHub,")
print("   en la raíz, junto a tu app.py.")
