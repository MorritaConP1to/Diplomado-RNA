"""
╔══════════════════════════════════════════════════════════════════════╗
║          MACHOTE DE MACHINE LEARNING — Funciones Reutilizables       ║
║          Diplomado en Redes Neuronales y Deep Learning — UAEM        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ¿CÓMO IMPORTAR ESTE ARCHIVO EN TU NOTEBOOK?                        ║
║  ─────────────────────────────────────────────────────────────────  ║
║  Pon estas líneas al inicio de tu notebook (Celda 1):               ║
║                                                                      ║
║      import sys                                                      ║
║      sys.path.append(r"C:\ruta\exacta\a\Machote_ML")               ║
║      from machote_ML import *                                        ║
║                                                                      ║
║  Reemplaza la ruta con donde guardaste ESTE archivo.                ║
║  Puedes ver la ruta exacta en VSCode haciendo clic derecho          ║
║  sobre el archivo → "Copy Path"                                     ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ORDEN RECOMENDADO PARA CUALQUIER PROYECTO DE ML:                   ║
║  ─────────────────────────────────────────────────────────────────  ║
║                                                                      ║
║  1. cargar_datos()          → obtener X e Y                         ║
║  2. resumen_rapido()        → ¿qué tengo? ¿hay NaN? ¿clases?       ║
║  3. ver_distribucion()      → ¿cómo se ven los valores?             ║
║  4. ver_correlacion_con_y() → ¿qué columnas importan?               ║
║  5. ver_mapa_calor()        → ¿hay columnas redundantes?            ║
║  6. seleccionar_features()  → quedarse solo con las útiles          ║
║  7. dividir_datos()         → separar train y test                  ║
║  8. preparar_datos()        → escalar O binarizar (no ambos)        ║
║  9. entrenar()              → fit() del modelo                      ║
║  10. evaluar()              → medir accuracy, ver matriz confusión  ║
║                                                                      ║
║  ⚠ REGLAS QUE NUNCA SE ROMPEN:                                      ║
║  - Siempre explorar ANTES de modelar (pasos 2-5)                    ║
║  - Nunca usar X_test para entrenar                                   ║
║  - Nunca hacer fit() del scaler con X_test                          ║
║  - Si usas MPNeuron: binarizar. Si usas otro modelo: escalar.       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── Imports necesarios ───────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    mean_squared_error,
    r2_score
)


# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 1 — CARGAR DATOS
#  ¿Cuándo usar cada función?
#    - cargar_desde_sklearn → cuando practiques con datasets de ejemplo
#    - cargar_desde_uci     → cuando el dataset venga de UCI sin descargarlo
#    - cargar_desde_csv     → cuando tengas un archivo .csv en tu máquina
# ════════════════════════════════════════════════════════════════════════════

def cargar_desde_sklearn(dataset_func):
    """
    Carga un dataset que ya viene dentro de scikit-learn.
    No necesita internet ni archivos externos.

    DATASETS DISPONIBLES (importar de sklearn.datasets):
        load_breast_cancer → clasificación binaria  (maligno/benigno)
        load_iris          → clasificación 3 clases (tipos de flor)
        load_wine          → clasificación 3 clases (tipos de vino)
        load_diabetes      → regresión (nivel de diabetes, número continuo)

    PARÁMETROS:
        dataset_func → la función de sklearn, SIN paréntesis
                       Ejemplo: load_breast_cancer  (no load_breast_cancer())

    REGRESA:
        X            → DataFrame con las características (columnas = features)
        Y            → array numpy con las etiquetas numéricas (0, 1, 2...)
        nombres_clases → lista de strings con el nombre de cada clase

    EJEMPLO DE USO:
        from sklearn.datasets import load_breast_cancer
        X, Y, clases = cargar_desde_sklearn(load_breast_cancer)
    """
    dataset = dataset_func()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    Y = dataset.target
    nombres_clases = list(dataset.target_names)

    print(f"✅ Dataset cargado: {dataset_func.__name__}")
    print(f"   Muestras (filas):       {X.shape[0]}")
    print(f"   Características (cols): {X.shape[1]}")
    print(f"   Clases:                 {nombres_clases}")
    print(f"   Etiquetas únicas en Y:  {np.unique(Y)}")

    return X, Y, nombres_clases


def cargar_desde_uci(uci_id, columna_y, mapeo_clases=None):
    """
    Carga un dataset desde UCI Machine Learning Repository.
    Requiere internet. No guarda archivos en tu máquina.

    PARÁMETROS:
        uci_id        → número de identificación del dataset en la página de UCI
                        Ejemplo: 17 para Breast Cancer Wisconsin
                        (lo ves en la URL: archive.ics.uci.edu/dataset/17/...)

        columna_y     → nombre exacto de la columna que contiene el diagnóstico
                        ⚠ ATENCIÓN: UCI a veces usa mayúscula. Si marca KeyError,
                        haz primero: print(Y_raw.columns) para ver el nombre exacto.
                        Breast Cancer Wisconsin usa 'Diagnosis' (con D mayúscula)

        mapeo_clases  → diccionario para convertir texto a números
                        Solo necesario si Y tiene texto ('M', 'B', 'Yes', 'No'...)
                        Si Y ya son números, pon None.
                        Ejemplo: {'M': 0, 'B': 1}

    REGRESA:
        X → DataFrame con las características
        Y → array numpy con las etiquetas numéricas

    EJEMPLO DE USO:
        X, Y = cargar_desde_uci(17, 'Diagnosis', {'M': 0, 'B': 1})
    """
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=uci_id)
    X = dataset.data.features
    Y_raw = dataset.data.targets

    # Ver nombre exacto de columnas si hay problemas:
    # print("Columnas en Y:", Y_raw.columns.tolist())

    if mapeo_clases is not None:
        Y = Y_raw[columna_y].map(mapeo_clases).to_numpy()
        # Verificar que no quedaron NaN (indicaría que el mapeo tiene errores)
        if np.isnan(Y).any():
            print("⚠ ADVERTENCIA: hay NaN en Y después del mapeo.")
            print("  Verifica que el mapeo cubra todos los valores únicos:")
            print(f"  Valores en la columna: {Y_raw[columna_y].unique()}")
    else:
        Y = Y_raw[columna_y].to_numpy()

    print(f"✅ Dataset UCI id={uci_id} cargado")
    print(f"   Muestras:        {X.shape[0]}")
    print(f"   Características: {X.shape[1]}")
    print(f"   Columna Y:       '{columna_y}'")
    if mapeo_clases:
        print(f"   Mapeo aplicado:  {mapeo_clases}")

    return X, Y


def cargar_desde_csv(ruta, columna_y):
    """
    Carga datos desde un archivo CSV en tu máquina o desde una URL.

    ⚠ PROBLEMA COMÚN CON RUTAS:
        Windows usa diagonal invertida: C:\\Users\\Diana\\datos.csv
        Python a veces lo interpreta mal. Soluciones:
            - Usar r antes del string:  r"C:\\Users\\Diana\\datos.csv"
            - Usar diagonal normal:     "C:/Users/Diana/datos.csv"
            - Usar doble diagonal:      "C:\\\\Users\\\\Diana\\\\datos.csv"

    PARÁMETROS:
        ruta      → ruta al archivo CSV, o URL pública completa
        columna_y → nombre exacto de la columna que contiene la etiqueta/resultado
                    (la que quieres predecir)

    REGRESA:
        X → DataFrame con las características (todo excepto columna_y)
        Y → Serie de pandas con las etiquetas

    EJEMPLO DE USO:
        X, Y = cargar_desde_csv(r"C:/mis_datos/cancer.csv", "diagnostico")
        X, Y = cargar_desde_csv("https://ejemplo.com/datos.csv", "price")
    """
    df = pd.read_csv(ruta)

    if columna_y not in df.columns:
        print(f"❌ ERROR: la columna '{columna_y}' no existe.")
        print(f"   Columnas disponibles: {df.columns.tolist()}")
        return None, None

    X = df.drop(columns=[columna_y])
    Y = df[columna_y]

    print(f"✅ CSV cargado desde: {ruta}")
    print(f"   Muestras:        {X.shape[0]}")
    print(f"   Características: {X.shape[1]}")
    print(f"   Columna Y:       '{columna_y}'")
    print(f"   Valores únicos:  {Y.unique()[:10]}")  # máx 10 para no saturar

    return X, Y


# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 2 — EXPLORAR DATOS (EDA)
#
#  EDA = Exploratory Data Analysis = "¿Qué tengo aquí?"
#
#  SIEMPRE hacer esto ANTES de entrenar cualquier modelo.
#  Muchos errores se evitan simplemente mirando los datos primero.
#
#  ORDEN SUGERIDO:
#    1. resumen_rapido()        → panorama general
#    2. ver_distribucion()      → cómo se ven valores individuales
#    3. ver_correlacion_con_y() → qué columnas importan para predecir
#    4. ver_mapa_calor()        → cuáles columnas se repiten entre sí
# ════════════════════════════════════════════════════════════════════════════

def resumen_rapido(X, Y):
    """
    Muestra un panorama completo del dataset de un vistazo.
    Es lo primero que debes correr después de cargar datos.

    DETECTA AUTOMÁTICAMENTE:
        - Cuántas muestras y características hay
        - Si hay valores faltantes (NaN) — esto rompe la mayoría de modelos
        - Cómo están distribuidas las clases (si hay desbalance)
        - Qué tipo de datos tiene cada columna

    ¿POR QUÉ IMPORTA EL DESBALANCE DE CLASES?
        Si tienes 90% clase A y 10% clase B, un modelo que siempre predice A
        tendrá 90% de accuracy sin aprender nada.
        Eso fue exactamente lo que pasó con el threshold=0 en la MPNeuron:
        predecía siempre "benigno" y acertaba el 62.7% sin aprender nada.

    PARÁMETROS:
        X → DataFrame con las características
        Y → array o Serie con las etiquetas

    REGRESA: nada, solo imprime el resumen
    """
    print("═" * 55)
    print("  RESUMEN DEL DATASET")
    print("═" * 55)
    print(f"  Muestras (filas):         {X.shape[0]}")
    print(f"  Características (cols):   {X.shape[1]}")
    print()

    # Valores faltantes — si hay, hay que manejarlos antes de entrenar
    faltantes = X.isnull().sum()
    cols_con_faltantes = faltantes[faltantes > 0]
    if len(cols_con_faltantes) == 0:
        print("  Valores faltantes (NaN): ninguno ✅")
    else:
        print(f"  ⚠ Valores faltantes en {len(cols_con_faltantes)} columnas:")
        for col, n in cols_con_faltantes.items():
            print(f"    {col}: {n} faltantes ({n/len(X)*100:.1f}%)")
    print()

    # Distribución de clases — detecta desbalance
    print("  Distribución de Y (etiquetas):")
    valores, conteos = np.unique(Y, return_counts=True)
    max_c = max(conteos)
    for v, c in zip(valores, conteos):
        barra = '█' * int(c / max_c * 25)
        alerta = " ⚠ DESBALANCE" if c / len(Y) > 0.75 else ""
        print(f"    Clase {v}: {c:4d} ({c/len(Y)*100:.1f}%) {barra}{alerta}")
    print()

    # Tipos de datos
    tipos = X.dtypes.value_counts()
    print("  Tipos de datos en X:")
    for tipo, cantidad in tipos.items():
        print(f"    {str(tipo):<12}: {cantidad} columnas")
    print("═" * 55)


def ver_distribucion(X, columna, Y=None):
    """
    Grafica cómo se distribuyen los valores de una columna específica.

    ¿POR QUÉ ES ÚTIL ANTES DE MODELAR?
        - Si la distribución está muy sesgada (todo en un lado), puede afectar
          el escalado y la binarización.
        - Si las clases se separan claramente en el histograma, esa
          característica es muy informativa para el modelo.
        - Si las clases se mezclan completamente, esa característica
          probablemente no ayude mucho.

    PARÁMETROS:
        X       → DataFrame con los datos
        columna → nombre de la columna a graficar (string)
        Y       → array con etiquetas (opcional)
                  Si se pone, colorea el histograma por clase.
                  Así puedes ver si esa columna distingue bien las clases.

    EJEMPLO DE USO:
        ver_distribucion(X, 'mean radius')           # sin separar por clase
        ver_distribucion(X, 'mean radius', Y)        # separado por clase ← más útil
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    if Y is not None:
        clases = np.unique(Y)
        colores = ['#3B8BD4', '#D85A30', '#2ca02c', '#9467bd']
        for i, clase in enumerate(clases):
            valores = X[columna][Y == clase]
            ax.hist(valores, bins=30, alpha=0.6,
                    label=f'Clase {clase} (n={sum(Y==clase)})',
                    color=colores[i % len(colores)])
        ax.legend()
        ax.set_title(f'Distribución de: {columna}\n(separado por clase)')
    else:
        ax.hist(X[columna], bins=30, color='#3B8BD4', alpha=0.8)
        ax.set_title(f'Distribución de: {columna}')

    ax.set_xlabel(columna)
    ax.set_ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()


def ver_correlacion_con_y(X, Y, umbral=0.5):
    """
    Grafica qué tan relacionada está cada característica con Y.
    Nos ayuda a decidir qué columnas son útiles y cuáles son ruido.

    CÓMO INTERPRETAR:
        Barra larga hacia la derecha (+) → valor alto = Y alto
        Barra larga hacia la izquierda (-) → valor alto = Y bajo
        Barra corta (cerca de 0) → esa columna no distingue las clases

        |correlación| > 0.7  → muy útil
        |correlación| 0.5-0.7 → útil
        |correlación| 0.2-0.5 → depende del caso
        |correlación| < 0.2  → probablemente ruido

    ⚠ ADVERTENCIA IMPORTANTE — EL BUG DE LOS BINARIOS:
        Si la mayoría de correlaciones son NEGATIVAS (barras a la izquierda),
        significa que valores ALTOS en las características corresponden a
        valores BAJOS en Y (ej: valores altos → maligno=0).
        En ese caso, si vas a usar MPNeuron, necesitas labels=[1, 0]
        (invertidos) en binarizar_datos().
        La función binarizar_datos() lo hace automáticamente,
        pero es bueno que lo entiendas mirando esta gráfica primero.

    PARÁMETROS:
        X       → DataFrame con las características
        Y       → array con las etiquetas numéricas
        umbral  → correlación mínima (absoluta) para resaltar en color
                  Default 0.5. Sube si quedan demasiadas, baja si muy pocas.

    REGRESA:
        correlaciones → Serie de pandas con la correlación de cada columna con Y
                        Úsala después en binarizar_datos() o seleccionar_features()
    """
    df_temp = X.copy()
    df_temp['__Y__'] = Y
    correlaciones = df_temp.corr()['__Y__'].drop('__Y__').sort_values(key=abs, ascending=False)

    # Gráfica
    fig, ax = plt.subplots(figsize=(8, max(4, len(correlaciones) * 0.3)))
    colors = ['#D85A30' if abs(v) >= umbral else '#aaaaaa' for v in correlaciones]
    correlaciones.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
    ax.axvline(x=0,       color='gray',  linewidth=0.8)
    ax.axvline(x=umbral,  color='green', linewidth=1.2, linestyle='--',
               label=f'umbral ±{umbral}')
    ax.axvline(x=-umbral, color='green', linewidth=1.2, linestyle='--')
    ax.set_title(f'Correlación de cada característica con Y\n(naranja = |corr| ≥ {umbral}, útiles)')
    ax.set_xlabel('Correlación con Y')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Resumen en texto
    utiles = correlaciones[correlaciones.abs() >= umbral]
    print(f"\nCaracterísticas útiles (|correlación| ≥ {umbral}):")
    for feat, val in utiles.items():
        direccion = "↑Y cuando sube" if val > 0 else "↓Y cuando sube"
        print(f"  {feat:<35} {val:+.4f}  ({direccion})")

    corr_prom = correlaciones.mean()
    print(f"\nCorrelación promedio: {corr_prom:+.4f}")
    if corr_prom < 0:
        print("  → Mayoría NEGATIVA: si usas MPNeuron, necesitarás labels=[1, 0]")
        print("    (binarizar_datos() lo hace automático si le pasas estas correlaciones)")
    else:
        print("  → Mayoría POSITIVA: labels=[0, 1] funciona normal")

    return correlaciones


def ver_mapa_calor(X, Y, top_n=15):
    """
    Grafica las correlaciones entre características entre sí (no con Y).
    Sirve para detectar columnas redundantes.

    ¿CUÁNDO HAY REDUNDANCIA?
        Si dos columnas tienen correlación 0.95 o más entre sí, básicamente
        están diciendo lo mismo con palabras distintas.
        Ejemplo: 'mean radius' y 'mean perimeter' — si la célula es grande,
        su radio es grande Y su perímetro es grande. Son casi la misma info.
        Puedes eliminar una sin perder mucho.

    ¿POR QUÉ IMPORTA?
        Columnas redundantes no mejoran el modelo, pero sí aumentan el tiempo
        de cómputo y pueden confundir modelos más sensibles.
        Para MPNeuron no es crítico, pero para redes neuronales sí importa.

    PARÁMETROS:
        X     → DataFrame con las características
        Y     → array con las etiquetas (se agrega al mapa como columna extra)
        top_n → cuántas características mostrar en el mapa
                Si pones None, muestra todas (puede quedar muy apretado con 30+)

    REGRESA: nada, solo la gráfica
    """
    df_temp = X.copy()
    df_temp['Y_diagnostico'] = Y

    if top_n is not None and top_n < X.shape[1]:
        corrs = df_temp.corr()['Y_diagnostico'].drop('Y_diagnostico')
        cols_top = corrs.abs().sort_values(ascending=False).head(top_n).index.tolist()
        df_temp = df_temp[cols_top + ['Y_diagnostico']]
        titulo = f'Mapa de calor — top {top_n} características + Y'
    else:
        titulo = 'Mapa de calor — todas las características + Y'

    n = len(df_temp.columns)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.65), max(6, n * 0.65)))
    sns.heatmap(
        df_temp.corr(),
        annot=True, fmt='.2f',
        cmap='RdBu_r',
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.3,
        ax=ax
    )
    ax.set_title(titulo + '\n(rojo=alta, blanco=ninguna, azul=negativa)', pad=15)
    plt.tight_layout()
    plt.show()


# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 3 — PREPARAR DATOS
#
#  REGLA DE ORO: siempre preparar train y test POR SEPARADO.
#  Si los preparas juntos, el test se contamina con info del train
#  y el modelo "hace trampa" → accuracy falso.
# ════════════════════════════════════════════════════════════════════════════

def seleccionar_features(X, Y, umbral=0.5):
    """
    Filtra X para quedarse solo con las características más correlacionadas con Y.
    Elimina el ruido y reduce el tamaño del problema.

    ¿CUÁNDO SUBIR O BAJAR EL UMBRAL?
        Si quedan demasiadas columnas (ej: 25 de 30) → sube el umbral a 0.6 o 0.7
        Si quedan muy pocas (ej: 3 de 30) → baja el umbral a 0.4 o 0.3
        No hay un número mágico: experimenta y compara el accuracy resultante.

    PARÁMETROS:
        X       → DataFrame completo con todas las características
        Y       → array con las etiquetas
        umbral  → correlación mínima (absoluta) para conservar una columna

    REGRESA:
        X_reducido       → DataFrame con solo las columnas útiles
        features_elegidas → lista de strings con los nombres de esas columnas
                           (guárdalos para saber qué usaste)
    """
    df_temp = X.copy()
    df_temp['__Y__'] = Y
    correlaciones = df_temp.corr()['__Y__'].drop('__Y__')
    features_elegidas = correlaciones[correlaciones.abs() >= umbral].index.tolist()

    if len(features_elegidas) == 0:
        print(f"⚠ Con umbral={umbral} no quedó ninguna característica.")
        print(f"  Intenta bajar el umbral.")
        return X, X.columns.tolist()

    print(f"✅ Selección con umbral={umbral}:")
    print(f"   Conservadas: {len(features_elegidas)} de {X.shape[1]} características")
    print(f"   Eliminadas:  {X.shape[1] - len(features_elegidas)} características")
    print(f"   Columnas conservadas: {features_elegidas}")

    return X[features_elegidas], features_elegidas


def dividir_datos(X, Y, proporcion_test=0.25, semilla=42, estratificar=True):
    """
    Separa los datos en train (entrenamiento) y test (evaluación).

    ¿POR QUÉ SEPARAR?
        Si entrenas y evalúas con los mismos datos, el modelo los "memoriza"
        y parece funcionar muy bien, pero falla con datos nuevos.
        Es como estudiar con el mismo examen que después te van a poner.

    ¿QUÉ ES LA SEMILLA (random_state)?
        La división es aleatoria, pero al fijar un número (la "semilla"),
        la aleatoriedad queda congelada. Tú y otra persona con semilla=42
        obtendrán exactamente la misma división. Esto hace los experimentos
        reproducibles (otra persona puede verificar tus resultados).
        Puedes usar cualquier número entero.

    ¿QUÉ ES estratificar?
        Con True: garantiza que train y test tengan la misma proporción
        de clases que el dataset original. Úsalo siempre en clasificación.
        Con False: división puramente aleatoria. Úsalo en regresión.

    PARÁMETROS:
        X               → DataFrame con características
        Y               → array con etiquetas
        proporcion_test → fracción que va al test (0.25 = 25% test)
        semilla         → número para reproducibilidad
        estratificar    → True para clasificación, False para regresión

    REGRESA:
        X_train, X_test, y_train, y_test
    """
    stratify = Y if estratificar else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y,
        test_size=proporcion_test,
        random_state=semilla,
        stratify=stratify
    )

    print(f"✅ Datos divididos (semilla={semilla}):")
    print(f"   Train: {len(X_train)} muestras ({(1-proporcion_test)*100:.0f}%)")
    print(f"   Test:  {len(X_test)} muestras ({proporcion_test*100:.0f}%)")
    if estratificar:
        vals_tr, cnts_tr = np.unique(y_train, return_counts=True)
        vals_te, cnts_te = np.unique(y_test,  return_counts=True)
        print(f"   Distribución train: { dict(zip(vals_tr.tolist(), cnts_tr.tolist())) }")
        print(f"   Distribución test:  { dict(zip(vals_te.tolist(), cnts_te.tolist())) }")

    return X_train, X_test, y_train, y_test


def escalar_datos(X_train, X_test, metodo='standard'):
    """
    Pone todos los valores en la misma escala numérica.
    USAR ESTO cuando el modelo NO es MPNeuron (regresión, SVM, redes neuronales).

    ¿POR QUÉ ESCALAR?
        Los modelos dan más peso a columnas con números grandes simplemente
        porque son más grandes, no porque sean más importantes.
        Ejemplo sin escalar: 'area' va de 143 a 2501, 'smoothness' de 0.05 a 0.16
        El modelo le da mucho más importancia a 'area' solo por ser grande.
        Después de escalar: ambas van en rangos comparables.

    MÉTODOS DISPONIBLES:
        'standard' → resta la media y divide entre desviación estándar
                     Resultado: valores centrados en 0, mayoría entre -3 y +3
                     Mejor para: regresión logística, SVM, redes neuronales

        'minmax'   → comprime todos los valores al rango [0, 1]
                     Mejor para: cuando necesitas que los valores sean positivos

    ⚠ REGLA IMPORTANTE:
        El scaler "aprende" la escala SOLO del train (fit_transform).
        Al test solo se le aplica esa escala ya aprendida (transform).
        Si hicieras fit con test, el scaler conocería datos que el modelo
        no debería ver → trampa → accuracy falso.

    PARÁMETROS:
        X_train → datos de entrenamiento
        X_test  → datos de prueba
        metodo  → 'standard' o 'minmax'

    REGRESA:
        X_train_scaled, X_test_scaled → arrays numpy escalados
    """
    if metodo == 'standard':
        scaler = StandardScaler()
    elif metodo == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("metodo debe ser 'standard' o 'minmax'")

    X_train_scaled = scaler.fit_transform(X_train)   # aprende + transforma
    X_test_scaled  = scaler.transform(X_test)         # solo transforma (sin aprender)

    print(f"✅ Datos escalados con método '{metodo}':")
    print(f"   X_train: {X_train_scaled.shape} | min={X_train_scaled.min():.2f}, max={X_train_scaled.max():.2f}")
    print(f"   X_test:  {X_test_scaled.shape}  | min={X_test_scaled.min():.2f}, max={X_test_scaled.max():.2f}")

    return X_train_scaled, X_test_scaled


def binarizar_datos(X_train, X_test, correlaciones):
    """
    Convierte valores continuos a 0 o 1.
    USAR ESTO SOLO para MPNeuron. Los demás modelos usan escalar_datos().

    ¿POR QUÉ SE NECESITA PARA MPNEURON?
        La MPNeuron solo sabe sumar. Si le das un valor de 1001 (área grande),
        esa sola columna domina la suma y el threshold no tiene sentido.
        Con binario, cada columna solo aporta 0 o 1 a la suma, sin importar
        cuánto valía originalmente.

    ⚠ EL BUG DEL INVERTIDO — MUY IMPORTANTE:
        Esta función necesita las correlaciones para detectar la dirección.
        Si las correlaciones son en su mayoría NEGATIVAS (como en Breast Cancer),
        significa que valores ALTOS en X corresponden a clase BAJA (maligno=0).
        En ese caso, el binarizado tiene que quedar:
            valor alto → 0  (probablemente maligno)
            valor bajo → 1  (probablemente benigno)
        Si lo pones al revés (valor alto → 1), la MPNeuron aprende exactamente
        lo contrario de lo correcto. Esto fue el bug del notebook original.
        Esta función lo detecta automáticamente con las correlaciones.

    PARÁMETROS:
        X_train      → DataFrame de entrenamiento
        X_test       → DataFrame de prueba
        correlaciones → Serie devuelta por ver_correlacion_con_y()
                        Se usa para decidir automáticamente la dirección

    REGRESA:
        X_train_bin, X_test_bin → DataFrames binarizados (valores 0 o 1)
    """
    corr_promedio = correlaciones.mean()

    if corr_promedio >= 0:
        labels = [0, 1]
        print("✅ Correlación promedio POSITIVA")
        print("   labels=[0, 1] → valor bajo=0, valor alto=1")
    else:
        labels = [1, 0]
        print("✅ Correlación promedio NEGATIVA")
        print("   labels=[1, 0] → valor alto=0, valor bajo=1  (invertido para compensar)")
        print("   Esto evita el bug del modelo que predice todo al revés.")

    X_train_bin = X_train.apply(pd.cut, bins=2, labels=labels)
    X_test_bin  = X_test.apply(pd.cut,  bins=2, labels=labels)

    print(f"   Shape final: {X_train_bin.shape}")

    return X_train_bin, X_test_bin


# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 4 — EVALUAR MODELOS
# ════════════════════════════════════════════════════════════════════════════

def evaluar_clasificacion(y_real, y_predicho, nombres_clases=None):
    """
    Mide qué tan bien predice un modelo de clasificación.
    Usar cuando Y son categorías (0/1, maligno/benigno, perro/gato).

    MÉTRICAS QUE MUESTRA:
        Accuracy  → de todos los casos, ¿qué % acerté?
                    Problema: si hay desbalance, puede ser engañoso.
                    (un modelo que dice siempre "benigno" tiene 62% de accuracy)

        Precision → de todos los que predije como positivo,
                    ¿cuántos realmente lo eran?
                    Alta precisión = pocos falsos positivos.

        Recall    → de todos los positivos reales,
                    ¿cuántos encontré?
                    Alto recall = pocos falsos negativos.
                    En medicina, recall alto es crítico: no puedes dejar
                    de detectar un cáncer real.

        F1        → balance entre precision y recall.
                    Útil cuando hay desbalance de clases.

    MATRIZ DE CONFUSIÓN:
        Cada celda dice cuántos casos caen en cada combinación:
        ┌─────────────────────┬──────────────────┐
        │ TN: predijo 0, era 0│ FP: predijo 1, era 0 ← PELIGROSO en medicina
        ├─────────────────────┼──────────────────┤
        │ FN: predijo 0, era 1│ TP: predijo 1, era 1 │
        └─────────────────────┴──────────────────┘

    PARÁMETROS:
        y_real         → etiquetas correctas (y_test)
        y_predicho     → predicciones del modelo
        nombres_clases → ['Maligno', 'Benigno'] (opcional, mejora legibilidad)

    REGRESA:
        acc → accuracy como float (útil para comparar modelos)
    """
    acc = accuracy_score(y_real, y_predicho)
    cm  = confusion_matrix(y_real, y_predicho)

    print(f"✅ Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print()
    print("Reporte completo:")
    print(classification_report(y_real, y_predicho, target_names=nombres_clases))

    etiquetas = nombres_clases if nombres_clases else [str(c) for c in np.unique(y_real)]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f'Pred. {e}' for e in etiquetas],
        yticklabels=[f'Real {e}'  for e in etiquetas],
        ax=ax
    )
    ax.set_title(f'Matriz de Confusión — Accuracy: {acc*100:.1f}%')
    plt.tight_layout()
    plt.show()

    return acc


def evaluar_regresion(y_real, y_predicho):
    """
    Mide qué tan bien predice un modelo de regresión.
    Usar cuando Y son números continuos (precios, temperatura, etc.)

    MÉTRICAS QUE MUESTRA:
        MSE  → Mean Squared Error (Error Cuadrático Medio)
               Promedio de (real - predicho)². Penaliza errores grandes.
               Unidades: el cuadrado de las unidades de Y (difícil de interpretar)

        RMSE → Raíz cuadrada del MSE.
               Mismas unidades que Y → más fácil de interpretar.
               Ejemplo: si Y son precios en pesos y RMSE=5000,
               el modelo se equivoca en promedio ±5000 pesos.

        R²   → Coeficiente de determinación.
               Qué fracción de la variación de Y explica el modelo.
               1.0 = perfecto, 0.0 = no explica nada, negativo = peor que nada.
               Es el más fácil de comunicar: "el modelo explica el X% de la variación"

    GRÁFICA Real vs Predicho:
        Si el modelo fuera perfecto, todos los puntos estarían sobre la línea roja.
        Puntos muy lejos de la línea = errores grandes.

    PARÁMETROS:
        y_real     → valores reales (y_test)
        y_predicho → predicciones del modelo

    REGRESA:
        dict con claves 'mse', 'rmse', 'r2'
    """
    mse  = mean_squared_error(y_real, y_predicho)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_real, y_predicho)

    print(f"  MSE:  {mse:.4f}  (error cuadrático medio — penaliza errores grandes)")
    print(f"  RMSE: {rmse:.4f} (raíz del error — mismas unidades que Y)")
    print(f"  R²:   {r2:.4f}  (1.0=perfecto, 0.0=no explica nada)")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_real, y_predicho, alpha=0.5, color='#3B8BD4', s=20)
    lims = [min(float(np.min(y_real)), float(np.min(y_predicho))),
            max(float(np.max(y_real)), float(np.max(y_predicho)))]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Predicción perfecta')
    ax.set_xlabel('Valores reales')
    ax.set_ylabel('Valores predichos')
    ax.set_title(f'Real vs Predicho — R²: {r2:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return {'mse': mse, 'rmse': rmse, 'r2': r2}


def comparar_modelos(resultados, metrica='accuracy'):
    """
    Grafica una comparación visual de varios modelos.
    Muy útil al final de un proyecto para elegir el mejor modelo.

    PARÁMETROS:
        resultados → diccionario con nombre del modelo y su métrica
                     Ejemplo: {'MPNeuron': 0.91, 'Reg. Logística': 0.96}
        metrica    → nombre de la métrica para el eje Y (solo para el título)

    EJEMPLO DE USO:
        resultados = {
            'MPNeuron': acc_mp,
            'Reg. Logística': acc_lr,
            'Árbol de Decisión': acc_dt
        }
        comparar_modelos(resultados)
    """
    nombres = list(resultados.keys())
    valores = list(resultados.values())
    mejor_val = max(valores)

    fig, ax = plt.subplots(figsize=(max(6, len(nombres) * 1.8), 4))
    colors = ['#D85A30' if v == mejor_val else '#3B8BD4' for v in valores]
    bars = ax.bar(nombres, [v * 100 for v in valores], color=colors, edgecolor='none')

    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel(f'{metrica} (%)')
    ax.set_ylim(0, 115)
    ax.set_title(f'Comparación de modelos — {metrica}')
    ax.axhline(y=mejor_val * 100, color='orange', linestyle='--', alpha=0.4, linewidth=1)
    plt.tight_layout()
    plt.show()

    mejor_nombre = max(resultados, key=resultados.get)
    print(f"✅ Mejor modelo: {mejor_nombre} con {resultados[mejor_nombre]*100:.1f}%")


# ════════════════════════════════════════════════════════════════════════════
#  BLOQUE 5 — UTILIDADES
# ════════════════════════════════════════════════════════════════════════════

def configurar_path(carpeta_machote):
    """
    Agrega la carpeta del machote al PATH de Python para que puedas importarlo.
    Llama esto UNA VEZ al inicio de tu notebook si tienes problemas de importación.

    PARÁMETROS:
        carpeta_machote → ruta a la carpeta donde está machote_ML.py
                          Ejemplo en Windows: r"C:/Users/Diana/Machote_ML"
                          (usa diagonal normal o r"..." para evitar problemas)

    EJEMPLO DE USO al inicio de tu notebook:
        import sys
        sys.path.append(r"C:/Users/Diana/DIPLOMADO-RNA/Machote_ML")
        from machote_ML import *
    """
    import sys, os
    if carpeta_machote not in sys.path:
        sys.path.append(carpeta_machote)
        print(f"✅ Path agregado: {carpeta_machote}")
    else:
        print(f"ℹ Path ya estaba configurado: {carpeta_machote}")

    if not os.path.exists(os.path.join(carpeta_machote, 'machote_ML.py')):
        print(f"⚠ No se encontró machote_ML.py en {carpeta_machote}")
        print(f"   Verifica que la ruta sea correcta.")


print("✅ machote_ML cargado correctamente")
print()
print("Funciones disponibles:")
print("  CARGAR:   cargar_desde_sklearn | cargar_desde_uci | cargar_desde_csv")
print("  EXPLORAR: resumen_rapido | ver_distribucion | ver_correlacion_con_y | ver_mapa_calor")
print("  PREPARAR: seleccionar_features | dividir_datos | escalar_datos | binarizar_datos")
print("  EVALUAR:  evaluar_clasificacion | evaluar_regresion | comparar_modelos")


# ════════════════════════════════════════════════════════════════════════════
#  CÓMO IMPORTAR DESDE CUALQUIER COMPUTADORA (ruta relativa)
#  ─────────────────────────────────────────────────────────────────────────
#  El problema con rutas absolutas (C:/Users/Diana/...) es que cambian
#  en cada computadora. En la escuela tendrás una ruta diferente.
#
#  Solución: usar una ruta RELATIVA que se calcula desde donde está tu notebook.
#
#  PON ESTO AL INICIO DE CADA NOTEBOOK EN LUGAR DE LA RUTA HARDCODEADA:
#
#      import sys, os
#
#      # Calcula la ruta del machote relativa a este notebook
#      # __file__ no funciona en notebooks, así usamos una ruta desde la raíz del repo
#      ruta_machote = os.path.abspath(
#          os.path.join(os.getcwd(), '..', '..', 'Machote_ML')
#      )
#      # Ajusta los '..' según cuántas carpetas arriba esté Machote_ML
#      # Si tu notebook está en Modulo-2/Ejercicios/ y Machote_ML está en la raíz:
#      #   2 niveles arriba → dos '..'
#
#      if ruta_machote not in sys.path:
#          sys.path.append(ruta_machote)
#      from machote_ML import *
#
#  Así funciona en tu casa, en la escuela, en cualquier computadora,
#  siempre que la estructura de carpetas del repo sea la misma.
# ════════════════════════════════════════════════════════════════════════════
