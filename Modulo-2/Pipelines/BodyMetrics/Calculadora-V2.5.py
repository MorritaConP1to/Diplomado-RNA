# ==========================================
# CALCULADORA DE FITNESS Y SALUD PERSONAL - V2
# ==========================================
# Proyecto del Módulo 2 — Python para IA
# Tema 2: Operadores en Python
#
# ¿Qué hace este programa?
# Calcula métricas de salud y fitness personales:
#   - IMC (Índice de Masa Corporal)
#   - Calorías diarias recomendadas (Harris-Benedict)
#   - Litros de agua recomendados al día
#   - Ritmo cardíaco máximo
#
# Conceptos de Python utilizados:
#   - Operadores aritméticos (+, -, *, /, **)
#   - Operadores de comparación (>=, <=, <)
#   - Operadores lógicos (and) → en versión V1
#   - Operadores booleanos como números (True=1, False=0) → truco en V2
#   - f-strings con formato numérico (:.2f)
#   - Conversión de tipos (float(), int(), bool())
# ==========================================


# ══════════════════════════════════════════════════════════════
#  FUNCIONES DE CÁLCULO
# ══════════════════════════════════════════════════════════════

def calcular_imc(peso_kg, altura_m):
    """
    Calcula el Índice de Masa Corporal (IMC).

    Parámetros:
        peso_kg (float): peso en kilogramos
        altura_m (float): altura en metros (NO centímetros)

    Retorna:
        float: el IMC calculado (sin redondear)

    Fórmula:
        IMC = peso / altura²

    Ejemplo:
        calcular_imc(60, 1.65) → 60 / (1.65²) → 60 / 2.7225 → 22.04

    Concepto: operador ** es potencia en Python
        altura_m ** 2 eleva al cuadrado
        Equivalente a: altura_m * altura_m
    """
    return peso_kg / (altura_m ** 2)


def clasificar_imc(imc):
    """
    Determina la categoría del IMC según la clasificación de la OMS.

    Parámetros:
        imc (float): el IMC calculado

    Retorna:
        str: categoría del IMC

    Clasificación OMS:
        < 18.5  → Bajo peso
        18.5-24.9 → Peso normal (saludable)
        25.0-29.9 → Sobrepeso
        >= 30   → Obesidad

    Concepto: operadores de comparación encadenados
        18.5 <= imc <= 24.9 equivale a (imc >= 18.5) and (imc <= 24.9)
        Python permite encadenar comparaciones directamente, lo cual
        es más legible y se parece más a la notación matemática.
    """
    if imc < 18.5:
        return "Bajo peso"
    elif 18.5 <= imc <= 24.9:
        return "Peso normal (saludable)"
    elif 25.0 <= imc <= 29.9:
        return "Sobrepeso"
    else:
        return "Obesidad"


def es_peso_saludable(imc):
    """
    Determina si el IMC está en rango saludable (18.5 - 24.9).

    Parámetros:
        imc (float): el IMC calculado

    Retorna:
        bool: True si está en rango saludable, False si no

    Concepto: la comparación encadenada devuelve directamente True o False,
    así que podemos usar 'return' directamente sin un if/else.
    """
    return 18.5 <= imc <= 24.9


def tiene_sobrepeso(imc):
    """
    Determina si hay sobrepeso (IMC >= 25).

    Parámetros:
        imc (float): el IMC calculado

    Retorna:
        bool: True si hay sobrepeso
    """
    return imc >= 25


def tiene_bajo_peso(imc):
    """
    Determina si hay bajo peso (IMC < 18.5).

    Parámetros:
        imc (float): el IMC calculado

    Retorna:
        bool: True si hay bajo peso
    """
    return imc < 18.5


def calcular_calorias_diarias(peso_kg, altura_cm, edad, es_hombre):
    """
    Calcula las calorías basales diarias usando la Fórmula de Harris-Benedict.

    Parámetros:
        peso_kg (float): peso en kilogramos
        altura_cm (float): altura en centímetros (NO metros)
        edad (int): edad en años
        es_hombre (bool): True si es hombre, False si es mujer

    Retorna:
        float: calorías basales diarias recomendadas (en kcal)

    Fórmulas:
        Hombre: 88.362 + (13.397 × peso) + (4.799 × altura) - (5.677 × edad)
        Mujer:  447.593 + (9.247 × peso)  + (3.098 × altura) - (4.330 × edad)

    ⭐ TRUCO BOOLEANO — Concepto clave de este proyecto:
        En Python, True == 1 y False == 0 matemáticamente.
        Esto permite elegir entre dos fórmulas sin usar if/else:

        es_hombre * calorias_hombre + (1 - es_hombre) * calorias_mujer

        Si es_hombre = True (1):
            1 * calorias_hombre + (1-1) * calorias_mujer
            = calorias_hombre + 0
            = calorias_hombre  ✓

        Si es_hombre = False (0):
            0 * calorias_hombre + (1-0) * calorias_mujer
            = 0 + calorias_mujer
            = calorias_mujer  ✓

        Este truco es elegante pero puede ser confuso para quien lee el código.
        En producción es preferible usar if/else para mayor claridad.
    """
    calorias_hombre = 88.362  + (13.397 * peso_kg) + (4.799 * altura_cm) - (5.677 * edad)
    calorias_mujer  = 447.593 + (9.247  * peso_kg) + (3.098 * altura_cm) - (4.330 * edad)

    # Selección de fórmula usando el truco booleano
    return es_hombre * calorias_hombre + (1 - es_hombre) * calorias_mujer


def calcular_agua_diaria(peso_kg):
    """
    Calcula los litros de agua recomendados al día.

    Parámetros:
        peso_kg (float): peso en kilogramos

    Retorna:
        float: litros de agua recomendados

    Fórmula:
        35 ml por cada kg de peso → resultado en litros (÷ 1000)

    Ejemplo:
        calcular_agua_diaria(60) → (60 × 35) / 1000 → 2.1 litros

    Concepto: operadores aritméticos * (multiplicación) y / (división)
    """
    return (peso_kg * 35) / 1000


def calcular_ritmo_cardiaco_maximo(edad):
    """
    Calcula el ritmo cardíaco máximo recomendado.

    Parámetros:
        edad (int): edad en años

    Retorna:
        int: ritmo cardíaco máximo en pulsaciones por minuto (ppm)

    Fórmula de Tanaka (simplificada):
        FCmax = 220 - edad

    Ejemplo:
        calcular_ritmo_cardiaco_maximo(30) → 220 - 30 → 190 ppm

    Concepto: operador - (resta)
    """
    return 220 - edad


# ══════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 50)
    print("  CALCULADORA DE FITNESS Y SALUD PERSONAL")
    print("=" * 50)
    print("Por favor, ingresa tus datos.\n")

    # ── Entrada de datos ──────────────────────────────────────
    nombre     = input("Ingresa tu nombre: ").strip()
    peso       = float(input("Ingresa tu peso en kg: "))

    # La altura se pide en cm (más intuitivo para el usuario)
    # pero calcular_imc necesita metros → dividimos entre 100
    altura_cm  = float(input("Ingresa tu altura en cm: "))
    altura_m   = altura_cm / 100   # conversión: 165 cm → 1.65 m

    edad       = int(input("Ingresa tu edad: "))

    # Convertimos el string del sexo a booleano para usar el truco
    # .lower() convierte a minúsculas para aceptar "Hombre", "HOMBRE", etc.
    sexo_str   = input("Ingresa tu sexo (hombre/mujer): ").lower().strip()
    es_hombre  = (sexo_str == "hombre")   # True si es hombre, False si es mujer

    # ── Cálculos ──────────────────────────────────────────────
    print(f"\nCalculando tus métricas corporales...")
    print(f"Muy bien {nombre}, aquí están tus datos:\n")

    # IMC (usa altura en metros)
    imc = calcular_imc(peso, altura_m)

    # Clasificación del IMC
    categoria = clasificar_imc(imc)

    # Calorías (usa altura en cm para la fórmula de Harris-Benedict)
    calorias = calcular_calorias_diarias(peso, altura_cm, edad, es_hombre)

    # Agua diaria
    agua = calcular_agua_diaria(peso)

    # Ritmo cardíaco
    ritmo = calcular_ritmo_cardiaco_maximo(edad)

    # ── Mostrar resultados ────────────────────────────────────
    # :.2f en f-strings formatea el número con 2 decimales
    print(f"  IMC:                     {imc:.2f}  → {categoria}")
    print(f"  Calorías recomendadas:   {calorias:.2f} kcal/día")
    print(f"  Agua recomendada:        {agua:.2f} litros/día")
    print(f"  Ritmo cardíaco máximo:   {ritmo} ppm")

    print("\n⚠️  Valores de referencia únicamente.")
    print("   Consulta a tu médico para diagnóstico profesional.")
