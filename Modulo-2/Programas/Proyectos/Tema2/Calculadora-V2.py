# ==========================================
# CALCULADORA DE FITNESS Y SALUD PERSONAL
# ==========================================

def calcular_imc(peso_kg, altura_m):
    """
    IMC = peso / (altura^2)
    :param peso_kg:
    :param altura_m:
    """
    return peso_kg / (altura_m ** 2)

def es_peso_saludable(imc):
    """
    Rango saludable 18.5 - 24.9
    :param imc:
    """
    return 18.5 <= imc <= 24.9

def tiene_sobrepeso(imc):
    """
    
    :param imc:
    """
    return imc >= 25

def tiene_bajo_peso(imc):
    """
    
    :param imc:
    """
    return imc < 18.5

def calcular_calorias_diarias(peso_kg, altura_cm, edad, es_hombre):
    """
    Fórmula de Harris-Benedict con truco booleano :param peso_kg: :param altura_cm:
    :param edad: :param es_hombre:
"""
    calorias_hombre = 88.362 + (13.397 * peso_kg) + (4.799 * altura_cm) - (5.677 * edad)
    calorias_mujer = 447.593 + (9.247 * peso_kg) + (3.098 * altura_cm) - (4.330 * edad)
    # True=1, False=0, así elegimos la fórmula correcta
    return es_hombre * calorias_hombre + (1 - es_hombre) * calorias_mujer

def calcular_agua_diaria(peso_kg):
    """
    35 ml por kg, resultado en litros
    :param peso_kg:
"""
    return (peso_kg * 35) / 1000

def calcular_ritmo_cardiaco_maximo(edad):
    """
    
    :param edad:
    """

    return 220 - edad

# --------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------
print("Bienvenido a la Calculadora de Fitness y Salud Personal")
print("Por favor, ingresa tus datos para calcular tus métricas corporales.\n")

nombre = input("Ingresa tu nombre: ")
peso = float(input("Ingresa tu peso en kg: "))
altura_cm = float(input("Ingresa tu altura en cm: "))
edad = int(input("Ingresa tu edad: "))
sexo_str = input("Ingresa tu sexo (hombre/mujer): ").lower()

# Convertir sexo a booleano
es_hombre = (sexo_str == "hombre")

# Calcular altura en metros para el IMC
altura_m = altura_cm / 100

print("\nCalculando tus métricas corporales...")
print(f"Muy bien {nombre}, aquí están tus datos:")

# IMC
imc = calcular_imc(peso, altura_m)
print(f"Tu IMC es: {imc:.2f}")

# Clasificación del peso
if es_peso_saludable(imc):
    print("Tu peso está en un rango saludable.")
elif tiene_sobrepeso(imc):
    print("Tienes sobrepeso.")
elif tiene_bajo_peso(imc):
    print("Tienes bajo peso.")

# Calorías (pasamos altura en cm)
calorias = calcular_calorias_diarias(peso, altura_cm, edad, es_hombre)
print(f"Calorías diarias recomendadas: {calorias:.2f} kcal")

# Agua
agua = calcular_agua_diaria(peso)
print(f"Litros de agua recomendados al día: {agua:.2f} Lts")

# Ritmo cardíaco
ritmo = calcular_ritmo_cardiaco_maximo(edad)
print(f"Tu ritmo cardíaco máximo debe ser de: {ritmo} ppm")

print("\nValores solo de referencia\nConsulte a su médico")
