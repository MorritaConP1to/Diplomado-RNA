##Calculadora de metricas corporales
## Calcular indicadores de dalud y fitnes
8
# ==========================================

# CALCULADORA DE FITNESS Y SALUD PERSONAL

# ==========================================
 

def calcular_imc(peso_kg, altura_m):
    """
    Calcula el Índice de Masa Corporal (IMC).
    Fórmula: IMC = peso / (altura^2)
    Parámetros:
    peso_kg (float): Peso en kilogramos
    altura_m (float): Altura en metros
    Retorna:
    float: El IMC calculado
    """
    imc = peso_kg / pow(altura_m,2)
    return imc 

def es_peso_saludable(imc):
    """
    Determina si el IMC está en rango saludable (18.5 - 24.9).
    Parámetro:
    imc (float): Índice de Masa Corporal
    Retorna:
    bool: True si está en rango saludable, False si no
    """
    # Operadores de comparación y lógicos
    return imc >= 18.5 and imc <= 24.9 

def tiene_sobrepeso(imc):
    """
    Determina si hay sobrepeso (IMC >= 25).
    """
    # TU CÓDIGO AQUÍ
    return imc >= 25

def tiene_bajo_peso(imc):
    """
    Determina si hay bajo peso (IMC < 18.5).
    """
    # TU CÓDIGO AQUÍ
    return imc < 18.5

def calcular_calorias_diarias(peso_kg, altura_cm, edad, es_hombre):
    """
    Calcula las calorías diarias recomendadas usando Fórmula de Harris-Benedict.
    Parámetros:
    peso_kg (float): Peso en kg
    altura_cm (float): Altura en cm
    edad (int): Edad en años
    es_hombre (bool): True si es hombre, False si es mujer
    Retorna:
    float: Calorías diarias recomendadas
    """
    # Operadores aritméticos y booleanos
    # Fórmula para hombres: 88.362 + (13.397 × peso) + (4.799 × altura) - (5.677 × edad)
    # Fórmula para mujeres: 447.593 + (9.247 × peso) + (3.098 × altura) - (4.330 × edad)
    # Usa el hecho de que True=1 y False=0
    # TU CÓDIGO AQUÍ
    if es_hombre in "h":
            calorias = 88.362 + (13.397 * peso_kg) + (4.799 * altura_cm) - (5.677 * edad)
    else:
            calorias = 447.593 + (9.247 * peso_kg) + (3.098 * altura_cm) - (4.330 * edad)
    return calorias

def calcular_agua_diaria(peso_kg):
    """
    Calcula litros de agua recomendados al día (35ml por kg de peso).
    """
    # TU CÓDIGO AQUÍ
    litros_agua = (peso_kg * 35) / 1000
    return litros_agua
 
def calcular_ritmo_cardiaco_maximo(edad):
    """
    Calcula el ritmo cardíaco máximo (220 - edad).
    """
    return 220 - edad

print("Bienvenido a la Calculadora de Fitness y Salud Personal")
print("Por favor, ingresa tus datos para calcular tus métricas corporales.")
nombre = input("Ingresa tu nombre: ")
peso = float(input("Ingresa tu peso en kg: "))
altura = float(input("Ingresa tu altura en cm: "))/100 # Convertir cm a m
edad = int(input("Ingresa tu edad: "))
sexo = input("Ingresa tu sexo (hombre/mujer): ")
print("\nCalculando tus métricas corporales...")
print("Muy bien ",nombre+" Aca estan tus datos:")
imc = calcular_imc(peso, altura)
print("Tu IMC es:", round(imc,2))
if es_peso_saludable(imc):
    print("Tu peso está en un rango saludable.")
elif tiene_sobrepeso(imc):
    print("Tienes sobrepeso.")
elif tiene_bajo_peso(imc):
    print("Tienes bajo peso.")

calorias_diarias = calcular_calorias_diarias(peso, altura, edad, sexo)
print("Calorías diarias recomendadas:", str(round(calorias_diarias,2)) + " Kcl")
agua_diaria = calcular_agua_diaria(peso)
print("Litros de agua recomendados al día:", str(agua_diaria)+" Lts")
ritmo_cardiaco_maximo = calcular_ritmo_cardiaco_maximo(edad)
print("Tu ritmo cardíaco Maximo debe ser de:", str(ritmo_cardiaco_maximo) + " ppm")
print("Valores solo de referencia \nConsulte a su medico")

##Calorias_hombre =  88.362 + (13.397 * peso) + (4.799 * altura) - (5.677 * edad)
##Calorias_mujer = 447.593 + (9.247 * peso) + (3.098 * altura) - (4.330 * edad)

##print("Diferiencia de imc" ,Calorias_hombre,Calorias_mujer)
