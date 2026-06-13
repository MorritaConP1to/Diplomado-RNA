# ==========================================
# GENERADOR DE GAMERTAGS - VERSIÓN 2
# ==========================================
# Proyecto del Módulo 2 — Python para IA
# Tema 1: Fundamentos de Python
#
# ¿Qué hace este programa?
# Genera diferentes estilos de GamerTags a partir del nombre,
# apellido y número favorito del jugador.
#
# Dependencias (instalar con pip):
#   pip install pyfiglet tabulate colorama
#
# Conceptos de Python utilizados:
#   - Funciones y docstrings
#   - Slicing de strings (nombre[:4], nombre[::-1], nombre[-2:])
#   - f-strings para formatear salida
#   - Manejo de excepciones (try/except)
#   - Módulos externos (pyfiglet, tabulate, colorama)
# ==========================================

import pyfiglet
from tabulate import tabulate
from colorama import init, Fore, Style

# init(autoreset=True) inicializa colorama y hace que los colores
# se reseteen automáticamente después de cada print — sin esto,
# el color se quedaría activo para todo el texto siguiente
init(autoreset=True)


# ══════════════════════════════════════════════════════════════
#  FUNCIONES DE VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════

def cabecera():
    """
    Muestra la cabecera del programa con título artístico y color.

    pyfiglet.figlet_format() convierte texto normal a arte ASCII.
    El parámetro font= define el estilo visual del texto.
    Ver más fuentes en: https://patorjk.com/software/taag/

    Fore.MAGENTA aplica color morado al texto (usando colorama).
    Style.RESET_ALL restaura el color al default del terminal.
    """
    titulo = pyfiglet.figlet_format("GamerTag", font="slant")
    print(Fore.MAGENTA + titulo)
    print(Fore.CYAN + "¡Crea tu propio GamerTag!" + Style.RESET_ALL)
    print("-" * 50)


def mostrar_estadisticas(nombre, apellido, numero_favorito):
    """
    Muestra un resumen de las características del nombre del jugador.

    Parámetros:
        nombre (str): nombre del jugador
        apellido (str): apellido del jugador
        numero_favorito (int): número favorito del jugador

    Usa tabulate para presentar los datos en forma de tabla.
    tablefmt="grid" dibuja la tabla con bordes tipo ASCII.
    """
    # Cada elemento de la lista es una fila de la tabla: [etiqueta, valor]
    stats = [
        ["Nombre completo",        f"{nombre} {apellido}"],
        ["Longitud del nombre",    len(nombre)],          # len() cuenta caracteres
        ["Primera letra",          nombre[0]],             # índice 0 = primer carácter
        ["Última letra",           nombre[-1]],            # índice -1 = último carácter
        ["Primera letra apellido", apellido[0]],
        ["Número favorito",        numero_favorito]
    ]

    print(Fore.YELLOW + "\n📊 RESUMEN DE TUS ESTADÍSTICAS" + Style.RESET_ALL)
    # headers define los encabezados de las columnas
    print(tabulate(stats, headers=["Estadística", "Valor"], tablefmt="grid"))


def mostrar_todas_las_tags(nombre, apellido, numero_favorito):
    """
    Muestra todos los GamerTags generados en una tabla comparativa.

    Parámetros:
        nombre (str): nombre del jugador
        apellido (str): apellido del jugador
        numero_favorito (int): número favorito del jugador

    Llama a cada función generadora y agrupa los resultados.
    """
    tags = [
        ["Básico",       crear_tag_basico(nombre)],
        ["Invertido",    crear_tag_invertido(nombre)],
        ["Intercalado",  crear_tag_intercalado(nombre, apellido)],
        ["Élite",        crear_tag_elite(nombre)],
        ["Con Número",   crear_tag_numero(nombre, numero_favorito)]
    ]

    print(Fore.GREEN + "\n🎮 TUS POSIBLES GAMERTAGS" + Style.RESET_ALL)
    print(tabulate(tags, headers=["Tipo de Tag", "GamerTag"], tablefmt="grid"))


# ══════════════════════════════════════════════════════════════
#  FUNCIONES GENERADORAS DE TAGS
# ══════════════════════════════════════════════════════════════

def crear_tag_basico(nombre):
    """
    Crea un GamerTag tomando las primeras 4 letras del nombre.

    Parámetros:
        nombre (str): nombre del jugador

    Retorna:
        str: las primeras 4 letras del nombre

    Ejemplo:
        crear_tag_basico("Diana") → "Dian"

    Concepto: slicing con [inicio:fin]
        nombre[:4] es equivalente a nombre[0:4]
        Si el nombre tiene menos de 4 letras, toma todas las que haya.
    """
    return nombre[:4]


def crear_tag_invertido(nombre):
    """
    Crea un GamerTag invirtiendo el nombre completo.

    Parámetros:
        nombre (str): nombre del jugador

    Retorna:
        str: el nombre escrito al revés

    Ejemplo:
        crear_tag_invertido("Diana") → "aiD anaiD" → "anaiD"

    Concepto: slicing con paso negativo [::-1]
        [inicio:fin:paso] donde paso=-1 recorre el string de derecha a izquierda.
        Los dos puntos vacíos al inicio y fin significan "desde el principio hasta el final".
    """
    return nombre[::-1]


def crear_tag_intercalado(nombre, apellido):
    """
    Crea un GamerTag mezclando nombre y apellido.

    Parámetros:
        nombre (str): nombre del jugador
        apellido (str): apellido del jugador

    Retorna:
        str: primera letra del nombre + primera letra del apellido
             + resto del nombre + resto del apellido

    Ejemplo:
        crear_tag_intercalado("Diana", "Blanco") → "D" + "B" + "iana" + "lanco" → "DBianalanco"

    Concepto: concatenación de strings con +
        nombre[0]   → primer carácter
        apellido[0] → primer carácter del apellido
        nombre[1:]  → todo el nombre desde el índice 1 en adelante
        apellido[1:]→ todo el apellido desde el índice 1 en adelante
    """
    return nombre[0] + apellido[0] + nombre[1:] + apellido[1:]


def crear_tag_elite(nombre):
    """
    Crea un GamerTag con las primeras 2 y las últimas 2 letras del nombre.

    Parámetros:
        nombre (str): nombre del jugador

    Retorna:
        str: primeras 2 letras + últimas 2 letras del nombre

    Ejemplo:
        crear_tag_elite("Diana") → "Di" + "na" → "Dina"
        crear_tag_elite("Fabian") → "Fa" + "an" → "Faan"

    Concepto: slicing con índices negativos
        nombre[:2]  → primeras 2 letras (índices 0 y 1)
        nombre[-2:] → últimas 2 letras (los últimos 2 caracteres)
    """
    return nombre[:2] + nombre[-2:]


def crear_tag_numero(nombre, numero_favorito):
    """
    Crea un GamerTag combinando las primeras 5 letras del nombre con el número favorito.

    Parámetros:
        nombre (str): nombre del jugador
        numero_favorito (int): número favorito del jugador

    Retorna:
        str: primeras 5 letras del nombre + número como string

    Ejemplo:
        crear_tag_numero("Diana", 7) → "Dian" + "7" → "Diana7"

    Concepto: str() convierte el número entero a string para poder concatenarlo.
        Los strings solo se pueden concatenar con otros strings,
        no directamente con números.
    """
    return nombre[:5] + str(numero_favorito)


# ══════════════════════════════════════════════════════════════
#  FUNCIÓN PRINCIPAL — ORQUESTA TODAS LAS ANTERIORES
# ══════════════════════════════════════════════════════════════

def mostrar_todo(nombre, apellido, numero_favorito):
    """
    Muestra estadísticas y todos los GamerTags en secuencia.

    Parámetros:
        nombre (str): nombre del jugador
        apellido (str): apellido del jugador
        numero_favorito (int): número favorito del jugador

    Esta función agrupa las dos funciones de visualización para
    mostrar toda la información relevante en un solo lugar.
    """
    mostrar_estadisticas(nombre, apellido, numero_favorito)
    mostrar_todas_las_tags(nombre, apellido, numero_favorito)


# ══════════════════════════════════════════════════════════════
#  PROGRAMA PRINCIPAL
# ══════════════════════════════════════════════════════════════
# if __name__ == "__main__" asegura que este bloque solo se ejecute
# cuando corremos este archivo directamente (no cuando se importa
# como módulo desde otro archivo).

if __name__ == "__main__":

    # 1. Mostrar cabecera antes de pedir datos
    cabecera()

    # 2. Pedir datos al usuario con validación
    # .strip() elimina espacios en blanco al inicio y final del input
    nombre   = input("Ingresa tu nombre: ").strip()
    apellido = input("Ingresa tu apellido: ").strip()

    # try/except para validar que el número favorito sea un entero
    # Si el usuario escribe "abc", int("abc") lanza ValueError
    # El while True repite la pregunta hasta obtener un número válido
    while True:
        try:
            numero_favorito = int(input("Ingresa tu número favorito: "))
            break  # sale del while cuando la conversión fue exitosa
        except ValueError:
            print(Fore.RED + "❌ Error: Debes ingresar un número entero. Intenta de nuevo." + Style.RESET_ALL)

    # 3. Mostrar resultados
    mostrar_todo(nombre, apellido, numero_favorito)
