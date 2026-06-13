# ==========================================
# GENERADOR DE GAMERTAGS - VERSIÓN 2
# ==========================================

import pyfiglet
from tabulate import tabulate
from colorama import init, Fore, Style

# Inicializar colorama
init(autoreset=True)

def cabecera():
    """
    Muestra la cabecera con título artístico y color.
    """
    titulo = pyfiglet.figlet_format("GamerTag", font="slant")
    print(Fore.MAGENTA + titulo)
    print(Fore.CYAN + "¡Crea tu propio GamerTag!" + Style.RESET_ALL)
    print("-" * 50)

def crear_tag_basico(nombre):
    return nombre[:4]

def crear_tag_invertido(nombre):
    return nombre[::-1]

def crear_tag_intercalado(nombre, apellido):
    return nombre[0] + apellido[0] + nombre[1:] + apellido[1:]

def crear_tag_elite(nombre):
    return nombre[:2] + nombre[-2:]

def crear_tag_numero(nombre, numero_favorito):
    return nombre[:5] + str(numero_favorito)

def mostrar_estadisticas(nombre, apellido, numero_favorito):
    stats = [
        ["Nombre completo", f"{nombre} {apellido}"],
        ["Longitud del nombre", len(nombre)],
        ["Primera letra", nombre[0]],
        ["Última letra", nombre[-1]],
        ["Primera letra del apellido", apellido[0]],
        ["Número favorito", numero_favorito]
    ]
    print(Fore.YELLOW + "\n📊 RESUMEN DE TUS ESTADÍSTICAS" + Style.RESET_ALL)
    print(tabulate(stats, headers=["Estadística", "Valor"], tablefmt="grid"))

def mostrar_todas_las_tags(nombre, apellido, numero_favorito):
    tags = [
        ["Básico", crear_tag_basico(nombre)],
        ["Invertido", crear_tag_invertido(nombre)],
        ["Intercalado", crear_tag_intercalado(nombre, apellido)],
        ["Élite", crear_tag_elite(nombre)],
        ["Con Número", crear_tag_numero(nombre, numero_favorito)]
    ]
    print(Fore.GREEN + "\n🎮 TUS POSIBLES GAMERTAGS" + Style.RESET_ALL)
    print(tabulate(tags, headers=["Tipo de Tag", "GamerTag"], tablefmt="grid"))

def mostrar_todo_junto(nombre, apellido, numero_favorito):
    """Muestra estadísticas y tags (sin repetir la cabecera)."""
    mostrar_estadisticas(nombre, apellido, numero_favorito)
    mostrar_todas_las_tags(nombre, apellido, numero_favorito)

# --------------------------------------------------
# PROGRAMA PRINCIPAL
# --------------------------------------------------
if __name__ == "__main__":
    # 1. Mostrar cabecera ANTES de pedir datos
    cabecera()

    # 2. Pedir datos con validación
    nombre = input("Ingresa tu nombre: ").strip()
    apellido = input("Ingresa tu apellido: ").strip()
    while True:
        try:
            numero_favorito = int(input("Ingresa tu número favorito: "))
            break
        except ValueError:
            print(Fore.RED + "❌ Error: Debes ingresar un número entero. Intenta de nuevo." + Style.RESET_ALL)

    # 3. Mostrar todo
    mostrar_todo_junto(nombre, apellido, numero_favorito)