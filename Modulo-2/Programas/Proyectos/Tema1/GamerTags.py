import tabulate
import pyfiglet
from colorama import Fore

def cabecera():
    
    """Función que muestra la cabecera del Generador de Tags de Jugadores."""
    # La función imprime un título artístico que representa el nombre del programa.
    # El título está diseñado con caracteres especiales para darle un aspecto llamativo y distintivo.
    # Se usa esta pagina "https://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type+Something+&x=none&v=4&h=4&w=80&we=false"
    # Se le coloca la r antes del string para que se muestre tal cual, sin interpretar caracteres especiales como saltos de línea o tabulaciones.
#    titulo = r"""  ______                                            ________                             
# /      \                                          |        \                            
#|  $$$$$$\  ______   ______ ____    ______    ______\$$$$$$$$______    ______    _______ 
#| $$ __\$$ |      \ |      \    \  /      \  /      \ | $$  |      \  /      \  /       \
#| $$|    \  \$$$$$$\| $$$$$$\$$$$\|  $$$$$$\|  $$$$$$\| $$   \$$$$$$\|  $$$$$$\|  $$$$$$$
#| $$ \$$$$ /      $$| $$ | $$ | $$| $$    $$| $$   \$$| $$  /      $$| $$  | $$ \$$    \ 
#| $$__| $$|  $$$$$$$| $$ | $$ | $$| $$$$$$$$| $$      | $$ |  $$$$$$$| $$__| $$ _\$$$$$$\
# \$$    $$ \$$    $$| $$ | $$ | $$ \$$     \| $$      | $$  \$$    $$ \$$    $$|       $$
#  \$$$$$$   \$$$$$$$ \$$  \$$  \$$  \$$$$$$$ \$$       \$$   \$$$$$$$ _\$$$$$$$ \$$$$$$$ 
#                                                                    |  \__| $$          
#                                                                      \$$    $$          
#                                                                       \$$$$$$           
#                                 !Crea tu propio GamerTag!                              
#                                                                       
#                                                                       
#                                                                       """
#    print(titulo)                      Version 2.0 con un nuevo diseño usando pyfiglet para generar el título artístico.
    titulo = pyfiglet.figlet_format("GamerTag", font="3-D")
    print(Fore.MAGENTA + titulo)
    print("¡Crea tu propio GamerTag!")
    return
    

def crear_tag_basico(nombre):
    """Función que crea un GamerTag básico a partir del nombre del jugador.
        Parametros:
        nombre (str): El nombre del jugador para el cual se generará el GamerTag.
        Retorna:
        str: Un GamerTag básico generado a partir del nombre del jugador.
    """
    # La función toma el nombre del jugador como entrada y genera un GamerTag básico.
    # El GamerTag se forma tomando las primeras cuatro letras del nombre.
    # Esto asegura que el GamerTag sea único y fácil de recordar.
    # usar el .upper() para que se muestre en mayúsculas, aunque no es necesario.
    tag = nombre[:4] # se deja vacio el espacio entre los dos puntos para que se tome todo el string, y el 4 para que se muestren las primeras 4 letras.
    return tag

def crear_tag_invertido(nombre):
    """Función que crea un GamerTag invertido a partir del nombre del jugador.
        Parametros:
        nombre (str): El nombre del jugador para el cual se generará el GamerTag.
        Retorna:
        str: Un GamerTag invertido generado a partir del nombre del jugador.
    """
    # La función toma el nombre del jugador como entrada y genera un GamerTag invertido.
    # El GamerTag se forma tomando todo el nombre y revirtiéndolo.
    # Esto asegura que el GamerTag sea único y tedioso aunque fácil de recordar.
    # usar el .upper() para que se muestre en mayúsculas, aunque no es necesario.
    tag = nombre[::-1]# de deja vacio el espacio entre los dos puntos para que se tome todo el string, y el -1 para que se muestre al revés.
    return tag

def crear_tag_intercalado(nombre,apellido):
    """Función que crea un GamerTag intercalado a partir del nombre y apellido del jugador.
        Parametros:
        nombre (str): El nombre del jugador para el cual se generará el GamerTag.
        apellido (str): El apellido del jugador para el cual se generará el GamerTag.
        Retorna:
        str: Un GamerTag intercalado generado a partir del nombre y apellido del jugador.
    """
    # La función toma el nombre y apellido del jugador como entrada y genera un GamerTag intercalado.
    tag = nombre[0] + apellido[0] + nombre[1:] + apellido[1:]  
    # se toma la primera letra del nombre y la primera letra del apellido, luego se toma el resto del nombre y el resto del apellido.
    return tag # se usa el sep="" para que no haya espacio entre las letras.

def crear_tag_elite(nombre):
    """Función que crea un GamerTag élite a partir del nombre y apellido del jugador.
        Parametros:
        nombre (str): El nombre del jugador para el cual se generará el GamerTag.
        Retorna:
        str: Un Gamertag elite usando los 2 primeros caracteres del nombre y los 2 ultimos caracteres del nombre.
        """
    # La función toma el nombre y apellido del jugador como entrada y genera un GamerTag élite.
    tag = nombre[:2] + nombre[-2:] # se toman los primeros dos caracteres del nombre y los últimos dos caracteres del nombre.
    return tag #se usa el sep="" para que no haya espacio entre las letras.

def crear_tag_numero(nombre, numero_favorito):
    """Función que crea un GamerTag al combinar las primeras 5 letras del nombre con el número favorito.
        Parametros:
        nombre (str): El nombre del jugador para el cual se generará el GamerTag.
        numero_favorito (int): El número favorito del jugador.

        Retorna:
        tag (str): Un GamerTag que combina las primeras 5 letras del nombre con el número favorito.
    """
    tag = nombre[:5] + str(numero_favorito)
    return tag

def mostrar_estadisticas(nombre,apellido ,numero_favorito):
    """Función que muestra las estadísticas de uso de cada tipo de GamerTag generado."""
    # La función muestra un resumen de cuántos GamerTags de cada tipo se han generado.
    # Esto puede ayudar al usuario que tipos de GamerTags son más populares o utilizados.
    # Muestra el nombre completo asi como la longitud del nombre del jugador.
    # muestra la primera letra del nombre y la primera letra del apellido.
    #print("\nEstadisticas de tu nombre:")  #v1 donde metemos todo por varios print, pero se ve desorganizado y poco atractivo visualmente.
    #print(f"Nombre completo: {nombre} {apellido}")
    #print(f"Longitud del tu nombre: {len(nombre)}")
    #print(f"Primera letra de tu nombre: {nombre[0]}")
    #print(f"Ultima letra de tu nombre: {nombre[-1]}")
    #print(f"Primera letra de tu apellido: {apellido[0]}")
    #print(f"Número favorito: {numero_favorito}")
    
    tags = [ #v2 donde metemos todo a una tabla usando la biblioteca tabulate para mostrar las cosas más organizada y visualmente atractiva.
        ["Nombre completo", (nombre) + " " + (apellido)],
        ["Longitud del nombre", len(nombre)],
        ["Primera letra del nombre", nombre[0]],
        ["Última letra del nombre", nombre[-1]],
        ["Primera letra del apellido", apellido[0]],
        ["Número favorito", numero_favorito]
    ]
    
    print("\nResumen de tus estadísticas:")
    print(tabulate.tabulate(tags, headers=["Estadística", "Valor"], tablefmt="grid"))
    


def mostrar_todas_las_tags(nombre, apellido, numero_favorito):
    """Función que muestra todas las tags generadas en una tabla."""
    # La función muestra todas las tags generadas en una tabla utilizando la biblioteca tabulate.
    # Esto proporciona una presentación clara y organizada de los diferentes tipos de GamerTags generados.
    tags = [
        ["Tag Básico", crear_tag_basico(nombre)],
        ["Tag Invertido", crear_tag_invertido(nombre)],
        ["Tag Intercalado", crear_tag_intercalado(nombre, apellido)],
        ["Tag Élité", crear_tag_elite(nombre)],
        ["Tag con Número", crear_tag_numero(nombre, numero_favorito)]
    ]
    print("\nTus posibles GamerTags son:")
    print(tabulate.tabulate(tags, headers=["Tipo de Tag", "GamerTag"], tablefmt="grid"))
    
def mostrar_todo_junto(nombre, apellido, numero_favorito): #v2.0 de poner oh mostrar todo
    """Función que muestra la cabecera, estadísticas y todas las tags generadas."""
    # La función muestra la cabecera del programa, las estadísticas del jugador y todas las tags generadas en una sola función.
    # Esto proporciona una experiencia de usuario más fluida y conveniente al mostrar toda la información relevante en un solo lugar.
    mostrar_estadisticas(nombre, apellido, numero_favorito)
    mostrar_todas_las_tags(nombre, apellido, numero_favorito)

# El programa comienza ejecutando la función cabecera para mostrar la cabecera del programa.
cabecera()
# Luego, se solicita al usuario que ingrese su nombre, apellido y número favorito
nombre = input("Ingresa tu nombre: ")
apellido = input("Ingresa tu apellido: ")
numero_favorito = int(input("Ingresa tu número favorito: "))
# Después de obtener la información del usuario, se llama a la función mostrar_estadisticas
# para mostrar las estadísticas relacionadas con el nombre del jugador.
#mostrar_estadisticas(nombre, apellido, numero_favorito)
mostrar_todo_junto(nombre, apellido, numero_favorito) #v2.0 donde mostramos todo junto en 2 tablas
# Finalmente, se llama a la función mostrar_todas_las_tags para mostrar todas las
# tags generadas en una tabla organizada.
#mostrar_todas_las_tags(nombre, apellido, numero_favorito)