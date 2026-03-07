from posixpath import sep
from tabulate import tabulate

def cabecera():
    """Función que muestra la cabecera del Generador de Tags de Jugadores."""
    # La función imprime un título artístico que representa el nombre del programa.
    # El título está diseñado con caracteres especiales para darle un aspecto llamativo y distintivo.
    # Se usa esta pagina "https://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type+Something+&x=none&v=4&h=4&w=80&we=false"
    # Se le coloca la r antes del string para que se muestre tal cual, sin interpretar caracteres especiales como saltos de línea o tabulaciones.
    titulo = r"""  ______                                            ________                             
 /      \                                          |        \                            
|  $$$$$$\  ______   ______ ____    ______    ______\$$$$$$$$______    ______    _______ 
| $$ __\$$ |      \ |      \    \  /      \  /      \ | $$  |      \  /      \  /       \
| $$|    \  \$$$$$$\| $$$$$$\$$$$\|  $$$$$$\|  $$$$$$\| $$   \$$$$$$\|  $$$$$$\|  $$$$$$$
| $$ \$$$$ /      $$| $$ | $$ | $$| $$    $$| $$   \$$| $$  /      $$| $$  | $$ \$$    \ 
| $$__| $$|  $$$$$$$| $$ | $$ | $$| $$$$$$$$| $$      | $$ |  $$$$$$$| $$__| $$ _\$$$$$$\
 \$$    $$ \$$    $$| $$ | $$ | $$ \$$     \| $$      | $$  \$$    $$ \$$    $$|       $$
  \$$$$$$   \$$$$$$$ \$$  \$$  \$$  \$$$$$$$ \$$       \$$   \$$$$$$$ _\$$$$$$$ \$$$$$$$ 
                                                                     |  \__| $$          
                                                                      \$$    $$          
                                                                       \$$$$$$           
                                 !Crea tu propio GamerTag!                              
                                                                       
                                                                       
                                                                       """
    print(titulo)

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

def mostrar_estadisticas(nombre,apellido,numero_favorito):
    """Función que muestra las estadísticas de uso de cada tipo de GamerTag generado."""
    # La función muestra un resumen de cuántos GamerTags de cada tipo se han generado.
    # Esto puede ayudar al usuario que tipos de GamerTags son más populares o utilizados.
    # Muestra el nombre completo asi como la longitud del nombre del jugador.
    # muestra la primera letra del nombre y la primera letra del apellido.
    print(f"Nombre completo: {nombre} {apellido}")
    print(f"Longitud del nombre: {len(nombre)}")
    print(f"Primera letra del nombre: {nombre[0]}")
    print(f"Primera letra del apellido: {apellido[0]}")
    print(f"Número favorito: {numero_favorito}")
    print(crear_tag_basico(nombre), sep="")
    print(crear_tag_invertido(nombre), sep="")
    print(crear_tag_intercalado(nombre,apellido), sep="")
    print(crear_tag_elite(nombre), sep="")
    print(crear_tag_numero(nombre, numero_favorito), sep="")


mostrar_estadisticas(input("Ingrese su nombre: "), input("Ingrese su apellido: "), int(input("Ingrese su número favorito: ")))

