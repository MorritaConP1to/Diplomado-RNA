def contar_caracteres(textito):
    """Esta funcion cuenta el numero del textito ingresada por el usuario"""
    
    if textito == "":
        textito = "Aprender Python es divertido"
    
    print("\nEl Texto:", textito, "tiene", len(textito), "caracteres")


def convertir_numero(numerito):
    """Convierte un numero a entero, string y flotante"""

    print("\nEl numero ingresado es:", numerito)

    numero_float = float(numerito)

    print("\nComo entero es:", int(numero_float))
    print("\nComo string es:", str(numerito))
    print("\nComo flotante es:", numero_float)


contar_caracteres(input("Escribe una frase para contar: (Opcional): "))
convertir_numero(input("Escribe un numero para convertirlo: "))