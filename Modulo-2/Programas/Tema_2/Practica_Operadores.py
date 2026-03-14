def comparar_longitud(palabra1, palabra2):
    lingitud1 = len(palabra1)
    longitud2 = len(palabra2)
    return lingitud1 == longitud2

palabra1 = 'gato'
palabra2 = 'perro'
print("¿Son 'gato' y 'perro' dos palabras con la misma longitud?")
print(comparar_longitud(palabra1, palabra2))
