Titulos_de_libros=["Don Quijote de la Mancha","Cien años de soledad","El amor en los tiempos del cólera","Historia de dos ciudades","El principito","El gran Gatsby"," El Señor de los Anillos","El hobbit,","Sueño en el pabellón rojo","Las aventuras de Alicia en el país de las maravillas"]
numero_de_libros=len(Titulos_de_libros)

for i in range(numero_de_libros):
    print("El libro", i+1, "es:", Titulos_de_libros[i],"Y tiene", len(Titulos_de_libros[i]), "caracteres")