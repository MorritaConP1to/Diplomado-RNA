def calcular_calorias(carbohidratos, proteinas, grasas):
    calorias = (carbohidratos * 4) + (proteinas * 4) + (grasas * 9)
    return calorias

calorias_totales = calcular_calorias(float(input("Ingrese la cantidad de carbohidratos en gramos: ")), float(input("Ingrese la cantidad de proteínas en gramos: ")), float(input("Ingrese la cantidad de grasas en gramos: ")))
print(f"La cantidad total de calorías es: {calorias_totales}")