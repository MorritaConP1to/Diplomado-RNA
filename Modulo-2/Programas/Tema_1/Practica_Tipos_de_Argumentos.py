def Organizar_Fiesta(invitados,tema="Python",lugar="Aula de Informatica"):
    print("Preparando una fiesta para", invitados, "invitados")
    print("El tema de la fiesta será", tema)
    print("La fiesta se celebrará en", lugar)

for i in range(3):
    invitados = input("\nIngrese el número de invitados: ")
    tema = input("Ingrese el tema de la fiesta (opcional): ")
    lugar = input("Ingrese el lugar de la fiesta (opcional): ")

    if tema =="":
            tema="Python"
    if lugar =="":
            lugar="Aula de Informatica" 

    Organizar_Fiesta(invitados, tema, lugar)