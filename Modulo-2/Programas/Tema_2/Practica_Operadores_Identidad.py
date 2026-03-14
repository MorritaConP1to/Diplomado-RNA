import random


##palabra_adivinar = random.choice(["7u7r","UwU","pyhton","programador","Programacion","alumno"])
palabra_adivinar = "python"
numero_intentos = 0


def adibinar_palabra(letra_prueba):
        if letra_prueba in palabra_adivinar:
            print("La letra " + letra_prueba + " se encuentra en la palabra a adivinar")
            palabra_intento = input("Introduce la palabra que crees que es: ")
            if palabra_intento == palabra_adivinar:
                print("¡Has adivinado la palabra! 7u7r")
                
            else:
                
                print("No has adivinado la palabra u.u")
            
        else:
            print("La letra " + letra_prueba + " no se encuentra en la palabra a adivinar")


adibinar_palabra(input("!Bienvenido al juego de adivinanza \n Introduce una letra para compenzar:"))