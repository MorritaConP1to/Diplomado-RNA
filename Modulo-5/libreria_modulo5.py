# ═══════════════════════════════════════════════════════════════════════════════
# 📦 libreria_modulo5.py — Embedded Systems & TinyML Utility Library
# Diplomado Superior en Redes Neuronales Artificiales y Deep Learning
# Módulo 5 | Diana Blanco
#
# Importa con:  from libreria_modulo5 import *
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, time, math
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — DETECCIÓN DE PLATAFORMA
# ═══════════════════════════════════════════════════════════════════════════════

def detectar_plataforma():
    """Detecta si estamos en Google Colab o en local.
    
    Returns:
        dict: Información del entorno
    """
    en_colab = 'google.colab' in sys.modules
    print(f"🌐 Plataforma: {'Google Colab' if en_colab else 'Local'}")
    return {'en_colab': en_colab, 'plataforma': 'colab' if en_colab else 'local'}


def configurar_reproducibilidad(semilla=42):
    """Fija semilla aleatoria para resultados reproducibles."""
    np.random.seed(semilla)
    print(f"✅ Reproducibilidad configurada (semilla={semilla})")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — COMUNICACIÓN SERIAL (ARDUINO)
# ═══════════════════════════════════════════════════════════════════════════════

def detectar_puertos():
    """Escanea y lista los puertos serie disponibles.
    
    Requiere: pyserial (pip install pyserial)
    
    Returns:
        list: Lista de puertos disponibles, ej. ['COM3', 'COM5']
    
    Ejemplo:
        >>> puertos = detectar_puertos()
        >>> print(puertos)
        ['COM3', 'COM5']
    """
    try:
        import serial.tools.list_ports as list_ports
        puertos = [p.device for p in list_ports.comports()]
        if puertos:
            print(f"🔌 Puertos detectados: {', '.join(puertos)}")
        else:
            print("⚠️  No se detectaron puertos serie")
        return puertos
    except ImportError:
        print("⚠️  pyserial no instalado. Para instalarlo: pip install pyserial")
        return []
    except Exception as e:
        print(f"⚠️  Error al detectar puertos: {e}")
        return []


def conectar_arduino(puerto='COM3', baudrate=9600, timeout=2):
    """Abre conexión serial con Arduino.
    
    Args:
        puerto (str): Puerto COM (Windows) o /dev/tty... (Linux/Mac)
        baudrate (int): Velocidad de comunicación (9600 típico)
        timeout (int): Timeout en segundos
    
    Returns:
        serial.Serial o None: Objeto de conexión o None si falla
    
    Ejemplo:
        >>> ser = conectar_arduino('COM3', 9600)
        >>> if ser:
        ...     enviar(ser, 'Hola Arduino')
    """
    try:
        import serial
        ser = serial.Serial(puerto, baudrate, timeout=timeout)
        time.sleep(2)
        print(f"✅ Conectado a {puerto} @ {baudrate} baud")
        return ser
    except ImportError:
        print("⚠️  pyserial no instalado. pip install pyserial")
        return None
    except serial.SerialException as e:
        print(f"❌ No se pudo conectar a {puerto}: {e}")
        print("   Verifica: puerto correcto, Arduino conectado, sin otro programa usándolo")
        return None


def enviar(ser, mensaje):
    """Envía un mensaje por serial a Arduino.
    
    Args:
        ser (serial.Serial): Conexión activa
        mensaje (str): Texto a enviar (se añade \\n automáticamente)
    
    Ejemplo:
        >>> enviar(ser, 'LED_ON')
        ✅ Enviado: LED_ON
    """
    try:
        if isinstance(mensaje, str):
            ser.write((mensaje + '\n').encode('utf-8'))
        else:
            ser.write(str(mensaje).encode('utf-8'))
        print(f"✅ Enviado: {mensaje}")
    except Exception as e:
        print(f"❌ Error al enviar: {e}")


def leer_linea(ser):
    """Lee una línea del serial.
    
    Args:
        ser (serial.Serial): Conexión activa
    
    Returns:
        str o None: Línea leída (decodificada) o None si timeout
    
    Ejemplo:
        >>> respuesta = leer_linea(ser)
        >>> print(respuesta)
    """
    try:
        linea = ser.readline().decode('utf-8').strip()
        return linea if linea else None
    except Exception as e:
        print(f"❌ Error al leer: {e}")
        return None


def leer_hasta(ser, termino='OK', timeout=5):
    """Lee líneas del serial hasta encontrar un término o timeout.
    
    Args:
        ser (serial.Serial): Conexión activa
        termino (str): Texto que detiene la lectura
        timeout (int): Tiempo máximo de espera en segundos
    
    Returns:
        list: Líneas leídas
    
    Ejemplo:
        >>> lineas = leer_hasta(ser, 'Listo')
    """
    lineas = []
    inicio = time.time()
    while time.time() - inicio < timeout:
        linea = leer_linea(ser)
        if linea:
            lineas.append(linea)
            print(f"   ← {linea}")
            if termino in linea:
                break
    return lineas


def cerrar(ser):
    """Cierra la conexión serial.
    
    Args:
        ser (serial.Serial): Conexión activa
    """
    try:
        ser.close()
        print("🔌 Conexión cerrada")
    except:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — COMPUERTAS LÓGICAS (IA SIMBÓLICA)
# ═══════════════════════════════════════════════════════════════════════════════

def compuerta_not(a):
    """NOT lógico: invierte la entrada.
    
    Args:
        a (int): 0 o 1
    
    Returns:
        int: 1 si a=0, 0 si a=1
    
    Ejemplo:
        >>> compuerta_not(0)
        1
        >>> compuerta_not(1)
        0
    """
    return int(not a)


def compuerta_and(a, b):
    """AND lógico: 1 solo si ambas entradas son 1.
    
    Args:
        a, b (int): 0 o 1
    
    Returns:
        int: 1 si a=1 y b=1, 0 otherwise
    
    Ejemplo:
        >>> compuerta_and(1, 1)
        1
        >>> compuerta_and(1, 0)
        0
    """
    return int(a and b)


def compuerta_or(a, b):
    """OR lógico: 1 si al menos una entrada es 1.
    
    Args:
        a, b (int): 0 o 1
    
    Returns:
        int
    """
    return int(a or b)


def compuerta_nand(a, b):
    """NAND: AND negado — universal (con ella construyes cualquier compuerta)."""
    return compuerta_not(compuerta_and(a, b))


def compuerta_nor(a, b):
    """NOR: OR negado."""
    return compuerta_not(compuerta_or(a, b))


def compuerta_xor(a, b):
    """XOR: 1 si las entradas son diferentes.
    
    Implementación: (A AND NOT B) OR (NOT A AND B)
    """
    return int(a != b)


def compuerta_xnor(a, b):
    """XNOR: 1 si las entradas son iguales (XOR negado)."""
    return compuerta_not(compuerta_xor(a, b))


def tabla_verdad(compuerta, nombre="Compuerta", n_entradas=2):
    """Genera e imprime la tabla de verdad de una compuerta.
    
    Args:
        compuerta (callable): Función de la compuerta
        nombre (str): Nombre para mostrar
        n_entradas (int): 1 (NOT) o 2 (AND, OR, etc.)
    
    Ejemplo:
        >>> tabla_verdad(compuerta_and, "AND")
    """
    print(f"\n{'='*30}")
    print(f"  Tabla de verdad: {nombre}")
    print(f"{'='*30}")
    
    if n_entradas == 1:
        print("  A | Salida")
        print("  ---+-------")
        for a in [0, 1]:
            print(f"  {a} |     {compuerta(a)}")
    else:
        print("  A   B | Salida")
        print("  ------+-------")
        for a in [0, 1]:
            for b in [0, 1]:
                print(f"  {a}   {b} |    {compuerta(a, b)}")


def construir_xor_desde_and_or_not():
    """Demuestra que XOR se construye combinando AND, OR y NOT.
    
    Fórmula: XOR = (A AND NOT B) OR (NOT A AND B)
    
    Returns:
        callable: Función XOR implementada con AND, OR, NOT
    """
    def xor_con_and_or_not(a, b):
        return compuerta_or(
            compuerta_and(a, compuerta_not(b)),
            compuerta_and(compuerta_not(a), b)
        )
    return xor_con_and_or_not


def circuito_sumador_medio(a, b):
    """Sumador medio: suma 2 bits, produce suma y acarreo.
    
    Args:
        a, b (int): Bits a sumar (0 o 1)
    
    Returns:
        dict: {'suma': bit_suma, 'acarreo': bit_acarreo}
    
    Ejemplo:
        >>> circuito_sumador_medio(1, 1)
        {'suma': 0, 'acarreo': 1}
    """
    suma = compuerta_xor(a, b)
    acarreo = compuerta_and(a, b)
    return {'suma': suma, 'acarreo': acarreo}


def circuito_sumador_completo(a, b, acarreo_entrada=0):
    """Sumador completo: suma 3 bits (a + b + acarreo previo).
    
    Returns:
        dict: {'suma': bit_suma, 'acarreo': bit_acarreo}
    """
    s1 = compuerta_xor(a, b)
    suma = compuerta_xor(s1, acarreo_entrada)
    acarreo = compuerta_or(
        compuerta_and(a, b),
        compuerta_and(s1, acarreo_entrada)
    )
    return {'suma': suma, 'acarreo': acarreo}


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — PERCEPTRÓN (IA CONEXIONISTA)
# ═══════════════════════════════════════════════════════════════════════════════

def perceptron_forward(W, b, X):
    """Forward pass del perceptrón binario.
    
    Args:
        W (list or array): Pesos [w1, w2, ..., wn]
        b (float): Sesgo (bias)
        X (list or array): Entradas [x1, x2, ..., xn]
    
    Returns:
        int: 1 si suma ponderada >= 0, 0 otherwise
    
    Ejemplo:
        >>> perceptron_forward([0.5, -0.3], 0.1, [1, 0])
        1
        >>> perceptron_forward([0.5, -0.3], 0.1, [0, 1])
        0
    """
    suma = np.dot(W, X) + b
    return 1 if suma >= 0 else 0


def perceptron_binario(W, b, entradas):
    """Aplica el perceptrón a múltiples entradas.
    
    Args:
        W (list): Pesos
        b (float): Sesgo
        entradas (list of list): Lista de vectores de entrada
    
    Returns:
        list: Predicciones (0 o 1) para cada entrada
    
    Ejemplo:
        >>> datos = [[0,0], [0,1], [1,0], [1,1]]
        >>> perceptron_binario([1, 1], -1.5, datos)
        [0, 0, 0, 1]  # AND
    """
    return [perceptron_forward(W, b, x) for x in entradas]


def perceptron_entrenar(X, y, lr=0.1, epochs=10, verbose=True):
    """Entrena un perceptrón binario con la regla de Rosenblatt.
    
    Args:
        X (array): Matriz de entrenamiento (n_muestras, n_features)
        y (array): Etiquetas (0 o 1)
        lr (float): Tasa de aprendizaje
        epochs (int): Iteraciones de entrenamiento
        verbose (bool): Si mostrar progreso
    
    Returns:
        dict: pesos (W), sesgo (b), history (errores por epoch)
    
    Ejemplo:
        >>> X = np.array([[0,0],[0,1],[1,0],[1,1]])
        >>> y = np.array([0,0,0,1])  # AND
        >>> res = perceptron_entrenar(X, y, epochs=20)
        >>> print(f"Pesos: {res['W']}, Bias: {res['b']}")
    """
    n_features = X.shape[1]
    W = np.random.uniform(-1, 1, n_features)
    b = np.random.uniform(-1, 1)
    history = []
    
    for epoch in range(epochs):
        errores = 0
        for i in range(len(X)):
            y_pred = perceptron_forward(W, b, X[i])
            error = y[i] - y_pred
            if error != 0:
                W += lr * error * X[i]
                b += lr * error
                errores += 1
        
        history.append(errores)
        if verbose and (epoch % 5 == 0 or epoch == epochs-1):
            print(f"   Época {epoch:3d}: errores={errores}")
        
        if errores == 0:
            if verbose:
                print(f"✅ Convergió en época {epoch}")
            break
    
    return {'W': W, 'b': b, 'history': history}


def perceptron_compuerta(tipo='AND', lr=0.1, epochs=20, verbose=False):
    """Entrena un perceptrón para imitar una compuerta lógica.
    
    Args:
        tipo (str): 'AND', 'OR', 'NAND', 'NOR', 'NOT'
        lr (float): Learning rate
        epochs (int): Épocas máximas
        verbose (bool): Mostrar progreso
    
    Returns:
        dict: Resultado del entrenamiento
    
    Ejemplo:
        >>> res = perceptron_compuerta('AND')
        >>> print(f"Pesos: {res['W']}")
    """
    compuertas = {
        'AND':  ([[0,0],[0,1],[1,0],[1,1]], [0,0,0,1]),
        'OR':   ([[0,0],[0,1],[1,0],[1,1]], [0,1,1,1]),
        'NAND': ([[0,0],[0,1],[1,0],[1,1]], [1,1,1,0]),
        'NOR':  ([[0,0],[0,1],[1,0],[1,1]], [1,0,0,0]),
        'NOT':  ([[0],[1]], [1,0])
    }
    
    if tipo not in compuertas:
        raise ValueError(f"Compuertas disponibles: {list(compuertas.keys())}")
    
    X, y = compuertas[tipo]
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n🧠 Entrenando perceptrón para {tipo}")
    resultado = perceptron_entrenar(X, y, lr, epochs, verbose)
    
    preds = perceptron_binario(resultado['W'], resultado['b'], X)
    acc = np.mean(preds == y) * 100
    print(f"   Precisión: {acc:.0f}%")
    print(f"   Pesos: {np.round(resultado['W'], 3)}, Bias: {round(resultado['b'], 3)}")
    
    resultado['accuracy'] = acc
    return resultado


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — CONVERSIONES Y UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def binario_a_decimal(bits):
    """Convierte lista de bits a entero decimal.
    
    Args:
        bits (list or str): Bits, ej. [1,0,1] o "101"
    
    Returns:
        int: Valor decimal
    
    Ejemplo:
        >>> binario_a_decimal([1,0,1])
        5
        >>> binario_a_decimal("101")
        5
    """
    if isinstance(bits, str):
        return int(bits, 2)
    return int(''.join(str(b) for b in bits), 2)


def decimal_a_binario(n, bits=8):
    """Convierte entero a lista de bits.
    
    Args:
        n (int): Número decimal
        bits (int): Número de bits de salida
    
    Returns:
        list: Bits [MSB ... LSB]
    
    Ejemplo:
        >>> decimal_a_binario(5, 4)
        [0, 1, 0, 1]
    """
    return [int(b) for b in format(n, f'0{bits}b')]


def mapear_0_1(valor, min_orig, max_orig):
    """Mapea un valor de un rango a [0, 1].
    Útil para normalizar lecturas de sensores antes del perceptrón.
    
    Args:
        valor (float): Valor a mapear
        min_orig (float): Mínimo del rango original
        max_orig (float): Máximo del rango original
    
    Returns:
        float: Valor en rango [0, 1]
    
    Ejemplo:
        >>> mapear_0_1(512, 0, 1023)
        0.5
    """
    return (valor - min_orig) / (max_orig - min_orig)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — TinyML: Exportar Perceptrón a C++
# ═══════════════════════════════════════════════════════════════════════════════

def exportar_perceptron_a_c(W, b, nombre_funcion='predecir', 
                             nombres_entrada=['x1', 'x2'], tipo='int'):
    """Genera código C++ para ejecutar el perceptrón en Arduino.
    
    Args:
        W (array): Pesos entrenados
        b (float): Sesgo
        nombre_funcion (str): Nombre de la función en C++
        nombres_entrada (list): Nombres de los parámetros
        tipo (str): 'int' o 'float' para los cálculos
    
    Returns:
        str: Código C++ listo para copiar a Arduino IDE
    
    Ejemplo:
        >>> res = perceptron_compuerta('AND', verbose=False)
        >>> codigo = exportar_perceptron_a_c(res['W'], res['b'])
        >>> print(codigo)
    """
    pesos_str = ' + '.join(
        f'{w}*{n}' for w, n in zip(np.round(W, 3), nombres_entrada)
    )
    
    if tipo == 'int':
        b_int = int(round(b * 100))
        codigo = f"""// ══════════════════════════════════════════════════════════════
// Perceptrón entrenado: {nombre_funcion}
// Pesos: {np.round(W, 3)}, Bias: {b}
// ══════════════════════════════════════════════════════════════

int {nombre_funcion}(int {', '.join(nombres_entrada)}) {{
    // Suma ponderada: w1*x1 + w2*x2 + ... + b
    // Usamos multiplicación escalada (×100) para evitar floats
    int suma = {b_int};
    {pesos_str};
    
    // Función escalón: si suma >= 0 → 1, si no → 0
    return (suma >= 0) ? 1 : 0;
}}
"""
    else:
        codigo = f"""// ══════════════════════════════════════════════════════════════
// Perceptrón entrenado: {nombre_funcion} (float)
// Pesos: {np.round(W, 3)}, Bias: {b}
// ══════════════════════════════════════════════════════════════

float {nombre_funcion}(float {', '.join(nombres_entrada)}) {{
    float suma = {round(b, 3)};
    {pesos_str};
    
    // Función escalón
    return (suma >= 0.0) ? 1.0 : 0.0;
}}
"""
    return codigo


def exportar_modelo_arduino_ino(W, b, nombre='modelo', tipo='int'):
    """Genera un sketch .ino completo de Arduino con el perceptrón.
    
    Args:
        W (array): Pesos
        b (float): Sesgo
        nombre (str): Nombre del modelo/proyecto
    
    Returns:
        str: Código .ino completo
    
    Ejemplo:
        >>> res = perceptron_compuerta('AND')
        >>> ino = exportar_modelo_arduino_ino(res['W'], res['b'], 'AND_perceptron')
    """
    funcion = exportar_perceptron_a_c(W, b, 'predecir', ['sensor1', 'sensor2'], tipo)
    
    codigo = f"""// ══════════════════════════════════════════════════════════════
// TinyML — {nombre}
// Generado por libreria_modulo5.py
// ══════════════════════════════════════════════════════════════

{funcion}

void setup() {{
    Serial.begin(9600);
    Serial.println("🧠 TinyML listo — {nombre}");
    pinMode(LED_BUILTIN, OUTPUT);
}}

void loop() {{
    // Simular lectura de sensores
    // En un caso real, reemplaza con analogRead() o digitalRead()
    int entrada1 = random(0, 2);
    int entrada2 = random(0, 2);
    
    // Inferencia
    int resultado = predecir(entrada1, entrada2);
    
    // Mostrar resultado
    Serial.print("Entradas: ");
    Serial.print(entrada1);
    Serial.print(", ");
    Serial.print(entrada2);
    Serial.print(" → Predicción: ");
    Serial.println(resultado);
    
    // LED: prende si la predicción es 1
    digitalWrite(LED_BUILTIN, resultado ? HIGH : LOW);
    
    delay(2000);
}}
"""
    return codigo


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — VISUALIZACIÓN Y DIAGNÓSTICO
# ═══════════════════════════════════════════════════════════════════════════════

def graficar_convergencia(history, titulo="Convergencia del Perceptrón"):
    """Grafica los errores por época durante el entrenamiento.
    
    Args:
        history (list): Lista de errores por época (del entrenamiento)
        titulo (str): Título del gráfico
    
    Ejemplo:
        >>> res = perceptron_entrenar(X, y, epochs=50, verbose=False)
        >>> graficar_convergencia(res['history'])
    """
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.plot(history, marker='o', color='steelblue', linewidth=2)
        plt.title(titulo)
        plt.xlabel('Época')
        plt.ylabel('Errores')
        plt.grid(alpha=0.3)
        plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label='Convergencia')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("⚠️  matplotlib no disponible. No se puede graficar.")


def graficar_frontera_decision(W, b, X, y, titulo="Frontera de Decisión"):
    """Grafica la frontera de decisión del perceptrón en 2D.
    
    Args:
        W (array): Pesos [w1, w2]
        b (float): Sesgo
        X (array): Datos 2D
        y (array): Etiquetas
        titulo (str): Título
    
    Ejemplo:
        >>> X = np.array([[0,0],[0,1],[1,0],[1,1]])
        >>> y = np.array([0,0,0,1])
        >>> res = perceptron_entrenar(X, y, epochs=20, verbose=False)
        >>> graficar_frontera_decision(res['W'], res['b'], X, y)
    """
    try:
        import matplotlib.pyplot as plt
        
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 100),
            np.linspace(y_min, y_max, 100)
        )
        Z = np.array([perceptron_forward(W, b, [x, y]) for x, y in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=100)
        plt.title(titulo)
        plt.xlabel('x₁')
        plt.ylabel('x₂')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("⚠️  matplotlib no disponible")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — SIMULACIÓN DE ARDUINO (sin hardware)
# ═══════════════════════════════════════════════════════════════════════════════

class ArduinoSimulado:
    """Simula un Arduino en software para practicar sin hardware.
    
    Ejemplo:
        >>> arduino = ArduinoSimulado()
        >>> arduino.configurar()
        >>> arduino.escribir_serial('HOLA')
        >>> arduino.leer_serial()
    """
    
    def __init__(self, nombre="ArduinoSim"):
        self.nombre = nombre
        self.pines = {f'D{i}': 0 for i in range(14)}
        self.pines_analog = {f'A{i}': 0 for i in range(6)}
        self.serial_buffer = []
        self.led_builtin = 0
        self.baudrate = 9600
        self.modo_pines = {}
    
    def configurar(self, baudrate=9600):
        """Simula Serial.begin()."""
        self.baudrate = baudrate
        print(f"✅ {self.nombre} — Serial iniciado @ {baudrate} baud")
        print(f"   Pines digitales: D0-D13 | Analógicos: A0-A5")
        return self
    
    def pin_mode(self, pin, modo):
        """Simula pinMode(). modo: 'OUTPUT' o 'INPUT'."""
        self.modo_pines[pin] = modo
        print(f"   pinMode({pin}, {modo})")
    
    def digital_write(self, pin, valor):
        """Simula digitalWrite()."""
        if pin == 'LED_BUILTIN':
            self.led_builtin = valor
        elif pin in self.pines:
            self.pines[pin] = valor
        estado = 'HIGH' if valor else 'LOW'
        print(f"   digitalWrite({pin}, {estado})")
    
    def digital_read(self, pin):
        """Simula digitalRead() — lee el valor actual del pin."""
        return self.pines.get(pin, 0)
    
    def analog_write(self, pin, valor):
        """Simula analogWrite() (PWM)."""
        if pin in self.pines:
            self.pines[pin] = min(255, max(0, valor))
        print(f"   analogWrite({pin}, {self.pines[pin]})")
    
    def analog_read(self, pin):
        """Simula analogRead() — valor aleatorio 0-1023."""
        import random
        self.pines_analog[pin] = random.randint(0, 1023)
        return self.pines_analog[pin]
    
    def serial_println(self, mensaje):
        """Simula Serial.println()."""
        self.serial_buffer.append(str(mensaje))
        print(f"   Serial: {mensaje}")
    
    def serial_available(self):
        """Simula Serial.available()."""
        return len(self.serial_buffer)
    
    def serial_read(self):
        """Simula Serial.read() — lee y descarta del buffer."""
        if self.serial_buffer:
            return self.serial_buffer.pop(0)
        return -1
    
    def delay(self, ms):
        """Simula delay() — solo imprime."""
        print(f"   ⏱ delay({ms}ms)")
    
    def estado(self):
        """Muestra el estado actual de todos los pines."""
        print(f"\n📊 Estado de {self.nombre}:")
        print(f"   LED_BUILTIN: {'ON' if self.led_builtin else 'OFF'}")
        print(f"   Pines digitales: {self.pines}")
        print(f"   Pines analógicos: {self.pines_analog}")
        print(f"   Buffer serial: {len(self.serial_buffer)} mensajes")


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — RESUMEN DEL AMBIENTE
# ═══════════════════════════════════════════════════════════════════════════════

def resumen_ambiente(info=None):
    """Imprime resumen del entorno de trabajo."""
    print("=" * 55)
    print("  📋 RESUMEN DE AMBIENTE — Módulo 5: Embedded Systems")
    print("=" * 55)
    if info:
        print(f"  Plataforma   : {info['plataforma']}")
    print(f"  Python       : {sys.version.split()[0]}")
    
    try:
        import serial
        print(f"  pySerial     : {serial.__version__}")
    except ImportError:
        print(f"  pySerial     : ❌ No instalado (pip install pyserial)")
    
    print(f"  NumPy        : {np.__version__}")
    print("=" * 55)
    print("  ✅ Todo listo — a programar 🤖")
    print("=" * 55)
