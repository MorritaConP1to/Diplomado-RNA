# Backpropagation — Guía para copiar a mano

> Copia cada bloque en tu libreta en este orden. Al final, toma la foto y arma tu PDF.

---

**1. Notación**

L = última capa · l = capa cualquiera
w⁽ˡ⁾, b⁽ˡ⁾ = peso y sesgo de la capa l
a⁽ˡ⁾ = activación de la capa l · z⁽ˡ⁾ = suma ponderada de la capa l
y = valor real · σ(z) = sigmoid

---

**2. Forward pass (capa L)**

z⁽ᴸ⁾ = w⁽ᴸ⁾ · a⁽ᴸ⁻¹⁾ + b⁽ᴸ⁾
a⁽ᴸ⁾ = σ(z⁽ᴸ⁾)

---

**3. Función de costo (MSE, 1 ejemplo)**

C₀ = (a⁽ᴸ⁾ − y)²

---

**4. Regla de la cadena**

∂C₀/∂w⁽ᴸ⁾ = ∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ · ∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ · ∂C₀/∂a⁽ᴸ⁾

---

**5. Las 3 derivadas parciales por separado**

∂C₀/∂a⁽ᴸ⁾ = 2(a⁽ᴸ⁾ − y)
∂a⁽ᴸ⁾/∂z⁽ᴸ⁾ = σ'(z⁽ᴸ⁾) = σ(z)·(1−σ(z))
∂z⁽ᴸ⁾/∂w⁽ᴸ⁾ = a⁽ᴸ⁻¹⁾
∂z⁽ᴸ⁾/∂b⁽ᴸ⁾ = 1

---

**6. Fórmulas finales del gradiente**

∂C₀/∂w⁽ᴸ⁾ = a⁽ᴸ⁻¹⁾ · σ'(z⁽ᴸ⁾) · 2(a⁽ᴸ⁾ − y)
∂C₀/∂b⁽ᴸ⁾ = σ'(z⁽ᴸ⁾) · 2(a⁽ᴸ⁾ − y)

---

**7. Actualización de pesos (descenso por gradiente)**

w_nuevo = w_viejo − lr · ∂C/∂w

---

**8. Nota: vanishing gradient**

Si z es muy positivo o muy negativo → σ'(z) ≈ 0 → el gradiente desaparece.
Soluciones: ReLU, Batch Normalization, ResNet (skip connections).

---

✅ Listo — eso es todo. Tómale foto a tu libreta y ya tienes tu PDF de apuntes a mano.
