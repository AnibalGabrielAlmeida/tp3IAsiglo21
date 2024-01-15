# Modelo de Hopfield para Reconocimiento de Patrones

Este código en Python implementa el Modelo de Hopfield para reconocimiento de patrones. El modelo utiliza la regla de aprendizaje de Hebb para ajustar los pesos entre los píxeles de un patrón original y luego intenta recuperar el patrón original a partir de una imagen con ruido.

## Patrón Original y Aprendizaje de Pesos

Se define un patrón original de 10x10 píxeles y se inicializa una matriz de pesos con ceros. Luego, se utiliza el método de Hebb para ajustar los pesos entre los píxeles.

```python
import numpy as np

# Se define el patrón original de 10x10 píxeles.
prototipo = np.array([
    # ... (patrón original)
])

# Inicializar la matriz de pesos con ceros.
pesos = np.zeros((10 * 10, 10 * 10))

# Método de Hebb para ajustar los pesos.
for i in range(10 * 10):
    for j in range(10 * 10):
        if i != j:
            pesos[i][j] += prototipo[i // 10][i % 10] * prototipo[j // 10][j % 10]

```
## Modelo de Hopfield para Reconocimiento de Patrones

El segundo código implementa el Modelo de Hopfield para reconocimiento de patrones. El modelo utiliza la regla de aprendizaje de Hebb para ajustar los pesos entre los píxeles de un patrón original y luego intenta recuperar el patrón original a partir de una imagen con ruido.

### Código

```python
import numpy as np

# Se define el patrón original de 10x10 píxeles.
prototipo = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

# Inicializar la matriz de pesos con ceros.
pesos = np.zeros((10 * 10, 10 * 10))

# Método de Hebb para ajustar los pesos.
for i in range(10 * 10):
    for j in range(10 * 10):
        if i != j:
            pesos[i][j] += prototipo[i // 10][i % 10] * prototipo[j // 10][j % 10]

# Crear una imagen con ruido basada en el patrón original.
noise_image = prototipo.copy()
# Introducir ruido en algunos píxeles.
noise_image[2][2] = -1
noise_image[5][7] = -1

# Inicializar la imagen de entrada con la imagen ruidosa.
input = noise_image.copy()

# Función para aplicar el modelo de Hopfield.
def modelo_de_hopfield(input, pesos):
    iteracciones_maximas = 100
    for iteracion in range(iteracciones_maximas):
        print(f"Iteración {iteracion + 1}:")
        for i in range(10 * 10):
            s = 0
            for j in range(10 * 10):
                s += pesos[i][j] * input[j // 10][j % 10]
            input[i // 10][i % 10] = 1 if s > 0 else -1
            print(f"Pixel ({i // 10}, {i % 10}) actualizado a {input[i // 10][i % 10]}")
    return input

# Aplicar el modelo para recuperar el patrón original.
imagen_recuperada = modelo_de_hopfield(input, pesos)

# Contar las iteraciones necesarias hasta alcanzar el equilibrio.
iteraciones = 0
while not np.array_equal(input, imagen_recuperada):
    iteraciones += 1
    input = imagen_recuperada.copy()
    imagen_recuperada = modelo_de_hopfield(input, pesos)

print(f"Iteraciones necesarias para alcanzar el equilibrio: {iteraciones}")
print("Imagen recuperada:")
print(imagen_recuperada)
