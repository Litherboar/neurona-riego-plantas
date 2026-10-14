# Neurona artificial para el riego de plantas

En este proyecto hice una neurona artificial desde cero usando solamente
NumPy. La idea es que la neurona revise la humedad del suelo y la temperatura
para estimar si una planta necesita riego.

La neurona recibe dos datos:

- `x1`: humedad del suelo, en porcentaje.
- `x2`: temperatura ambiental, en grados Celsius.

La salida es binaria:

- `1`: regar.
- `0`: no regar.

Los datos son solo para probar el funcionamiento del modelo. No se deben
tomar como una recomendación real para una planta específica.

## Requisitos

Para ejecutar el proyecto se necesita:

- Python 3.12 o superior.
- [uv](https://docs.astral.sh/uv/).

## Instalación y ejecución

Primero hay que ubicarse en la carpeta del proyecto y ejecutar:

```bash
uv sync
uv run main.py
```

También dejé configurado un comando para ejecutar el proyecto directamente:

```bash
uv run neurona-riego-plantas
```

No es necesario instalar NumPy por separado. El comando `uv sync` instala lo
que está definido en `pyproject.toml` y verifica las versiones que están en
`uv.lock`.

## Datos usados para entrenar

Usé los diez casos solicitados en la guía:

| Caso | Humedad (%) | Temperatura (°C) | Salida |
|---:|---:|---:|---:|
| 1 | 80 | 18 | 0 |
| 2 | 70 | 22 | 0 |
| 3 | 65 | 28 | 0 |
| 4 | 55 | 25 | 0 |
| 5 | 50 | 32 | 0 |
| 6 | 40 | 30 | 1 |
| 7 | 35 | 25 | 1 |
| 8 | 30 | 32 | 1 |
| 9 | 20 | 35 | 1 |
| 10 | 10 | 38 | 1 |

Antes de entrenar, dividí la humedad por `100` y la temperatura por `50`:

```python
ESCALA = np.array([100, 50], dtype=float)
X_normalizado = X / ESCALA
```

La misma variable `ESCALA` se usa después con los datos nuevos. Esto es
importante porque no tendría sentido entrenar con una escala y predecir con
otra diferente.

## Cómo funciona la neurona

La activación que usé es la sigmoide:

```text
sigmoide(z) = 1 / (1 + exp(-z))
```

En cada época el modelo hace lo siguiente:

1. Calcula la suma ponderada `z`.
2. Convierte ese resultado en una probabilidad usando la sigmoide.
3. Calcula el error cuadrático medio.
4. Calcula los gradientes de los pesos y del sesgo.
5. Actualiza los valores usando descenso de gradiente batch.

Después convierto la probabilidad en una decisión usando `0.5` como umbral:

```python
decision = (probabilidad >= 0.5).astype(int)
```

La neurona está implementada en
[`src/neurona_riego_plantas/__init__.py`](src/neurona_riego_plantas/__init__.py).
El archivo [`main.py`](main.py) se usa como punto de entrada para ejecutar la
actividad.

## Resultado del entrenamiento

La configuración base quedó así:

- Tasa de aprendizaje: `0.5`.
- Épocas: `10000`.
- Semilla: `42`.
- Pesos iniciales aleatorios, pero reproducibles.

Al ejecutar `uv run main.py` obtuve:

| Parámetro | Valor |
|---|---:|
| Peso de la humedad | -19.5201 |
| Peso de la temperatura | 2.1946 |
| Sesgo | 7.4780 |
| Error cuadrático medio final | 0.019539 |
| Aciertos en entrenamiento | 10/10 |

El peso de la humedad quedó negativo. Esto quiere decir que, manteniendo la
temperatura igual, una humedad más alta baja la probabilidad de regar.

El peso de la temperatura quedó positivo. En este caso, una temperatura más
alta aumenta la probabilidad de riego.

## Predicciones con datos nuevos

Probé la neurona con cinco condiciones diferentes. Estos datos también se
normalizaron usando la misma escala del entrenamiento:

| Humedad (%) | Temperatura (°C) | Probabilidad | Decisión |
|---:|---:|---:|---|
| 75 | 30 | 0.0029 | No regar (0) |
| 45 | 34 | 0.5464 | Regar (1) |
| 25 | 22 | 0.9724 | Regar (1) |
| 50 | 25 | 0.2342 | No regar (0) |
| 30 | 40 | 0.9670 | Regar (1) |

Por ejemplo, con `45 %` de humedad y `34 °C`, la probabilidad queda cerca de
`0.5`, por eso la decisión termina siendo regar, aunque no es un resultado tan
marcado como los otros casos.

## Pruebas con diferentes épocas y tasas

Para comparar los resultados mantuve la misma semilla, los mismos datos y la
misma inicialización. Solo cambié las épocas o la tasa de aprendizaje.

| Experimento | Épocas | Tasa | Error final | Aciertos | P(45 %, 34 °C) | Interpretación |
|---|---:|---:|---:|---:|---:|---|
| Pocas épocas | 100 | 0.5 | 0.169361 | 10/10 | 0.513819 | El modelo aprende, pero todavía tiene bastante error |
| Cantidad intermedia | 1000 | 0.5 | 0.066970 | 10/10 | 0.573881 | Hay una mejora clara, aunque todavía puede ajustar más |
| Prueba base | 10000 | 0.5 | 0.019539 | 10/10 | 0.546449 | El entrenamiento es estable |
| Más épocas | 20000 | 0.5 | 0.011669 | 10/10 | 0.524623 | El error baja, pero la mejora ya es más pequeña |
| Tasa pequeña | 10000 | 0.01 | 0.131320 | 10/10 | 0.531842 | El aprendizaje es lento |
| Tasa moderada | 10000 | 0.1 | 0.049284 | 10/10 | 0.580368 | Aprende de forma estable, pero más lento que con `0.5` |
| Tasa alta | 10000 | 1.0 | 0.011669 | 10/10 | 0.524633 | En estos datos converge rápido y de forma estable |
| Tasa muy alta | 10000 | 2.0 | 0.006556 | 10/10 | 0.507036 | Funciona aquí, pero podría ser inestable con otros datos |

### Lo que observé en las pruebas

- Con solo 100 épocas la neurona ya clasifica los diez ejemplos, pero el
  error sigue siendo alto. Por eso acertar todos los casos no es suficiente
  para decir que las probabilidades son buenas.
- Pasar de 1000 a 10000 épocas reduce bastante el error.
- Al pasar de 10000 a 20000 épocas el error sigue bajando, aunque la mejora
  es menor.
- Con una tasa de `0.01` el aprendizaje avanza más despacio y termina con un
  error mayor.
- Las tasas `0.1` y `0.5` dan un aprendizaje estable.
- En esta prueba las tasas `1.0` y `2.0` también convergen. Eso no quiere
  decir que una tasa alta funcione siempre, porque con otros datos puede
  producir oscilaciones o divergencia.
- La probabilidad para `45 %` de humedad y `34 °C` cambia entre experimentos
  porque los pesos no quedan ajustados de la misma forma en todos los casos.

## Comparación de umbrales

Después del entrenamiento base comparé diferentes umbrales sin cambiar los
pesos:

| Umbral | Aciertos en los 10 datos |
|---:|---:|
| 0.4 | 10/10 |
| 0.5 | 10/10 |
| 0.6 | 10/10 |

Cambiar el umbral solo cambia la forma de convertir la probabilidad en `0` o
`1`. No cambia los pesos ni el sesgo. En estos datos ninguna probabilidad de
entrenamiento está en una zona que cambie la clasificación entre esos
umbrales.

Con datos nuevos sí puede cambiar la decisión, sobre todo cuando la
probabilidad queda cerca de `0.5`.

## Respuestas a las preguntas

### ¿Por qué normalicé los datos?

La humedad puede llegar aproximadamente a `100` y la temperatura a `50`. Si
se usan directamente, las entradas tienen escalas diferentes y una de ellas
podría influir demasiado en el gradiente.

Al dividirlas por sus valores máximos esperados, las dos quedan en una escala
parecida, normalmente entre `0` y `1`. Esto ayuda a que el entrenamiento sea
más estable.

### ¿Dónde uso `X_normalizado` y por qué conservo `X`?

`X_normalizado` se usa en la suma ponderada y en el cálculo del gradiente de
los pesos.

Conservo `X` para mostrar los datos originales en porcentajes y grados
Celsius. Así los resultados son más fáciles de interpretar.

### ¿Por qué convierto la probabilidad en 0 o 1?

La sigmoide entrega un valor continuo entre `0` y `1`. Ese valor se puede
interpretar como una probabilidad, pero el sistema necesita una decisión
concreta.

Por eso uso un umbral. Si la probabilidad es mayor o igual a `0.5`, la salida
es `1` y se indica regar. Si es menor, la salida es `0` y se indica no regar.

### ¿Qué limitaciones tiene el proyecto?

Esta es una prueba didáctica con dos variables y diez ejemplos artificiales.
La neurona no tiene información sobre:

- El tipo de planta.
- La radiación solar.
- La lluvia.
- La humedad ambiental.
- El tipo de suelo.
- La estación.
- La hora del día.
- Los sensores reales.

Por esto, el resultado sirve para mostrar cómo funciona una neurona sencilla,
pero no se debe usar como una recomendación agronómica sin más datos y una
validación adecuada.

## Estructura del proyecto

```text
neurona-riego-plantas/
├── main.py
├── pyproject.toml
├── README.md
├── uv.lock
└── src/
    └── neurona_riego_plantas/
        └── __init__.py
```
