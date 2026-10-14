"""Neurona artificial para decidir si una planta necesita riego."""

from dataclasses import dataclass

import numpy as np

X = np.array(
    [
        [80, 18],
        [70, 22],
        [65, 28],
        [55, 25],
        [50, 32],
        [40, 30],
        [35, 25],
        [30, 32],
        [20, 35],
        [10, 38],
    ],
    dtype=float,
)

y = np.array([[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]], dtype=float)
ESCALA = np.array([100, 50], dtype=float)
CONDICIONES_NUEVAS = np.array(
    [[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float
)


@dataclass
class ResultadoEntrenamiento:
    """Resultados y parámetros de una ejecución de entrenamiento."""

    tasa_aprendizaje: float
    epocas: int
    pesos: np.ndarray
    sesgo: float
    probabilidades: np.ndarray
    error_final: float

    @property
    def decisiones(self) -> np.ndarray:
        return (self.probabilidades >= 0.5).astype(int)

    @property
    def aciertos(self) -> int:
        return int(np.sum(self.decisiones == y))


def sigmoide(z: np.ndarray) -> np.ndarray:
    """Convierte la suma ponderada en una probabilidad entre 0 y 1."""
    return 1 / (1 + np.exp(-z))


def entrenar(
    tasa_aprendizaje: float,
    epocas: int,
    semilla: int = 42,
) -> ResultadoEntrenamiento:
    """Entrena la neurona con descenso de gradiente batch."""
    if tasa_aprendizaje <= 0:
        raise ValueError("La tasa de aprendizaje debe ser positiva.")
    if epocas < 0:
        raise ValueError("La cantidad de épocas no puede ser negativa.")

    X_normalizado = X / ESCALA
    rng = np.random.default_rng(semilla)
    pesos = rng.standard_normal((2, 1)) * 0.1
    sesgo = 0.0

    for epoca in range(epocas + 1):
        z = X_normalizado @ pesos + sesgo
        probabilidad = sigmoide(z)
        error = probabilidad - y
        gradiente_z = error * probabilidad * (1 - probabilidad) * (2 / len(y))
        gradiente_pesos = X_normalizado.T @ gradiente_z
        gradiente_sesgo = float(np.sum(gradiente_z))

        if epoca < epocas:
            pesos -= tasa_aprendizaje * gradiente_pesos
            sesgo -= tasa_aprendizaje * gradiente_sesgo

    probabilidades = sigmoide(X_normalizado @ pesos + sesgo)
    error_final = float(np.mean((probabilidades - y) ** 2))
    return ResultadoEntrenamiento(
        tasa_aprendizaje=tasa_aprendizaje,
        epocas=epocas,
        pesos=pesos,
        sesgo=sesgo,
        probabilidades=probabilidades,
        error_final=error_final,
    )


def predecir(
    resultado: ResultadoEntrenamiento,
    datos: np.ndarray,
    umbral: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula probabilidades y decisiones para datos nuevos."""
    if not 0 < umbral < 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    datos_normalizados = np.asarray(datos, dtype=float) / ESCALA
    probabilidades = sigmoide(datos_normalizados @ resultado.pesos + resultado.sesgo)
    decisiones = (probabilidades >= umbral).astype(int)
    return probabilidades, decisiones


def ejecutar_experimentos() -> list[ResultadoEntrenamiento]:
    """Ejecuta todas las configuraciones solicitadas en la guía."""
    configuraciones = [
        (0.5, 100),
        (0.5, 1000),
        (0.5, 10000),
        (0.5, 20000),
        (0.01, 10000),
        (0.1, 10000),
        (1.0, 10000),
        (2.0, 10000),
    ]
    return [entrenar(lr, epocas) for lr, epocas in configuraciones]


def main() -> None:
    """Muestra el entrenamiento base, las predicciones y los experimentos."""
    resultado = entrenar(tasa_aprendizaje=0.5, epocas=10000)
    X_normalizado = X / ESCALA

    print("=== DATOS NORMALIZADOS ===")
    print(X_normalizado)
    print("\n=== ENTRENAMIENTO BASE ===")
    print(f"Tasa de aprendizaje: {resultado.tasa_aprendizaje}")
    print(f"Épocas: {resultado.epocas}")
    print(f"Peso de la humedad:      {resultado.pesos[0, 0]:.4f}")
    print(f"Peso de la temperatura:  {resultado.pesos[1, 0]:.4f}")
    print(f"Sesgo:                   {resultado.sesgo:.4f}")
    print(f"Error final (ECM):       {resultado.error_final:.6f}")
    print(f"Aciertos:                {resultado.aciertos}/10")

    print("\nHumedad  Temp   Esperado  Probabilidad  Decisión")
    for (humedad, temperatura), esperado, probabilidad, decision in zip(
        X,
        y.ravel(),
        resultado.probabilidades.ravel(),
        resultado.decisiones.ravel(),
    ):
        print(
            f"{humedad:>5.0f}%  {temperatura:>4.0f}°C"
            f"    {int(esperado)}        {probabilidad:.4f}"
            f"         {int(decision)}"
        )

    print("\n=== INTERPRETACIÓN DE LOS PESOS ===")
    humedad_signo = "negativo" if resultado.pesos[0, 0] < 0 else "positivo"
    temperatura_signo = "negativo" if resultado.pesos[1, 0] < 0 else "positivo"
    print(
        f"- Humedad: peso {humedad_signo}; a mayor humedad, "
        "menor probabilidad de regar."
    )
    print(
        f"- Temperatura: peso {temperatura_signo}; a mayor temperatura, "
        "mayor probabilidad de regar."
    )

    probabilidades_nuevas, decisiones_nuevas = predecir(
        resultado, CONDICIONES_NUEVAS
    )
    print("\n=== CONDICIONES NUEVAS ===")
    print("Humedad  Temp   Probabilidad  Decisión")
    for (humedad, temperatura), probabilidad, decision in zip(
        CONDICIONES_NUEVAS,
        probabilidades_nuevas.ravel(),
        decisiones_nuevas.ravel(),
    ):
        texto = "Regar (1)" if decision == 1 else "No regar (0)"
        print(
            f"{humedad:>5.0f}%  {temperatura:>4.0f}°C"
            f"    {probabilidad:.4f}      {texto}"
        )

    print("\n=== EXPERIMENTOS ===")
    print("Épocas  Tasa    ECM final  Aciertos  P(45%, 34°C)")
    for experimento in ejecutar_experimentos():
        probabilidad, _ = predecir(
            experimento, np.array([[45, 34]], dtype=float)
        )
        print(
            f"{experimento.epocas:>6}  {experimento.tasa_aprendizaje:>4.2f}"
            f"    {experimento.error_final:.6f}"
            f"     {experimento.aciertos:>2}/10"
            f"       {probabilidad[0, 0]:.6f}"
        )

    print("\n=== COMPARACIÓN DE UMBRALES ===")
    for umbral in (0.4, 0.5, 0.6):
        _, decisiones = predecir(resultado, X, umbral)
        aciertos = int(np.sum(decisiones == y))
        print(f"Umbral {umbral:.1f}: {aciertos}/10 aciertos")


if __name__ == "__main__":
    main()
