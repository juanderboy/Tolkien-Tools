# Activación conjunta y equilibrio ácido–base

Esta rutina ajusta constantes cinéticas medidas a distintas temperaturas y pHs
para dos formas reactivas, HA y A, que intercambian rápidamente un protón.
Se ejecuta dentro del módulo cinético; no requiere espectros ni los vuelve a ajustar.
Requiere Python 3.10+, NumPy, SciPy y Matplotlib para las figuras.

## Uso y archivo de entrada

```bash
tolkien-tools kinetics activation mediciones.csv --output resultados
tolkien-tools kinetics activation mediciones.tsv --fixed-pka 7.5 --seed 12 --starts 50
tolkien-tools kinetics activation --interactive
tolkien-tools kinetics activation --help
```

Desde `tolkien-tools`, elegir 4 (cinéticas), después 4 («Mecanismos y estudios especiales») y
6 («Análisis térmico de especiación ácido–base de disulfuros»). Se puede indicar un archivo o introducir mediciones
manualmente. Las preguntas permiten elegir pKa fijo o libre, temperatura de
referencia y carpeta de salida. Para configurar semillas o cantidad de
inicializaciones, usar las opciones de la línea de comandos.

CSV separado por comas o TSV separado por tabulaciones, con encabezado obligatorio:

```csv
id,T_K,pH,k
m1,293.15,6.5,0.0012
m2,303.15,6.5,0.0024
m3,293.15,7.5,0.0051
```

Estas tres filas ilustran el formato; no constituyen un conjunto suficiente
para ajustar el modelo. Se requieren más mediciones que parámetros libres
(más de seis con pKa libre; más de cinco con pKa fijo), al menos dos temperaturas
y dos pHs. Esto es un mínimo numérico, no una garantía de identificabilidad.
Conviene medir varias temperaturas para cada pH y cubrir ambos lados del pKa.

| Columna | Significado |
|---|---|
| `T_K` | Temperatura absoluta positiva en kelvin. Convertir °C sumando 273.15. |
| `pH` | pH de esa medición, adimensional. Puede incluir pH medido a cada T. |
| `k` | Constante positiva de primer orden o pseudo-primer orden, en s⁻¹. |
| `id` | Identificador opcional; las réplicas permanecen como filas independientes. |
| `sigma_log_k` | Incertidumbre estándar opcional de ln(k), positiva y adimensional. |

Usar punto decimal y notación científica usual (`1.2e-3`); no usar coma decimal,
separadores de miles, unidades dentro de las celdas ni celdas vacías obligatorias.
No se eliminan mediciones automáticamente. Preparar explícitamente la tabla
que se desea analizar. `sigma_log_k` debe estar presente en todas las filas o
ausente en todas. Para incertidumbres relativas pequeñas, sigma_ln(k)≈sigma_k/k;
la rutina no hace esa conversión automáticamente.

Entrada manual: una línea por medición, `T_K pH k`, opcionalmente seguida de
`sigma_log_k`; Enter vacío termina. Se guarda la tabla introducida en la salida.

## Modelo y supuestos

HA ⇌ A + H⁺ es un equilibrio rápido frente al paso cinético. Ambas formas
reaccionan por vías paralelas; la constante observada es:

```text
alpha_A = 1/[1+10^(pKa(T)-pH)]
alpha_HA = 1-alpha_A
k_obs = alpha_HA*k_HA + alpha_A*k_A
```

Las constantes ingresadas deben describir esa etapa. Si proceden de un modelo
que ya separó asociación y reacción, no se añade aquí otra contribución de binding.
No interpretar cualquier velocidad global como compatible automáticamente.

Con Tref=298.15 K por defecto, R=0.00831446261815324 kJ mol⁻¹ K⁻¹:

```text
k_i(T) = k_i_ref*(T/Tref)*exp[-H_i/R*(1/T-1/Tref)]
pKa(T) = pKa_ref + H_acidity/[R*ln(10)]*(1/T-1/Tref)
```

H_acidity corresponde a HA→A+H⁺: un valor positivo disminuye pKa al calentar.
Se suponen entalpías constantes, sin términos de capacidad calorífica, y ausencia
de otras vías. Se usa el pH de cada fila: cuando se ingresa el mismo pH nominal
para todas las temperaturas, se asume constante. Una variación no medida del pH
del buffer puede confundirse con la entalpía de desprotonación.

## Parámetros y método numérico

Se optimizan ln(k_HA_ref), ln(k_A_ref), H_HA, H_A, H_acidity y pKa_ref.
`--fixed-pka` fija el último parámetro a Tref. `--reference` cambia Tref en K.
El objetivo es suma[(ln(k_pred)-ln(k))/sigma_log_k]². Sin incertidumbres, cada
fila tiene peso uno: se minimiza el error logarítmico, no el error absoluto de k.

SciPy `least_squares`, Jacobiano analítico, xtol=ftol=gtol=1e-12 y
max_nfev=3000 por inicialización. Por defecto se realizan 50 inicializaciones
con semilla 0 (`--starts`, `--seed`). La primera parte de las tasas centrales y
entalpías genéricas; las restantes exploran tasas y entalpías aleatorias.
Se elige el menor objetivo entre soluciones convergidas. Se registran todos
los intentos, incluidos los fallidos. Múltiples inicializaciones no prueban
matemáticamente que el mínimo sea global.

Límites numéricos: ln(kref) entre −30 y 30, entalpías de activación entre −200
 y 400 kJ/mol, entalpía de acidez entre −300 y 300 kJ/mol, pKa entre −5 y 20.
Son límites amplios de búsqueda, no información experimental. Se informan
proximidad a límites, rango y valores singulares del Jacobiano ponderado.
Los valores singulares dependen de las unidades y escalas de los parámetros.

La evaluación usa `logaddexp` y la función logística para evitar desbordamientos
en el cálculo de poblaciones y suma de contribuciones. Las tasas se parametrizan
en logaritmos para conservar positividad.

## Cantidades derivadas

Con prefactor de Eyring k_B T/h y coeficiente de transmisión igual a uno:

```text
S_i = R_J*[ln(k_i_ref)-ln(k_B*Tref/h)] + H_i_J/Tref
w_A = alpha_A*k_A/k_obs
H_app_local = (1-w_A)*H_HA + w_A*H_A + (w_A-alpha_A)*H_acidity
```

Las entropías se expresan en J mol⁻¹ K⁻¹; las entalpías en kJ/mol.
Las entropías intrínsecas corresponden a constantes en s⁻¹ y al prefactor
convencional; no ingresar constantes bimoleculares sin reformular el estado estándar.
Alpha es población; w es contribución al flujo. H_app_local es una derivada
local a pH constante, no necesariamente la pendiente de una recta en una ventana
finita. Si el pH varía con T, la derivada a lo largo de esa serie requiere también
su contribución térmica; la columna exportada sigue siendo la derivada a pH fijo.

## Resultados y revisión

- `input.tsv`: mediciones efectivamente utilizadas, también para entrada manual.
- `result.json`: parámetros, tasas y entropías intrínsecas, configuración,
  versiones del entorno, hash SHA-256 del archivo fuente, covarianza y errores estándar.
- `multistart.json`: inicializaciones, objetivos, parámetros y convergencia.
- `predictions.tsv`: predicciones, residuos logarítmicos, pKa(T), poblaciones,
  contribución al flujo y entalpía aparente local para cada fila.
- `summary.txt`: resumen legible y notas de interpretación.
- `activation_fit.png/pdf/svg`: Eyring observado/modelo y residuos frente a T.
  `--no-plot` omite las figuras.

Sin `--output`, se crea `results_activation_<nombre>` junto a la tabla; en el
menú se propone `results_activation`. Una nueva ejecución en la misma carpeta
reemplaza esos archivos: elegir carpetas distintas para preservar comparaciones.

El RMS logarítmico no es incertidumbre de las entalpías. La estabilidad entre
semillas tampoco constituye un intervalo de confianza.
El resumen en pantalla y `summary.txt` presentan cada magnitud con unidades y
`+/-` un error estándar (SE), obtenido por covarianza local del Jacobiano.
Sin `sigma_log_k`, se supone varianza común en ln(k), estimada como SSE/(n-p).
Si se proporciona `sigma_log_k`, se interpreta como desviación estándar absoluta
conocida de cada ln(k): no se reescala la covarianza por el residuo.
Los SE de tasas y entropías se propagan con la covarianza completa.
Un pKa fijado se marca `fixed`, sin asignarle error experimental cero.
Con Jacobiano de rango insuficiente o parámetros en límites numéricos, los SE
se indican como no disponibles. Son aproximaciones locales condicionadas al
modelo; no intervalos del 95 %, ni incertidumbres experimentales totales.
Además se generan perfiles reoptimizados y una matriz de compensación descriptiva
(ver las opciones `--profile-points`, `--profile-extent` y `--no-profiles`).
Con pKa fijo la matriz tiene cinco parámetros. Los contornos no son regiones
de confianza; las grillas y su convergencia se exportan para revisión.
Esta versión no asigna intervalos de confianza a esos contornos. Se deben considerar
el procesamiento previo, correlaciones y elección del modelo.
Comparar pKa fijo/libre, inspeccionar residuos y revisar cobertura térmica y de
pH antes de interpretar parámetros microscópicos. Un mínimo estable puede
coexistir con correlaciones fuertes y amplias regiones de parámetros compatibles.
