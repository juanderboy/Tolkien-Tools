---
title: "Modelos cineticos de TOLKinetics"
subtitle: "Rutina multilambda de tolkien-tools"
date: "2026-07-20"
geometry: margin=2.4cm
fontsize: 11pt
---

Documento de referencia para la rutina de cineticas multilambda de
`tolkien-tools`.

Fecha: 2026-07-20

# Objetivo general de la rutina

La rutina cinetica de Tolkien Tools ajusta series espectrofotometricas
multilambda. Cada archivo de entrada se interpreta como una matriz:

$$
A(\lambda,t)
$$

donde cada columna es un espectro a un tiempo experimental y cada fila es una
longitud de onda. El ajuste combina dos niveles:

1. Un modelo cinetico que predice perfiles de concentracion `C(t)`.
2. Un modelo espectral lineal que reconstruye la absorbancia:

$$
A_{\mathrm{calc}}(\lambda,t) =
\sum_i E_i(\lambda)\,C_i(t)
$$

$E_i(\lambda)$ contiene los espectros puros recuperados o fijados.

# Analisis inicial por SVD

Antes de elegir y ajustar un modelo cinetico, la rutina realiza un analisis por
SVD. El objetivo de este paso no es obtener constantes cineticas, sino estimar
cuantos componentes espectralmente significativos contiene la matriz
experimental. Esa informacion orienta la eleccion posterior del mecanismo:
un sistema que parece tener dos componentes distinguibles no deberia ajustarse
de entrada con tres o cuatro especies espectralmente independientes, salvo que
haya una razon quimica fuerte.

El uso de SVD y analisis de factores en datos espectroscopicos sigue la logica
clasica descripta por Malinowski en *Factor Analysis in Chemistry*. La idea es
reproducir la matriz experimental con la menor dimensionalidad compatible con el
error experimental. Primero se obtiene una solucion matematica abstracta; luego
se decide cuantos de esos factores abstractos son necesarios para describir la
informacion quimica dominante sin empezar a reproducir ruido.

## Idea matematica

La matriz experimental puede escribirse como:

$$
A = U S V^T
$$

donde:

- $A$ es la matriz de absorbancias. Si hay $n_\lambda$ longitudes de onda y
  $n_t$ espectros temporales, entonces $A$ tiene dimension
  $n_\lambda \times n_t$.
- $p=\min(n_\lambda,n_t)$ es el numero maximo de componentes matematicos que
  puede tener la descomposicion completa.
- $U$ tiene dimension $n_\lambda \times p$ y sus columnas son vectores
  ortogonales en el espacio espectral. Cada columna describe una direccion
  espectral abstracta, no necesariamente un espectro puro quimico.
- $S$ tiene dimension $p \times p$ y es diagonal. Sus elementos diagonales son
  los valores singulares, ordenados de mayor a menor.
- $V^T$, que en el codigo suele aparecer como `Vt`, tiene dimension
  $p \times n_t$. Sus filas son vectores ortogonales en el espacio temporal.

En este contexto, las columnas de $U$ y las filas de $V^T$ no deben
interpretarse directamente como espectros puros y perfiles de concentracion. Son
factores abstractos: una base matematica ortogonal que reproduce los datos. En
terminos de Malinowski, el primer paso del analisis de factores es obtener esa
reproduccion abstracta; la interpretacion quimica viene despues, cuando se
compara la dimensionalidad sugerida con modelos, restricciones y conocimiento
previo del sistema.

Lo que se grafica en la rutina como guia inicial es la diagonal de la matriz
central $S$:

$$
S=\mathrm{diag}(s_1,s_2,\ldots,s_p)
$$

es decir, la serie de valores singulares $s_i$.

Cada valor singular mide cuanta variacion de la matriz explica el componente
correspondiente. Si la muestra contiene pocas especies absorbentes reales, los
primeros valores singulares suelen ser grandes y luego aparece una caida hacia
valores chicos dominados por ruido.

La rutina grafica estos valores para que el usuario vea la jerarquia de factores.
Los primeros factores, asociados a los valores singulares grandes, son los
candidatos a factores primarios. Los factores de la cola, asociados a valores
singulares pequenos, suelen ser secundarios y normalmente describen ruido,
baseline residual u otras perturbaciones menores. Retener demasiados factores
puede hacer que el analisis reproduzca error experimental; retener demasiado
pocos deja informacion estructurada en los residuos.

## Rango efectivo y numero de especies

En ausencia de ruido, una mezcla lineal de $n$ especies absorbentes
independientes tendria rango $n$:

$$
A(\lambda,t)=\sum_{i=1}^{n}E_i(\lambda)C_i(t)
$$

En datos reales nunca hay un corte perfecto, porque hay ruido instrumental,
baseline residual, pequenas derivas y errores de preparacion. Por eso el SVD se
usa como criterio practico: se busca cuantos componentes son claramente mayores
que el fondo de ruido.

La decision practica es una compresion de factores: conservar el menor numero de
componentes que reproduce la matriz dentro del error experimental esperado. Ese
numero define el rango efectivo o dimensionalidad experimental del sistema. No
es una propiedad puramente algebraica del archivo, porque depende de la relacion
senal/ruido y de la calidad del preprocesado.

El numero de componentes sugerido por SVD no es automaticamente el numero de
pasos cineticos. Es el numero minimo de contribuciones espectrales
distinguibles que la matriz parece necesitar. Por ejemplo, dos especies
cineticamente distintas pero con el mismo espectro cuentan como una sola
contribucion espectral para SVD.

Por la misma razon, el rango por SVD tampoco prueba por si solo un mecanismo.
Solo indica cuantas dimensiones espectrales independientes hay en los datos. La
seleccion del modelo cinetico debe combinar ese resultado con el mecanismo
quimico propuesto, las condiciones experimentales y la estructura de los
residuos.

## Uso real de la diagonal de S en la rutina

En el flujo interactivo actual, el analisis SVD inicial se usa de forma visual y
cualitativa. Al cargar un experimento, la rutina muestra el panel espectral y el
grafico de la diagonal de $S$. Ese grafico es la informacion que el usuario usa
para estimar cuantas contribuciones espectrales dominantes hay antes de elegir el
modelo cinetico.

La lectura practica es contar cuantas especies coloreadas, o mas precisamente
cuantas contribuciones espectrales independientes, parecen estar presentes. Para
eso se mira la jerarquia de los valores singulares:

- los primeros valores grandes suelen corresponder a senales espectrales reales;
- una caida abrupta seguida por valores pequenos y parecidos suele marcar la
  entrada en la zona dominada por ruido;
- un valor intermedio puede indicar una especie coloreada debil, una deriva de
  baseline, scattering o algun problema de preprocesado.

En este sentido, `diag(S)` funciona como una guia para contar especies
espectralmente distinguibles. Si se observan dos valores singulares claramente por
encima de la cola de ruido, el experimento probablemente requiere dos espectros
visibles. Si aparecen tres, conviene considerar modelos con tres especies
absorbentes. Si una especie cinetica no cambia el espectro, o comparte espectro
con otra, no necesariamente aumenta el numero de componentes visibles en
`diag(S)`.

La decision no debe tomarse de manera mecanica. La diagonal de $S$ orienta, pero
la eleccion final del modelo debe combinar esa lectura con el mecanismo quimico,
las condiciones experimentales y el comportamiento del ajuste final.

Por ejemplo, si se ven dos especies coloreadas dominantes, es razonable empezar
probando un modelo de dos especies visibles, como `A -> B`, y luego avanzar a
modelos mas complejos solo si la quimica o el ajuste lo justifican.

# Preprocesado comun

## Estructura de los archivos de entrada

La rutina esta pensada principalmente para trabajar con archivos cineticos de
espectrofotometros HP/Agilent, como los HP8453 o HP8452. Estos equipos generan
archivos `*.KD`, que `tolkien-tools` detecta y convierte automaticamente a una
tabla de texto antes del analisis.

Tambien se pueden usar archivos de texto si tienen una estructura matricial
definida. La primera fila contiene los tiempos experimentales; la primera
columna contiene las longitudes de onda; el bloque central contiene las
absorbancias. Por defecto, las columnas se separan con punto y coma (`;`). La
celda superior izquierda no representa un dato experimental y puede escribirse
como `0`.

Un ejemplo generico es:

```text
0;             t_1;          t_2;          ...; t_{n_t}
lambda_1;      a_11;         a_12;         ...; a_1,n_t
lambda_2;      a_21;         a_22;         ...; a_2,n_t
...;           ...;          ...;          ...; ...
lambda_nlambda; a_nlambda,1; a_nlambda,2; ...; a_nlambda,n_t
```

En esta tabla, `a_ij` es la absorbancia medida a la longitud de onda
`lambda_i` y al tiempo `t_j`. Despues de leer el archivo, la rutina interpreta
la matriz como:

$$
A_{ij}=A(\lambda_i,t_j)
$$

con $i=1,\ldots,n_\lambda$ y $j=1,\ldots,n_t$.

## Puesta a punto de los datos antes del ajuste

Antes de ajustar un modelo cinetico, la rutina permite preparar la matriz
experimental para que el ajuste trabaje sobre la parte util del experimento.
Estas decisiones son importantes porque definen que espectros y que longitudes
de onda entran realmente al calculo.

El descarte de espectros por indice permite quitar mediciones individuales que
se consideran problematicas. Puede usarse, por ejemplo, si un espectro tiene un
artefacto evidente, una burbuja, un salto instrumental o una mezcla incompleta.

El recorte temporal permite definir desde que momento y hasta que momento se
analiza la reaccion. El recorte inicial es util cuando los primeros espectros no
corresponden todavia al regimen que se quiere ajustar; el recorte final sirve
para restringir el analisis a una fase inicial o evitar zonas tardias con deriva
instrumental.

La correccion de baseline remueve desplazamientos aditivos de absorbancia. La
rutina puede sugerir automaticamente una region plana, usar una region elegida
por el usuario, usar puntos especificos o no aplicar correccion. La eleccion
depende del experimento y de si existe una zona espectral donde no deberia haber
cambios quimicos relevantes.

El rango de longitudes de onda define la ventana espectral que se usa para el
ajuste. Conviene excluir regiones con ruido alto, saturacion, scattering fuerte o
bandas que no aportan informacion sobre las especies de interes.

La concentracion inicial `c0` fija la escala de los perfiles de concentracion.
Si no se necesita una escala absoluta, puede usarse como concentracion aparente;
si se quiere interpretar espectros o constantes en escala fisica, debe cargarse
con la concentracion experimental correspondiente.

Los espectros conocidos permiten fijar una o varias especies a partir de archivos
externos. En ese caso, la forma espectral queda impuesta por el usuario y la
rutina ajusta la escala compatible con el modelo cinetico.

Tambien se puede fijar el primer espectro experimental como espectro del
reactivo, o el ultimo como espectro del producto. Esto es util cuando esos
espectros son representativos de especies puras y se quiere evitar que la rutina
los modifique durante la recuperacion espectral.

# Obtencion de espectros puros

Una vez elegido un modelo cinetico, la rutina debe resolver dos problemas
acoplados: encontrar las constantes cineticas y construir los espectros puros de
las especies visibles. Esa construccion puede combinar informacion conocida,
provista por el usuario o tomada del propio experimento como se describe en la
seccion 4.1, con espectros recuperados por ajuste para las especies restantes,
como se describe en la seccion 4.2. Para esa recuperacion espectral, la rutina
usa el metodo NNLS.

## Espectros conocidos o fijados

La situacion mas directa ocurre cuando el usuario ya conoce el espectro de una
especie. En ese caso puede entregarlo desde un archivo externo con
`--known-spectrum`. La forma espectral queda fija y la rutina ajusta la escala
compatible con el modelo cinetico y con el resto de los datos experimentales.

Tambien se pueden fijar espectros usando el propio experimento. Con
`--fix-initial-spectrum`, la primera especie visible del modelo toma el primer
espectro experimental retenido despues del preprocesado. Ese espectro se
interpreta como espectro puro del reactivo y deja de ajustarse. Con
`--fix-final-spectrum`, la ultima especie visible del modelo toma el ultimo
espectro experimental retenido y se interpreta como espectro puro del producto.

Estas opciones son utiles cuando el experimento realmente contiene un espectro
representativo de una especie pura. Por ejemplo, si el primer espectro retenido
corresponde al reactivo antes de que haya conversion apreciable, puede fijarse
como reactivo. Si el ultimo espectro corresponde al producto final, puede
fijarse como producto.

Tambien se pueden combinar:

```bash
tolkien-tools 4 experimento.dat \
  --fix-initial-spectrum \
  --fix-final-spectrum
```

Si algunas especies tienen espectros conocidos o fijados y otras no, la rutina
mantiene fijos los espectros conocidos y ajusta solamente los espectros
restantes.

## Espectros desconocidos por NNLS

`nnls` significa *non-negative least squares*, o minimos cuadrados no
negativos. En cada prueba del ajuste, la rutina fija un conjunto de constantes
cineticas candidato. Con esas constantes calcula los perfiles de concentracion
del modelo:

$$
C_i(t_j)
$$

donde $i$ recorre las especies visibles del modelo y $j$ recorre los tiempos
experimentales. Con esos perfiles fijos, el problema espectral es lineal:

$$
A(\lambda,t_j) \simeq \sum_i E_i(\lambda)C_i(t_j)
$$

Para cada longitud de onda, la rutina busca los valores $E_i(\lambda)$ que
minimizan el error cuadratico:

$$
\sum_j
\left[
A(\lambda,t_j) -
\sum_i E_i(\lambda)C_i(t_j)
\right]^2
$$

sujeto a la restriccion:

$$
E_i(\lambda) \ge 0
$$

Esto significa que el ajuste espectral se resuelve fila por fila de la matriz de
absorbancia: para una longitud de onda dada, los datos son las absorbancias a
todos los tiempos, y las incognitas son las absorbancias molares aparentes de las
especies a esa longitud de onda.

No hay un *initial guess* espectral en el sentido usual de una optimizacion no
lineal. Para cada conjunto de constantes cineticas de prueba, los espectros que
mejor explican los datos se calculan directamente mediante el problema NNLS. La
parte no lineal del ajuste esta en las constantes cineticas; la parte espectral es
un subproblema lineal con restriccion de no negatividad.

En forma matricial, para perfiles $C$ fijos, la rutina busca $E$ tal que:

$$
A_{\mathrm{calc}} = E C
$$

minimizando:

$$
\|A-E C\|^2
$$

con:

$$
E \ge 0
$$

El error que se informa para una prueba cinetica es el residuo entre la matriz
experimental y la matriz reconstruida:

$$
R = A - E C
$$

La optimizacion externa modifica las constantes cineticas, recalcula $C(t)$,
resuelve nuevamente los espectros por NNLS y conserva el conjunto de parametros
que produce el menor residuo global.

En la implementacion hay dos niveles numericos distintos. El subproblema
espectral NNLS se resuelve directamente para cada longitud de onda: como los
modelos tienen pocas especies visibles, la rutina enumera los conjuntos activos
posibles y calcula la mejor solucion de minimos cuadrados compatible con
$E_i(\lambda)\ge 0$. Por encima de eso corre la optimizacion no lineal de las
constantes cineticas. Para los ajustes directos con `nnls`, la rutina usa
`scipy.optimize.minimize` con el metodo `Powell`, trabajando sobre el logaritmo
de las constantes para mantenerlas positivas y dentro de los rangos definidos.

El metodo de Powell es un metodo de busqueda directa: minimiza una funcion sin
usar derivadas. En lugar de calcular gradientes, realiza minimizaciones
unidimensionales sucesivas a lo largo de un conjunto de direcciones, actualizando
esas direcciones durante el proceso. Esto es util en esta rutina porque la
funcion objetivo incluye subproblemas NNLS y no tiene una forma analitica suave
y simple para derivar respecto de las constantes cineticas.

La ventaja practica de NNLS es que evita espectros puros con absorbancias
negativas no fisicas. Esto hace que el ajuste sea menos flexible que una
pseudoinversa sin restricciones, pero normalmente mas estable y mas interpretable
quimicamente.

# Modelos generales

Los modelos generales describen esquemas cineticos simples que no dependen de
una quimica particular. Sirven como punto de partida para conversiones
elementales, reacciones consecutivas e intermediarios reversibles. La rutina
incluye:

- $A \longrightarrow B$: conversion irreversible de primer orden entre dos
  especies visibles.
- $A \longrightarrow B \longrightarrow C$: mecanismo consecutivo irreversible
  con un intermedio espectralmente distinguible.
- $A \rightleftharpoons B \longrightarrow C$: formacion reversible de un
  intermedio seguida de conversion irreversible a producto.

## Como se obtienen las constantes cineticas

En todos los modelos generales, las constantes cineticas se obtienen por el
mismo esquema global. Para un conjunto de constantes de prueba, la rutina
calcula los perfiles de concentracion $C_i(t)$ del mecanismo elegido. Con esos
perfiles fijos, recupera los espectros puros desconocidos por NNLS y reconstruye
la matriz de absorbancias:

$$
A_{\mathrm{calc}} = E C
$$

El residuo del ajuste es:

$$
R = A - A_{\mathrm{calc}}
$$

La funcion objetivo que se minimiza es la norma global de ese residuo. La
optimizacion externa modifica las constantes cineticas, recalcula los perfiles
de concentracion, vuelve a resolver los espectros por NNLS y conserva el
conjunto de constantes que da el menor error.

Las constantes se optimizan en espacio logaritmico, de modo que durante la
busqueda permanecen positivas. En la implementacion actual, los modelos
generales usan el metodo de Powell como minimizador externo.

## $A \longrightarrow B$

Este modelo describe la conversion directa de una especie inicial $A$ en un
producto $B$, sin intermediarios. Es el caso mas simple: hay una sola constante
cinetica aparente, $k$, y dos especies absorbentes.

$$
A \longrightarrow B
$$

Ecuaciones diferenciales:

$$
\frac{d[A]}{dt}=-k[A],\qquad
\frac{d[B]}{dt}=k[A]
$$

con condiciones iniciales:

$$
[A](0)=c_0,\qquad [B](0)=0
$$

La solucion analitica usada para los perfiles de concentracion es:

$$
[A](t)=c_0 e^{-kt},\qquad
[B](t)=c_0-[A](t)
$$

Las especies absorbentes son $A$ y $B$. La unica variable cinetica ajustada es
$k$. Durante el ajuste, la rutina prueba distintos valores de $k$; para cada
valor calcula $[A](t)$ y $[B](t)$, recupera los espectros de $A$ y $B$ por NNLS
y evalua el residuo global. El valor informado de $k$ es el que minimiza ese
residuo.

Este modelo es apropiado para conversiones simples entre dos especies
espectralmente distinguibles, cuando no hay evidencia de intermediarios
coloreados.

## $A \longrightarrow B \longrightarrow C$

Este modelo describe una reaccion consecutiva en la que $A$ forma un
intermedio $B$, y $B$ se transforma luego en el producto $C$. Las dos constantes
cineticas ajustadas son $k_1$ para el primer paso y $k_2$ para el segundo.

$$
A \longrightarrow B \longrightarrow C
$$

Ecuaciones diferenciales:

$$
\frac{d[A]}{dt}=-k_1[A],\qquad
\frac{d[B]}{dt}=k_1[A]-k_2[B],\qquad
\frac{d[C]}{dt}=k_2[B]
$$

con condiciones iniciales:

$$
[A](0)=c_0,\qquad [B](0)=0,\qquad [C](0)=0
$$

Cuando $k_1 \ne k_2$, los perfiles analiticos son:

$$
[A](t)=c_0 e^{-k_1t},\qquad
[B](t)=c_0\frac{k_1}{k_2-k_1}\left(e^{-k_1t}-e^{-k_2t}\right),\qquad
[C](t)=c_0-[A](t)-[B](t)
$$

Si $k_1$ y $k_2$ coinciden, el codigo usa el limite analitico correspondiente.

Las especies absorbentes son $A$, $B$ y $C$. Las variables cineticas ajustadas
son $k_1$ y $k_2$. Durante el ajuste, la rutina modifica simultaneamente ambas
constantes; para cada par de valores calcula los perfiles de $A$, $B$ y $C$,
recupera sus espectros por NNLS y evalua el residuo global. El modelo es util
cuando la matriz experimental contiene evidencia de tres contribuciones
espectrales y se espera un intermedio acumulable.

## $A \rightleftharpoons B \longrightarrow C$

Este modelo permite que la formacion del intermedio $B$ sea reversible. La
especie $A$ se convierte en $B$ con constante $k_1$, $B$ vuelve a $A$ con
constante $k_{-1}$, y $B$ tambien puede avanzar irreversiblemente hacia el
producto $C$ con constante $k_2$.

$$
A \rightleftharpoons B \longrightarrow C
$$

Ecuaciones diferenciales:

$$
\frac{d[A]}{dt}=-k_1[A]+k_{-1}[B],\qquad
\frac{d[B]}{dt}=k_1[A]-(k_{-1}+k_2)[B],\qquad
\frac{d[C]}{dt}=k_2[B]
$$

con condiciones iniciales:

$$
[A](0)=c_0,\qquad [B](0)=0,\qquad [C](0)=0
$$

La rutina escribe este sistema en forma matricial. La idea es juntar las tres
concentraciones en un unico vector:

$$
\mathbf{c}(t)=
\begin{pmatrix}
[A](t)\\
[B](t)\\
[C](t)
\end{pmatrix}
$$

Entonces, las tres ecuaciones diferenciales se pueden escribir juntas como:

$$
\frac{d\mathbf{c}}{dt}=K\mathbf{c}
$$

donde la matriz cinetica $K$ se arma con los coeficientes que multiplican a
$[A]$, $[B]$ y $[C]$ en las ecuaciones diferenciales:

$$
K=
\begin{pmatrix}
-k_1 & k_{-1} & 0\\
k_1 & -(k_{-1}+k_2) & 0\\
0 & k_2 & 0
\end{pmatrix}
$$

La primera fila representa la ecuacion para $d[A]/dt$, la segunda la ecuacion
para $d[B]/dt$ y la tercera la ecuacion para $d[C]/dt$. Por ejemplo, como
$d[A]/dt=-k_1[A]+k_{-1}[B]+0[C]$, la primera fila de $K$ es
$(-k_1,\ k_{-1},\ 0)$.

con:

$$
\mathbf{c}(0)=
\begin{pmatrix}
c_0\\
0\\
0
\end{pmatrix}
$$

La solucion formal es:

$$
\mathbf{c}(t)=\exp(Kt)\,\mathbf{c}(0)
$$

donde $\exp(Kt)$ es la exponencial de una matriz. Es la version matricial de la
solucion de una ecuacion de primer orden simple. Para una sola especie,
$d[A]/dt=-k[A]$ da $[A](t)=e^{-kt}[A](0)$; para varias especies acopladas, el
numero $-k$ queda reemplazado por la matriz $K$. En la implementacion, esa
solucion se obtiene diagonalizando $K$.

Para escribir la solucion analitica, se definen:

$$
s=k_1+k_{-1}+k_2,\qquad
\Delta=\sqrt{s^2-4k_1k_2},\qquad
\alpha=\frac{s-\Delta}{2},\qquad
\beta=\frac{s+\Delta}{2}
$$

Con esas definiciones, los perfiles temporales son:

$$
[B](t)=c_0\frac{k_1}{\Delta}
\left(e^{-\alpha t}-e^{-\beta t}\right)
$$

$$
[A](t)=\frac{c_0}{\Delta}
\left[
(k_{-1}+k_2-\alpha)e^{-\alpha t}
+(\beta-k_{-1}-k_2)e^{-\beta t}
\right]
$$

$$
[C](t)=c_0-[A](t)-[B](t)
$$

Las especies absorbentes son $A$, $B$ y $C$. Las variables cineticas ajustadas
son $k_1$, $k_{-1}$ y $k_2$. Luego, como en los otros modelos generales, los
espectros visibles se recuperan por NNLS y se evalua el residuo global.

Este modelo puede ser sensible a correlaciones entre parametros, especialmente
si el intermedio no se acumula de forma clara o si sus bandas se parecen mucho a
las de $A$ o $C$. Conviene compararlo contra modelos mas simples antes de
interpretar mecanisticamente las tres constantes. Como recomendacion practica,
antes de analizar un experimento con el mecanismo reversible conviene ajustarlo
tambien con el modelo irreversible $A \longrightarrow B \longrightarrow C$ y
comparar la calidad del ajuste. Si el modelo irreversible ya describe bien los
datos, el uso del modelo reversible puede introducir una sobreparametrizacion
sin una ganancia experimental clara.

# Modelos especiales

Los modelos especiales fueron definidos para reacciones especificas estudiadas
por el grupo. A diferencia de los modelos generales, no pretenden describir
cualquier proceso cinetico simple, sino mecanismos particulares que incorporan
hipotesis quimicas propias de ciertos sistemas experimentales. Por eso se
presentan en una seccion separada: pueden ser muy utiles para esos sistemas,
pero no deben usarse como modelos generales fuera del contexto para el que
fueron formulados.

## Como se obtienen las constantes cineticas

El ajuste mantiene la misma logica global que en los modelos generales: para un
conjunto de constantes de prueba se calculan perfiles de concentracion, luego se
recuperan los espectros visibles por NNLS y finalmente se evalua el residuo
global entre la matriz experimental y la matriz reconstruida. La diferencia es
que algunos modelos especiales no tienen perfiles temporales simples y requieren
resolver ecuaciones diferenciales no lineales muchas veces durante la
optimizacion.

El modelo autocatalitico de dos especies tiene una solucion analitica compacta,
por lo que su evaluacion es relativamente rapida. En cambio, los modelos con
binding inicial y con transsulfuracion por HSS- se integran numericamente con
`scipy.integrate.solve_ivp`, usando el metodo `DOP853`. Como cada prueba de
constantes requiere volver a integrar el sistema y volver a resolver NNLS, estos
ajustes pueden demorar mas que los modelos generales. Esto es especialmente
notorio en el modelo con HSS-, que tiene mas parametros cineticos y especies
internas.

En la implementacion actual, el modelo autocatalitico simple y el modelo con
binding inicial usan Powell como optimizador externo. El modelo con HSS- usa
`L-BFGS-B` en espacio logaritmico para acelerar la busqueda dentro de los
rangos permitidos.

## Reducción de MbFeIII por HS- sin binding inicial

Este modelo describe experimentos en los que el binding inicial de $\mathrm{HS^-}$
ya ocurrio antes de la ventana temporal analizada. Por lo tanto, la especie
inicial visible se toma como $\mathrm{MbFeIII\!-\!HS}$ y no se modela la etapa de
coordinacion. La reduccion hacia $\mathrm{MbFeII}$ comienza por una via lenta
asociada al sulfuro coordinado, pero se acelera a medida que avanza la reaccion.

$$
\mathrm{MbFeIII\!-\!HS}
\longrightarrow
\mathrm{MbFeII}+\mathrm{HS^\bullet}
$$

Los radicales sulfurados formados no se siguen explicitamente. Se asume que
terminan generando polisulfuros por vias no detalladas:

$$
\mathrm{HS^\bullet}\longrightarrow \cdots \longrightarrow
\mathrm{polysulfides}
$$

Estos polisulfuros pueden coordinar o reaccionar con la mioglobina ferrica y
acelerar la reduccion:

$$
\mathrm{MbFeIII}+\mathrm{polysulfides}
\longrightarrow
\mathrm{MbFeII}
$$

Como no se conoce la estequiometria exacta de formacion de polisulfuros, y como
las especies ferricas $\mathrm{MbFeIII\!-\!S_n}$ tienen espectros indistinguibles
del complejo $\mathrm{MbFeIII\!-\!HS}$ en esta rutina, el proceso se modela de
manera fenomenologica como una autocatálisis. La constante autocatalitica
$k_{\mathrm{cat}}$ va ganando peso conforme se forma producto; en la salida del
codigo este parametro aparece como `k_auto`.

Las especies que se siguen explicitamente en la cinetica visible son
$\mathrm{MbFeIII\!-\!HS}$ y $\mathrm{MbFeII}$. Si:

$$
x=\frac{[\mathrm{MbFeII}]}{c_0}
$$

entonces las ecuaciones diferenciales para esas especies son:

$$
\frac{d[\mathrm{MbFeIII\!-\!HS}]}{dt}
=-(k_{\mathrm{slow}}+k_{\mathrm{cat}}x)[\mathrm{MbFeIII\!-\!HS}],\qquad
\frac{d[\mathrm{MbFeII}]}{dt}
=(k_{\mathrm{slow}}+k_{\mathrm{cat}}x)[\mathrm{MbFeIII\!-\!HS}]
$$

con condiciones iniciales
$[\mathrm{MbFeIII\!-\!HS}](0)=c_0$ y $[\mathrm{MbFeII}](0)=0$. Las constantes
ajustadas son $k_{\mathrm{slow}}$, que describe la fase lenta inicial, y
$k_{\mathrm{cat}}$ (`k_auto`), que describe la aceleracion aparente asociada a la
formacion de especies reactivas de azufre. Esta constante no debe interpretarse
como una constante elemental de reduccion por una especie quimica unica.

## Reducción de MbFeIII por HS- con binding inicial

Este modelo agrega una etapa inicial de coordinacion por sulfuro. Es apropiado
cuando el experimento empieza con mioglobina ferrica libre y la formacion del
complejo $\mathrm{MbFeIII\!-\!HS}$ ocurre dentro de la ventana temporal
observada. Luego, el complejo coordinado se reduce hacia $\mathrm{MbFeII}$ con
la misma aceleracion autocatalitica fenomenologica del modelo anterior.

$$
\mathrm{MbFeIII}+\mathrm{HS^-}
\longrightarrow
\mathrm{MbFeIII\!-\!HS}
\longrightarrow
\mathrm{MbFeII}
$$

La variable autocatalitica vuelve a ser:

$$
x=\frac{[\mathrm{MbFeII}]}{[\mathrm{Mb}]_{\mathrm{total}}}
$$

Las ecuaciones diferenciales son:

$$
\begin{aligned}
\frac{d[\mathrm{MbFeIII}]}{dt}
&=-k_{\mathrm{on}}[\mathrm{MbFeIII}],\\
\frac{d[\mathrm{MbFeIII\!-\!HS}]}{dt}
&=k_{\mathrm{on}}[\mathrm{MbFeIII}]
-(k_{\mathrm{slow}}+k_{\mathrm{auto}}x)[\mathrm{MbFeIII\!-\!HS}],\\
\frac{d[\mathrm{MbFeII}]}{dt}
&=(k_{\mathrm{slow}}+k_{\mathrm{auto}}x)[\mathrm{MbFeIII\!-\!HS}]
\end{aligned}
$$

con condiciones iniciales
$[\mathrm{MbFeIII}](0)=c_0$,
$[\mathrm{MbFeIII\!-\!HS}](0)=0$ y
$[\mathrm{MbFeII}](0)=0$. Las especies absorbentes visibles son
$\mathrm{MbFeIII}$, $\mathrm{MbFeIII\!-\!HS}$ y $\mathrm{MbFeII}$. Las
constantes ajustadas son $k_{\mathrm{on}}$, $k_{\mathrm{slow}}$ y
$k_{\mathrm{auto}}$. En un experimento individual, $k_{\mathrm{on}}$ se trata
como constante aparente pseudo-primer orden; para estimar una constante
bimolecular habria que dividir por la concentracion efectiva de $\mathrm{HS^-}$,
si se conoce y se mantiene en exceso.

Este modelo se integra numericamente porque el termino autocatalitico acopla la
velocidad de reduccion con la fraccion reducida $x$. La integracion se realiza
para cada conjunto de constantes probado durante el ajuste.

## Reducción de MbFeIII por HS- con agregado de HSS-

Este modelo describe los experimentos en los que primero se forma rapidamente
$\mathrm{MbFeIII\!-\!HS}$ por agregado de un gran exceso de $\mathrm{HS^-}$ y
luego se agrega una cantidad conocida de $\mathrm{HSS^-}$ externo. La hipotesis
del mecanismo es que el $\mathrm{HSS^-}$ transfiere azufre al sulfuro
coordinado, formando una especie $\mathrm{MbFeIII\!-\!HSS}$ que se reduce mas
rapido hacia $\mathrm{MbFeII}$.

Aunque $\mathrm{MbFeIII\!-\!HS}$ y $\mathrm{MbFeIII\!-\!HSS}$ son especies
cineticamente distintas, se consideran espectralmente indistinguibles en esta
rutina. Por eso el ajuste ve una unica especie ferrica coordinada, que es la
suma de ambas. Ademas, la via autocatalitica endogena del mecanismo con
$\mathrm{HS^-}$ sigue presente.

La relacion experimental fija es:

$$
R_{\mathrm{HSS}}=
\frac{[\mathrm{HSS^-}]_{\mathrm{agregado}}}
{[\mathrm{Mb}]_{\mathrm{total}}}
$$

Internamente, el modelo usa:

```text
A = MbFeIII-HS
B = MbFeIII-HSS
P = MbFeII
S = HSS- libre agregado efectivo
x = P / [Mb]total
```

Las ecuaciones diferenciales son:

$$
\begin{aligned}
\frac{dA}{dt}&=-(k_{\mathrm{slow}}+k_{\mathrm{auto}}x)A-k_{\mathrm{ts}}AS,\\
\frac{dB}{dt}&=k_{\mathrm{ts}}AS-k_{\mathrm{fast}}B,\\
\frac{dP}{dt}&=(k_{\mathrm{slow}}+k_{\mathrm{auto}}x)A+k_{\mathrm{fast}}B,\\
\frac{dS}{dt}=-k_{\mathrm{ts}}AS
\end{aligned}
$$

con condiciones iniciales:

$$
A(0)=c_0,\quad B(0)=0,\quad P(0)=0,\quad S(0)=R_{\mathrm{HSS}}c_0
$$

Las especies absorbentes visibles son:

$$
\mathrm{MbFeIII\!-\!S_x}=A+B,\qquad
\mathrm{MbFeII}=P
$$

Las constantes ajustadas son $k_{\mathrm{slow}}$, $k_{\mathrm{auto}}$,
$k_{\mathrm{ts}}$ y $k_{\mathrm{fast}}$. El parametro $k_{\mathrm{ts}}$
representa la transsulfuracion bimolecular por $\mathrm{HSS^-}$ agregado,
$k_{\mathrm{fast}}$ representa la reduccion rapida del intermedio
$\mathrm{MbFeIII\!-\!HSS}$, y $k_{\mathrm{auto}}$ conserva la aceleracion
fenomenologica asociada a polisulfuros endogenos.

Este modelo se integra numericamente con `solve_ivp`. Como $A$ y $B$ comparten
espectro, $k_{\mathrm{ts}}$ y $k_{\mathrm{fast}}$ pueden estar correlacionados
si el intermedio no se acumula de una manera cineticamente distinguible. Por eso
conviene interpretar esos parametros junto con la calidad del ajuste, la serie
de experimentos a distintos $R_{\mathrm{HSS}}$ y la estabilidad de los espectros
recuperados.

# Archivos exportados

Cada ajuste aceptado genera una carpeta:

```text
results_<archivo>
```

Dentro se escriben:

```text
<archivo>_concentrations.dat
<archivo>_pure_spectra.dat
<archivo>_fit_summary.dat
<archivo>_fit_panel.png
```

Si la carpeta de salida no es escribible, la rutina usa una carpeta alternativa
numerada, por ejemplo:

```text
results_<archivo>_1
```

# Recomendaciones practicas

- Usar espectros conocidos o fijados cuando haya una razon experimental clara
  para considerarlos especies puras.
- Comparar modelos simples contra modelos complejos antes de interpretar
  constantes mecanisticas.
- Si se fija el primer espectro como reactivo, asegurarse de que la poda
  temporal realmente deje como primer espectro una muestra representativa del
  reactivo.
- Si se fija el ultimo espectro como producto, asegurarse de que la reaccion
  haya llegado suficientemente cerca del producto final.
- En modelos con especies espectralmente indistinguibles, recordar que el
  ajuste solo ve la suma de esas especies, no sus concentraciones internas por
  separado.

# Referencias

Malinowski, E. R. (2002). *Factor Analysis in Chemistry* (3rd ed.).
Wiley-Interscience, New York. ISBN 978-0-471-13479-4.

Powell, M. J. D. (1964). An efficient method for finding the minimum of a
function of several variables without calculating derivatives. *The Computer
Journal*, 7(2), 155-162. https://doi.org/10.1093/comjnl/7.2.155

SciPy Developers. `scipy.optimize.minimize`, method `Powell`.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
