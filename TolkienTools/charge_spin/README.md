# Charge And Spin Analysis

Rutina conectada actualmente:

```text
TolkienTools/charge_spin/charge_spin_analysis.py
```

Uso desde el menu maestro:

```bash
tolkien-tools
```

Este modulo contiene la rutina de analisis de cargas Mulliken, CHELPG, Lowdin,
Hirshfeld y poblaciones de spin.

En el modo individual, todos los archivos generados se escriben dentro de
`charge_spin_results/` en la carpeta desde la que se ejecuta el analisis. Esto
incluye archivos ORCA/LIO consolidados, series, promedios, histogramas, figuras,
reportes y visores HTML. Los outputs originales, XYZ y demas inputs permanecen
en la carpeta principal.

Estructura interna:

- `charge_spin_analysis.py`: entrada compatible usada por el launcher.
- `charge_spin_cli.py`: flujo interactivo y orquestacion.
- `charge_spin_common.py`: utilidades, prompts y configuraciones.
- `charge_spin_io.py`: lectura/escritura de archivos consolidados y series.
- `charge_spin_orca.py`: extraccion de poblaciones desde outputs ORCA.
- `charge_spin_stats.py`: KDE, histogramas, normalizacion y checks de spin.
- `charge_spin_plotting.py`: figuras de histogramas y series temporales.
- `charge_spin_viewer.py`: visor HTML de localizacion de spin y geometria.
- `charge_spin_coordination.py`: propuesta de ligandos y componentes
  moleculares para complejos de coordinacion.
- `charge_spin_global.py`: comparaciones globales entre subdirectorios.

En el modo global entre subdirectorios, el programa genera siempre la figura
normal `global_<analisis>_histograms.png`. Antes de escribir las figuras tambien
pregunta si se quiere una figura de paper; si se responde que si, agrega
`global_<analisis>_histograms_paper.png`, una version limpia sin titulos,
leyendas, nombres de ejes ni numeros de ejes, pero conservando las marcas de
ticks para editar externamente.

El modo global acepta tanto atomos individuales como actores definidos en el
modo de fragmentos. La opcion de reutilizar entidades lee
`spin_fragment_definitions.dat` y busca las series `actor_<nombre>_*` del
ultimo analisis, evitando mezclar archivos viejos de atomos o fragmentos.
Tambien se pueden ingresar nombres como `Fe_Porphyrin`, `X1` o `X2`
manualmente. Guiones, puntos y guiones bajos se normalizan para poder comparar,
por ejemplo, `Fe-Porphyrin` con `Fe_Porphyrin` entre sistemas.
Para análisis nuevos, busca estos archivos dentro de
`charge_spin_results/`. También conserva compatibilidad con resultados
anteriores escritos directamente en cada carpeta de sistema.

Para poblaciones de spin, el flujo interactivo permite elegir como se arman la
estadistica, los histogramas y los archivos de salida:

- `Raw spin values from each snapshot`: usa directamente el valor de spin que
  sale de cada foto/output. Esta es la opcion por defecto.
- `Spin fraction relative to the selected atoms`: normaliza cada snapshot por
  la suma de spin de los atomos/entidades elegidos.

Si hay poblaciones de spin, la seleccion de atomos puede hacerse manualmente o
automaticamente por localizacion de spin. El modo automatico calcula, para cada
snapshot, `abs(spin_atom) / sum(abs(spin_todos_los_atomos))`, promedia esa
fraccion en la dinamica y genera histogramas individuales solo para los atomos
que superan el umbral minimo pedido (5% por defecto). El resto de los atomos se
agrupa como una entidad `resto`, con su propio histograma.

Tambien existe un modo de fragmentos moleculares para analizar zonas como
Fe/porfirina/agua/histidina. Antes de pedir la composicion de cada fragmento,
el flujo permite elegir entre definicion manual y una propuesta automatica
para complejos de coordinacion.

La propuesta automatica detecta metales de transicion, infiere contactos
metal-atomo por distancia, retira los metales del grafo molecular y asigna
`L1`, `L2`, ..., a los componentes conectados restantes. Para cada componente
informa si esta coordinado, su numero y tipos de atomos, los atomos donores y
la denticidad. Un componente cercano pero no enlazado al metal queda marcado
explicitamente como no coordinado. La propuesta se guarda en
`coordination_ligand_proposal.dat` y se muestra en
`coordination_ligand_viewer.html`, con un color por componente y etiquetas de
indices y grupos. Es una ayuda basada en distancias y debe revisarse antes de
aceptarla.

En WSL, cuando se pide abrir un visor, la rutina ejecuta `explorer.exe .` en
la carpeta que contiene el HTML para evitar los errores de `xdg-open`; el
archivo se abre manualmente con doble clic. En Linux nativo intenta abrir el
HTML con el navegador predeterminado.

Al definir los fragmentos se pueden combinar los grupos detectados y los
metales, por ejemplo `Fe88 + L1`, además de usar atomos separados por espacios
o comas, rangos como `10-18`, y el token `remaining`. En modo manual se genera
`spin_fragment_numbering_viewer.html` con todos los atomos numerados. Cada
fragmento se analiza como una entidad cuyo valor de spin por snapshot es la
suma de los spines atomicos que lo forman. Las definiciones usadas se guardan
en `spin_fragment_definitions.dat` y las salidas principales llevan el sufijo
`with_fragments`. Si quedan atomos sin asignar a ningun fragmento, el programa
avisa cuales son y permite dejarlos fuera del analisis, agruparlos
automaticamente como `resto`, o redefinir los fragmentos. Si se dejan fuera,
el modo de spin crudo conserva las poblaciones de los fragmentos elegidos y el
modo de fraccion normaliza cada snapshot usando solamente la suma de esos
fragmentos.

En modo ORCA, aunque el histograma se arme con muchos `SP_*.out`, el visor
`spin_localization_viewer.html` usa una unica geometria representativa: toma el
primer output ORCA ordenado que contenga un bloque
`CARTESIAN COORDINATES (ANGSTROEM)`. En modo LIO, si existe un XYZ en la carpeta
(`qm_completo.xyz`, `qm.xyz` u otro `*.xyz`), el mismo visor se genera desde ese
XYZ. El visor escribe conectividad explicita inferida por distancias
conservadoras para evitar enlaces largos espurios del auto-bonding de 3Dmol. En
ambos casos, el flujo puede intentar abrir el HTML en el navegador por defecto.
