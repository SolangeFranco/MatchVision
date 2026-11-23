# Anotación humana y validación de identidades

El sistema actual de ID persistente es experimental. Un menor número de ID,
una puntuación de apariencia alta o una visualización plausible **no prueban**
que se haya reconocido correctamente a las personas. No hay anotaciones
humanas de identidad de un partido real ni métricas de precisión verificadas
con ese tipo de datos. El protocolo local para comparar contra un conjunto
congelado está en [model-evaluation.md](model-evaluation.md#evaluacion-local-independiente);
las pruebas sintéticas no constituyen validación deportiva.
 El protocolo operativo, muestreo reproducible y estado de revisión están en
 [ground-truth-protocol.md](ground-truth-protocol.md).

## Herramientas disponibles

- [`tools/annotation/reid_annotator.html`](../tools/annotation/reid_annotator.html)
  es un anotador local para navegador. La anotación por fotograma funciona
  directamente sobre el video original y no requiere cargar predicciones,
  trayectorias ni propuestas. Los candidatos automáticos son opcionales y
  nunca se aceptan como verdad automáticamente. Se pueden dibujar/corregir
  recuadros, asignar IDs humanos persistentes, clasificar jugadores,
  porteros, árbitros, balón u otros, marcar observaciones desconocidas,
  ambiguas o ignoradas, y registrar manualmente apariciones, desapariciones,
  oclusiones, cruces entre jugadores y cambios de escena/cámara.
- Las anotaciones se guardan automáticamente en IndexedDB, vinculadas al
  SHA-256 del video original. También se pueden descargar e importar como
  archivo JSON portátil.
- Los fotogramas marcados como exhaustivos se incluyen en la evaluación solo
  si sus observaciones tienen identidad resuelta. No se debe adivinar una
  identidad ni marcar como exhaustivo un fotograma sin revisar la imagen
  completa. Usa siempre la misma etiqueta humana para la misma persona; no
  copies IDs del tracker.
- Antes de mirar resultados de modelos, congela la exportación con el botón
  **Congelar conjunto de prueba** o con
  `matchvision-freeze-identity-set --input anotaciones.json --output
  conjunto-prueba.json`. El comando valida el JSON y crea un archivo nuevo sin
  sobrescribir otro existente. Un conjunto congelado debe contener al menos un
  fotograma exhaustivo con identidades resueltas; la huella SHA-256 permite
  detectar cambios posteriores en las etiquetas. Conserva el original y el
  archivo congelado por separado. La huella comprueba integridad, no calidad
  de anotación.
- El esquema JSON versión 1 está en
  `src/matchvision/schemas/identity-annotations.schema.json`. El validador
  Python comprueba clases, eventos, coordenadas, rangos, IDs repetidos en un
  fotograma, consistencia de tipos para cada identidad y la huella del conjunto
  congelado.
- `matchvision-select-review-cases` prioriza propuestas ReID, huecos,
  reinicios de tracks y proximidades entre recuadros. Son **candidatos de
  revisión**, no cruces, oclusiones ni errores confirmados.
- `matchvision-evaluate-identities` compara dos archivos MOT sobre los mismos
  fotogramas exhaustivos aptos, verifica el hash del video y guarda un informe
  estructurado.

La congelación no impide que una persona edite el archivo con otra herramienta;
si el contenido cambia, el validador Python rechaza la huella. No se debe
modificar el conjunto de prueba después de consultar métricas. La primera
anotación continúa siendo responsabilidad humana: actualmente no hay Ground
Truth real ni evaluación independiente completa de detección, tracking y ReID.

## Casos preparados

El video original se decodificó en 958 fotogramas a 29.97002997002997 FPS,
aunque la fuente declara 959. Para estos resultados se debe usar el número
**procesado**, 958. Los artefactos originales de YOLO/ByteTrack y ReID no se
modifican.

El archivo ya generado `outputs/identity-review/review-cases-final.json`
contiene 120 casos:

| Prioridad | Tipo de caso candidato | Cantidad |
|---|---|---:|
| 1 | Propuesta ReID con score inferior a 0.86 | 14 |
| 2 | Otras propuestas ReID | 1 |
| 3 | Hueco/desaparición dentro de un track | 48 |
| 4 | Reinicio cercano de ID temporal | 48 |
| 5 | Recuadros próximos o solapados; posible cruce/oclusión | 9 |

Las cantidades describen el selector, **no eventos confirmados**. La interfaz
presenta un caso por vez en ese orden, muestra su prioridad, fotogramas y
evento, y permite saltar directamente al fotograma del evento. Usa los botones
de caso o ↑/↓; usa ←/→ para revisar los fotogramas contiguos.

Para regenerar los casos, sin sobrescribir el archivo existente, cambia la ruta
de salida:

```powershell
python -m matchvision.tracking.review_cli `
  --baseline-mot outputs\video-runs\8ec67506-93cc-4f17-b80f-d98faabea23f.txt `
  --trajectories outputs\video-runs\4052ed58-6c37-4a2c-b62a-920dfe1e700b_trajectories.csv `
  --video data\videos\amateur-football-clip.webm `
  --frame-count 958 `
  --maximum-cases 120 `
  --radius-frames 30 `
  --minimum-crossing-distance-heights 1.25 `
  --output outputs\identity-review\review-cases-revision-2.json
```

## Anotar Ground Truth independiente

Desde la raíz del proyecto, inicia un servidor local:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Abre `http://127.0.0.1:8765/tools/annotation/reid_annotator.html`, carga el
video original y pulsa **Verificar correspondencia**. Introduce el FPS,
número de fotogramas procesados y dimensiones originales; usa valores
verificados con `matchvision-inspect-video`, no los infieras de las
predicciones. Para el video ya preparado, los valores confirmados son
29.97002997002997 FPS, 958 fotogramas procesados y 3840×2160 píxeles. Ya existe
una copia H.264 compatible con navegador, con 958 fotogramas en orden y el
mismo FPS racional que el original; la copia reducida a 1920×1080 conserva la
correspondencia espacial y las anotaciones se guardan en coordenadas del
original. La copia es opcional; si se usa, carga además el manifiesto para
verificar la correspondencia. No edites ni reemplaces los artefactos ni uses
copias que hayan cambiado el orden, eliminado fotogramas o alterado la
temporización.

1. Revisa cada fotograma completo que quieras medir. Usa los controles de
   fotograma y reproduce/pausa para revisar entradas, salidas y oclusiones.
2. Dibuja cada recuadro manualmente. Selecciona tipo de objeto y asigna una
   etiqueta humana estable. Corrige o borra errores; el mismo ID no puede
   representar dos objetos en un fotograma.
3. Registra eventos solo cuando se observen y se puedan atribuir a una
   identidad. No infieras una desaparición por un simple hueco del tracker.
4. Marca **Todas las personas visibles están anotadas** únicamente cuando se
   hayan resuelto todas las observaciones relevantes del fotograma. Los
   fotogramas con cajas desconocidas, ambiguas o ignoradas no entran en las
   métricas.
5. El guardado local es automático. Descarga el JSON periódicamente como
   respaldo; para continuar, vuelve a cargar el mismo video en el mismo perfil
   del navegador o importa el JSON.
6. Antes de ejecutar modelos o mirar sus métricas, congela y conserva una copia
   de solo lectura del Ground Truth. Verifica el archivo importándolo de nuevo
   en la interfaz; no edites el conjunto congelado.

Para continuar una revisión anterior, importa el JSON previo. Los archivos de
predicciones, casos priorizados y propuestas ReID son opcionales y sirven solo
como apoyo visual; su carga no es necesaria para producir Ground Truth
independiente.

## Preparar evidencia visual automática

El apoyo visual es opcional y no modifica las anotaciones. Reutiliza el
descriptor HSV/HOG y el detector de cambios bruscos existentes. Genera nuevas
salidas (el comando se niega a sobrescribirlas). El artefacto v3 mencionado
arriba es histórico; la versión actual genera v4:

```powershell
python -m matchvision.reidentification.review_assistant `
  --output outputs\identity-review\reid-review-proposals-v4.json `
  --evidence-dir outputs\identity-review\reid-review-evidence-v4
```

Por defecto procesa el video de revisión, manifiesto, trayectorias y casos
finales ya preparados. Verifica hashes, dimensiones y recuento de fotogramas
antes de producir resultados.

La salida contiene fotogramas al inicio, evento y final de cada caso,
comparaciones de candidatos no solapados, similitudes HSV/HOG, señales
aproximadas de movimiento/escala y alertas heurísticas de calidad, oclusión o
apariciones muestreadas de cada track. Las asociaciones que atraviesan un
cambio de escena se descartan. Estos avisos no confirman que haya ocurrido un evento. Los
scores ordenan candidatos, pero no son probabilidades ni identidades.

Las propuestas de enlace repetidas se agrupan por pareja de tracks y se
marcan con `duplicate_group_id`. Los grupos exploratorios de posible identidad
se forman como componentes conectados solo entre enlaces con score heurístico
igual o superior a 0.82. Se buscan pares de tracks que coincidan en un mismo
fotograma; si existen, el grupo se marca con conflicto y requiere revisión.
Incluso un grupo sin conflicto temporal es solo una ayuda para organizar la
revisión: no demuestra una identidad común.

En el anotador, carga **Apoyo visual ReID JSON** tras cargar los videos y los
casos. La sección de apoyo muestra evidencia, apariciones muestreadas y permite saltar a sus
fotogramas. No introduce observaciones, no propone valores para el campo de
identidad, no guarda anotaciones y nunca marca fotogramas exhaustivos. La
persona que revisa debe decidir manualmente si existe evidencia suficiente;
si no, debe dejar el caso sin resolver o marcar la observación como ambigua.

Para registrar comparaciones de HSV/HOG, embeddings y modo híbrido, instala
`.[reid]`, descarga los pesos verificados y ejecuta el mismo comando con
`--reid-method hybrid`. El JSON incluye `method_scores` por propuesta y el
método configurado determina el orden. Carga ese JSON en «Apoyo visual ReID».
Las señales aprendidas son asistencia visual solamente: no confirman
identidades ni sustituyen la revisión manual descrita en esta guía.

La generación híbrida v4 ya procesó los 120 casos existentes y produjo 246
propuestas con comparaciones de los tres métodos. Se codificaron 434 recortes
con OSNet en CPU en 136,65 s. No se asignó ningún nombre, dorsal o identidad
humana y no se calcularon métricas de precisión; la revisión manual sigue
pendiente.

Cada propuesta tiene controles de decisión separados. «Aceptar como hipótesis
de misma persona» registra una decisión humana de revisión, no una identidad.
Los cruces simultáneos y continuidad dentro de un track usan textos de
decisión distintos para no presentarlos como asociaciones de identidad. Las
decisiones se guardan de forma incremental en IndexedDB junto a la revisión
local del video, pero separadas de las observaciones anotadas. Al recargar el
video en el mismo perfil del navegador se recuperan anotaciones, decisiones e
historial de cambios. Usa «Exportar decisiones de revisión (JSON)» para
guardar una copia portátil independiente; no es ground truth. Las propuestas
duplicadas conservan decisiones individuales y la interfaz advierte si las
apariciones del mismo par reciben decisiones contradictorias.
También señala cuando una pareja se presenta con etiquetas de relación
distintas en propuestas diferentes. Esa variación de etiqueta requiere revisar
cada caso, pero por sí sola no indica identidades incompatibles.

## Identidades humanas y Ground Truth ReID

La sección **Identidades humanas y Ground Truth** gestiona asociaciones
confirmadas entre tracks; es distinta de las anotaciones manuales de recuadros
por fotograma y no marca fotogramas como exhaustivos.

1. Revisa las apariciones y el contexto temporal de una propuesta. Registra
   **Aceptar**, **Rechazar**, **Dudosa** o **Dejar pendiente**. «Aceptar» solo
   confirma una hipótesis para revisión; todavía no crea Ground Truth.
2. Crea un ID local (por ejemplo, `PLAYER-001`) y, si ayuda, un alias. El ID
   es una etiqueta para organizar tracks, no un nombre ni una persona inferida.
   Si el jugador ya tiene una etiqueta, selecciona esa misma.
3. En una hipótesis aceptada, selecciona el ID y pulsa **Confirmar T:x y T:y
   como la misma identidad**. Esta segunda acción explícita crea las
   asignaciones manuales y conserva propuesta, caso, fotogramas, hash y fecha.
   Para sumar más tracks a esa identidad, repite el paso con otras propuestas
   aceptadas y selecciona el mismo ID.
4. El sistema impide asignar como una identidad dos tracks que coaparecen según
   el CSV cargado. También bloquea conflictos entre asignaciones existentes y
   decisiones humanas aceptadas/rechazadas. Si un track ya tenía otra
   identidad, el cambio requiere confirmación y mantiene el historial.
5. **Deshacer última decisión** revierte en orden la última decisión,
   asignación o creación de ID, sin borrar el registro histórico.
   **Historial de cambios humanos** muestra las decisiones y correcciones.

Las decisiones, IDs, asignaciones e historial se guardan automáticamente en
IndexedDB con la clave SHA-256 del video original. Para recuperar en el mismo
perfil del navegador, vuelve a cargar el original, copia de revisión,
manifiesto, CSV, casos y el mismo archivo de propuestas. Para mover el trabajo
a otro perfil, exporta la revisión completa e impórtala en una sesión vacía,
con los mismos hashes de video y propuestas. La importación no modifica las
anotaciones por fotograma.

**Exportar revisión y Ground Truth (JSON/CSV)** produce
`reid-human-review-and-ground-truth.json` y
`reid-human-review-and-ground-truth.csv`. Ambos contienen los estados de todas
las propuestas (aceptada, rechazada, dudosa, pendiente o no revisada), la
procedencia e historial. Solo aparecen bajo `identities` las etiquetas con
tracks asignados mediante la acción humana explícita, con estado
`validated_manual`, evidencia y hashes. Una etiqueta creada sin tracks y una
propuesta aceptada sin asignación explícita no son Ground Truth. El CSV separa
filas de revisión de filas `asignacion_validada`.

El evaluador usa las parejas de tracks con ID humano confirmado: una pareja es
positiva si sus tracks pertenecen a la misma identidad; la presencia de una
propuesta `candidate_same_player` o `existing_reid_link` es una predicción
positiva. El informe solo es un diagnóstico condicionado a los tracks y
propuestas seleccionados; **no mide generalización ni equivale a un benchmark
independiente**. En particular, el flujo actual crea las asignaciones mediante
propuestas aceptadas, así que los tracks etiquetados no constituyen una muestra
independiente de la clasificación/ranking del sistema. Precision, Recall, F1,
FPR, FNR, matriz de confusión y Top-1/Top-5 se interpretan únicamente en ese
universo condicionado, nunca como calidad del partido completo. Cada métrica
sin denominador se marca no disponible. El informe requiere hashes de video y
propuestas idénticos y marca explícitamente que no se permite afirmar
generalización. Los IDs con prefijo `TEST-ONLY-` se rechazan.

No mezcles estos diagnósticos ReID con otras tareas: precisión/recall de
detección requiere cajas humanas de referencia; HOTA/MOTA/IDF1 requieren
anotaciones MOT y se ejecutan por separado con TrackEval mediante
`matchvision-evaluate-identities`. El evaluador de propuestas ReID no estima
calidad de detección, cajas, cobertura de trayectorias ni cambios de ID MOT.

Para una evaluación de generalización válida hace falta una cohorte de prueba
separada, con identidades asignadas manualmente **sin usar las propuestas ni
sus scores para elegir o confirmar los tracks**, y que quede congelada antes
de medir. La interfaz actual no ofrece aún esa anotación independiente de
tracks: hasta que se prepare ese protocolo y el Ground Truth correspondiente,
no informes estos valores como métricas de rendimiento del ReID.

Para guardar un informe fuera del navegador, especifica una ruta nueva; el
comando no sobrescribe resultados:

```powershell
python -m matchvision.evaluation.reid_metrics_cli `
  --review outputs\identity-review\reid-human-review-and-ground-truth.json `
  --proposals outputs\identity-review\reid-review-proposals-v3.json `
  --output outputs\identity-review\reid-metrics-v1.json
```

## Estado de los recursos reales y primer Ground Truth

Auditoría del 10-11-2026: el recurso disponible es
`data/videos/amateur-football-clip.webm`, 3840×2160 y 29,97 FPS. Se decodifican
958 fotogramas (aprox. 32 s); no es un partido completo. El anotador trabaja
con la copia de revisión de 958 fotogramas, el CSV de trayectorias y los 120
casos priorizados con 246 propuestas. Los casos priorizados son una selección
para reducir trabajo, no una revisión exhaustiva ni cobertura de un partido.

No existe una exportación de revisión humana o Ground Truth real en
`outputs/identity-review`. La identidad `TEST-ONLY-REID-E2E` se usó únicamente
en una sesión E2E temporal, y se eliminó junto con su almacén de prueba. No
debe copiarse a la anotación real.

El generador opcional de propuestas ahora también puede usar embeddings OSNet
o un modo híbrido; están disponibles las instrucciones y el checksum en
[`tracking-reid.md`](tracking-reid.md). Estas señales solo ordenan candidatos:
no crean identidades, no reemplazan la revisión humana y no son una
comparación evaluativa contra Ground Truth. La inferencia técnica sobre el
clip no produjo etiquetas humanas ni métricas de identidad.

Para iniciar el primer conjunto humano, carga en el anotador el video original,
la copia de revisión, su manifiesto, el CSV de trayectorias, los casos finales
y `reid-review-proposals-v3.json`. Empieza por los casos prioritarios y usa
IDs sin inferir nombres o dorsales: `PLAYER-001`, `PLAYER-002`, etc. Revisa
apariciones completas y asigna un ID solo si la continuidad visual está
respaldada; de lo contrario deja la propuesta dudosa o pendiente. Exporta
JSON/CSV a una ubicación nueva, nunca sobre los recursos históricos. Los
resultados actuales no permiten calcular métricas reales de ReID.

## Procedimiento de anotación para reducir errores

1. **Revisa el fragmento completo.** Pulsa «Ir al evento» y avanza/retrocede
   para observar a la persona antes, durante y después del candidato. No
   concluyas que hubo desaparición, cruce u oclusión solo por el score o el
   nombre del caso.
2. **Identifica a la persona por evidencia visual.** Los rectángulos y los ID
   temporales/persistentes son sugerencias del sistema, no verdad de terreno.
   No copies un ID del tracker al campo de identidad humana.
3. **Mantén un ID estable.** Escribe un identificador humano como
   `PLAYER-001` y reutilízalo solo cuando la continuidad de la persona esté
   suficientemente respaldada por el video. La lista desplegable ayuda a
   reutilizar los ID ya introducidos y evitar errores tipográficos.
4. **Corrige las detecciones.** Selecciona una sugerencia para etiquetarla.
   Activa «Dibujar recuadro manual» y arrastra para añadir una persona ausente
   de las detecciones. Selecciona una anotación guardada para corregirla o
   eliminarla.
5. **No adivines.** Marca desconocida si la persona es visible pero no se puede
   identificar. Marca ambigua si no se puede decidir si dos apariciones son la
   misma persona. Ignora recuadros cuyo límite no se pueda fijar de forma
   fiable. Cualquiera de estas marcas excluye el fotograma completo de las
   métricas.
6. **Marca exhaustivo solo tras revisar toda la imagen.** Deben estar anotadas
   todas las personas visibles, incluidas las que no detectó el sistema. Si
   falta alguien o no has inspeccionado el fotograma completo, déjalo como no
   exhaustivo; no equivale a un fotograma vacío.
7. **Guarda y comprueba el estado.** El estado del fotograma muestra cuántas
   anotaciones hay, si la revisión es exhaustiva y cuántos fotogramas se han
   revisado. La interfaz guarda localmente tras cada cambio; confirma el
   mensaje de guardado y descarga periódicamente una copia JSON.
8. **Haz una segunda pasada.** Antes de exportar, recorre los casos otra vez y
   verifica consistencia de los ID, personas omitidas, cambios de identidad y
   límites de caja. No marques un caso como resuelto basándote únicamente en
   las predicciones.

Color de camiseta/equipo, rol (árbitro/jugador) e identidad individual son
atributos distintos. Esta tarea etiqueta únicamente identidad individual.
Un color de camiseta parecido no basta para decidir que dos recuadros
pertenecen a la misma persona.

Las anotaciones se guardan en IndexedDB del perfil actual del navegador,
asociadas al SHA-256 del original; no se suben ni comparten. Para continuar en
el mismo perfil, vuelve a cargar el original, la copia de revisión y el
manifiesto: el anotador recupera el estado local. Para trasladar o respaldar
el trabajo, descarga el JSON y selecciónalo en «Importar anotaciones previas».
No uses «Borrar copia local» salvo que realmente quieras eliminar la copia
local del video actual.

Las anotaciones de cajas por fotograma se exportan con **Descargar anotaciones
JSON**. La revisión de tracks e identidades se exporta por separado con los
controles de Ground Truth descritos arriba. No se ha generado ni prellenado
ninguna identidad humana para el video existente.

## Evaluar baseline y ReID

Ejecuta la evaluación solo después de disponer de anotaciones revisadas y
exportadas:

```powershell
python -m matchvision.evaluation.identity_cli `
  --annotations outputs\identity-review\identity-ground-truth.json `
  --video data\videos\amateur-football-clip.webm `
  --baseline outputs\video-runs\8ec67506-93cc-4f17-b80f-d98faabea23f.txt `
  --reid outputs\video-runs\4052ed58-6c37-4a2c-b62a-920dfe1e700b.txt `
  --visualizations outputs\identity-review\error-frames `
  --output outputs\identity-review\comparison.json
```

El comando no sobrescribe un informe existente y exige que el SHA-256 del
video coincida con el de las anotaciones. Compara baseline y ReID sobre los
mismos fotogramas exhaustivos aptos y los ordena cronológicamente para
TrackEval. El informe incluye HOTA, MOTA, IDF1, ID Precision/Recall, cambios de
ID, fragmentación, diagnósticos de fusiones/divisiones, hashes y exclusiones.
Las visualizaciones opcionales separan errores de baseline y ReID.

## Interpretación y límites

Para medir detección y transiciones de identidad de forma significativa,
anota todos los fotogramas de los fragmentos elegidos, no solo muestras
aisladas. Los no revisados se excluyen y no se consideran vacíos. Con muestras
discontinuas, TrackEval compacta la secuencia temporal: cambios de ID y
fragmentación describen solo transiciones entre muestras y no son comparables
directamente con métricas de video continuo o benchmarks.

`false_association_merge_events` y los diagnósticos de división son análisis
adicionales basados en emparejamiento IoU; no son métricas oficiales de
TrackEval. Las anotaciones MOT tampoco contienen clase, equipo ni rol. Hasta
que se exporte ground truth humano y se ejecute la evaluación, no se debe
afirmar que ReID mejora la precisión de identidad.

## Preservación y reproducibilidad

Los artefactos de revisión e informes nuevos deben guardarse en
`outputs/identity-review/` con nombres nuevos. Los archivos baseline, ReID y
video existentes son entradas de solo lectura. Conserva cada JSON exportado y
anota qué hash de anotaciones se usó en cada informe posterior.
