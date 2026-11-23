# MatchVision

Análisis de video de fútbol a partir de una cámara, sin sensores corporales.
El proyecto avanza por etapas y conserva los resultados experimentales
anteriores. El baseline actual ejecuta detección y tracking fotograma a
fotograma; la asociación de identidades persistentes y la clasificación de
equipos son estimaciones experimentales, no resultados validados contra
etiquetas humanas.

## Arquitectura

```text
Video
  -> segmentación de escenas/planos (pendiente)
  -> detección de jugadores, árbitros, porteros y balón
  -> tracking multiobjeto por plano
  -> asociación opcional de apariencia/movimiento (ReID)
  -> clasificación visual de equipo
  -> calibración de cancha
  -> análisis espacial, visualizaciones y evaluación
```

Los tracks y las identidades de partido son conceptos distintos. El tracker
mantiene ID locales durante una secuencia; la asociación ReID propone
continuidad con evidencia visual y de movimiento, pero todavía no se ha
validado con ground truth humano. El color de camiseta solo estima equipo; no
identifica a una persona. Ante ambigüedad, el sistema debe abstenerse.

## Datos y evaluación de modelos

**Benchmark MOT principal: SoccerNet-Tracking.** Es un benchmark de tracking
de fútbol con personas y tracks de balón. Sus exportaciones MOT no incluyen
clase ni equipo, y el protocolo no valida por sí mismo el ReID entre planos,
la clasificación de equipos ni AP de balón por clase.

Datasets complementarios:

- [SoccerNet Re-Identification](https://github.com/SoccerNet/sn-reid) para
  embeddings y correspondencias de apariencia. Las identidades están
  delimitadas a acciones, no a partidos enteros.
- [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate)
  para localización de jugadores y atributos de equipo/posición.
- [SportsMOT](https://github.com/MCG-NJU/SportsMOT) para tracking deportivo
  en clips cortos sin cortes.
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) para HOTA, MOTA,
  IDF1 y métricas relacionadas.

Las limitaciones de anotación, acceso y licencia, y el protocolo reproducible
se describen en [docs/model-evaluation.md](docs/model-evaluation.md). La
matriz de experimentos está en [configs/benchmark.yaml](configs/benchmark.yaml).
El benchmark local contra Ground Truth congelado y la evaluación independiente
de apariencias se ejecutan con `matchvision-benchmark` y
`matchvision-evaluate-reid-retrieval`; el protocolo está en
[docs/model-evaluation.md](docs/model-evaluation.md#evaluacion-local-independiente).
No se deben usar etiquetas ReID limitadas a una acción como prueba de identidad
persistente durante un partido.

## Hitos implementados

- Lectura de metadatos y fotogramas, parser MOT y pruebas.
- Validación de manifiestos de datasets, separación de splits y evaluación
  TrackEval de HOTA/MOTA/IDF1.
- Pipeline local YOLO + ByteTrack con exportación MOT, MP4 y JSON de ejecución,
  y evaluación opcional contra ground truth.
- Capa CPU opcional de propuesta de identidades persistentes, exportación de
  video/CSV y estadísticas descriptivas. No es reconocimiento de identidad
  validado.
- ReID opcional OSNet-AIN con embeddings de 512 dimensiones, galería acotada y
  modos `hsv_hog`, `embedding` e `hybrid`; HSV/HOG sigue siendo el valor
  predeterminado y ningún score constituye una identidad confirmada.
- Selector determinista de 120 casos prioritarios para revisión humana,
  conversión de video compatible con navegador y anotador local.
- Generador de evidencia visual ReID que extrae fotogramas, compara
  tracklets mediante HSV/HOG y presenta candidatos sin crear identidades
  humanas ni anotaciones.
- Agrupamiento no supervisado de dos equipos por histogramas de camiseta,
  con abstención y correcciones manuales explícitas.
- CLI independiente de detección/seguimiento temporal de balón, con
  predicciones durante oclusiones breves, recuperación conservadora,
  exportación CSV y manifiesto con hashes.
- API base de homografía manual RANSAC para proyectar puntos de imagen al plano
  de cancha declarado; aún no detecta referencias automáticamente.

El baseline YOLO11n + ByteTrack se ejecutó sobre un clip local de fútbol
amateur de 958 fotogramas. La configuración, hashes y resultados están en
[docs/experiment-run-2026-11-08.md](docs/experiment-run-2026-11-08.md). El
experimento más reciente de ReID está en
[docs/experiment-tracking-reid-2026-11-09.md](docs/experiment-tracking-reid-2026-11-09.md).
No hay ground truth de identidad disponible, por lo que no se afirma que el
ReID mejore la precisión.

## Validación manual de identidades

La herramienta local y el procedimiento detallado están en
[docs/identity-validation.md](docs/identity-validation.md). El anotador no
infiere etiquetas humanas: los 120 casos son candidatos y el usuario debe
revisar las apariciones, anotar personas omitidas, mantener ID estables y
marcar como ambiguas las identidades dudosas.

Desde la raíz del proyecto:

```powershell
python -m http.server 8765 --bind 127.0.0.1
```

Abre
`http://127.0.0.1:8765/tools/annotation/reid_annotator.html` y sigue el orden
de carga indicado en la guía. Las anotaciones se guardan en IndexedDB local
vinculadas al hash del video y se pueden respaldar como JSON. El archivo JSON
de identidades debe ser creado por revisión humana.

Para generar, sin sobrescribir resultados existentes, fotogramas comparativos
y sugerencias heurísticas para los casos:

```powershell
python -m matchvision.reidentification.review_assistant
```

Por defecto crea propuestas nuevas HSV/HOG en
`outputs/identity-review/reid-review-proposals-v4.json` y evidencias en
`outputs/identity-review/reid-review-evidence-v4/`; no sobrescribe la
generación histórica v3. Para ordenar candidatos con OSNet híbrido, instala
`.[reid]`, descarga los pesos verificados y ejecuta:

```powershell
python -m matchvision.reidentification.review_assistant --reid-method hybrid
```

La generación v4 puede mostrar en el anotador scores HSV/HOG, embedding y
híbrido por separado; solo el método seleccionado ordena los candidatos. Los
scores no son probabilidades, no son Ground Truth y no crean identidades.
Carga el JSON nuevo en «Apoyo visual ReID» después de verificar los videos.

El anotador permite aceptar/rechazar cada propuesta como decisión de revisión,
marcarla como dudosa o dejarla explícitamente pendiente. Guarda los cambios en
IndexedDB local asociado al hash del video y recupera decisiones, identidades
y asignaciones al volver a cargarlo en el mismo perfil del navegador. Se puede
deshacer la última acción humana y conservar el historial.

La revisión de propuestas y la anotación de Ground Truth son flujos separados.
Aceptar o rechazar una propuesta no crea una etiqueta humana para Ground Truth.

La herramienta exporta la revisión de propuestas por separado. Para generar
Ground Truth independiente, carga únicamente el video original: no hace falta
cargar predicciones, casos priorizados ni propuestas ReID. Revisa cada
fotograma que vayas a incluir, dibuja los recuadros manualmente y asigna un ID
humano estable. El mismo jugador debe conservar ese ID entre fotogramas; los
IDs del tracker y las propuestas nunca se copian automáticamente. Selecciona
el tipo de objeto (jugador, portero, árbitro, balón u otro), corrige o elimina
anotaciones cuando sea necesario y registra manualmente apariciones,
desapariciones y oclusiones. Marca el fotograma como exhaustivo solo después
de revisar toda la imagen. Si no puedes resolver una identidad, indícala como
desconocida/ambigua o exclúyela; no la adivines. El progreso se guarda en
IndexedDB local asociado al hash del video y puede exportarse e importarse como
JSON.

Antes de consultar resultados de modelos, usa **Congelar conjunto de prueba**.
Solo se permite congelar un conjunto con al menos un fotograma exhaustivo y
con identidades resueltas; la interfaz lo deja de solo lectura y exporta un
JSON con SHA-256 del contenido. También puede congelarse una exportación con
el comando siguiente, que valida el formato y se niega a sobrescribir la
salida. Conserva el JSON congelado separado de las predicciones. La
congelación protege la integridad del archivo, pero no sustituye la revisión
humana ni demuestra por sí misma que las etiquetas sean correctas.

```powershell
matchvision-freeze-identity-set `
  --input anotaciones.json `
  --output conjunto-prueba.json
```

Cuando haya suficientes asignaciones manuales, evalúa las propuestas sin
sobrescribir informes previos:

```powershell
python -m matchvision.evaluation.reid_metrics_cli `
  --review outputs\identity-review\reid-human-review-and-ground-truth.json `
  --proposals outputs\identity-review\reid-review-proposals-v3.json `
  --output outputs\identity-review\reid-metrics-v1.json
```

El evaluador de propuestas existente informa Precision, Recall, F1, FPR, FNR,
Top-1, Top-5 y matriz de confusión cuando sus denominadores están disponibles.
Sigue siendo un diagnóstico condicionado a propuestas, no una evaluación
independiente. La comparación de trayectorias MOT mediante
`matchvision-evaluate-identities` es un flujo separado. La evaluación
independiente de detección, tracking y ReID contra el nuevo Ground Truth
congelado es una etapa posterior; todavía no se generaron etiquetas humanas ni
métricas reales para el video existente.

## Clasificación visual de equipos

El módulo HSV agrupa apariciones de tracks por apariencia de camiseta, sin
fusionar identidades individuales. Devuelve `assigned`, `ambiguous`,
`outlier`, `insufficient_samples` o `insufficient_data`; las asignaciones
manuales se aplican mediante un archivo JSON de correcciones. No identifica
árbitros ni sabe qué equipo es local/visitante. Ver
[docs/team-classification.md](docs/team-classification.md).

```powershell
python -m matchvision.teams.cli `
  --video data\videos\amateur-football-clip.webm `
  --trajectories outputs\video-runs\<run-id>_trajectories.csv `
  --identity-column persistent_id `
  --output outputs\team-classification\<run-id>_team_assignments.csv
```

El módulo procesa tracks y video existentes; no vuelve a ejecutar detección ni
tracking. La salida incluye los hashes de entrada y no sobrescribe resultados.
La primera pasada sobre el clip local asignó automáticamente 26 de 66 ID (23 a
`equipo_a`, 3 a `equipo_b`) y dejó el resto ambiguo, atípico o sin muestras
suficientes. El desequilibrio requiere revisión manual y no es evidencia de
precisión; los detalles están en
[docs/team-classification.md](docs/team-classification.md).

## Instalación

Se requiere Python 3.10 o posterior.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,vision]"
```

La configuración del workspace apunta Pylance a
`.venv\Scripts\python.exe`. Si VS Code conserva otro entorno, ejecuta
**Python: Select Interpreter**, elige el entorno del proyecto y abre una
terminal nueva.

Ejecuta detección y tracking sobre un video local:

```powershell
matchvision-track-video path\to\match.mp4 --output outputs\video-runs
```

El modelo COCO `yolo11n.pt` detecta `person` y `sports ball`; no distingue
jugadores de árbitros, no clasifica equipos y no garantiza detectar el balón
pequeño y rápido. Consulta [docs/detection-tracking.md](docs/detection-tracking.md)
antes de elegir un detector específico de fútbol. Los pesos no se versionan
en el repositorio.

Para activar la propuesta opcional de identidades:

```powershell
matchvision-track-video path\to\match.mp4 `
  --output outputs\video-runs `
  --device cpu `
  --persistent-identities `
  --reid-config configs\reidentification.yaml
```

El pipeline exporta MOT, video anotado y JSON de ejecución; con ReID también
exporta trayectorias y estadísticas descriptivas.

### ReID con OSNet-AIN

Instala el extra opcional, descarga una vez los pesos fijados y ejecuta el modo
híbrido (los modos y parámetros quedan registrados por separado en cada
ejecución):

```powershell
python -m pip install -e ".[reid]"
matchvision-download-reid-model
matchvision-track-video data\videos\amateur-football-clip.webm `
  --output outputs\video-runs `
  --device cpu `
  --persistent-identities `
  --reid-method hybrid `
  --reid-config configs\reidentification.yaml
```

El extractor selecciona CUDA/MPS/CPU mediante `embedding_device: auto`, procesa
recortes en lotes acotados y conserva hasta cinco muestras por identidad. El
modelo es OSNet-AIN x1.0 preentrenado en MSMT17; sus pesos se verifican por
SHA-256, no se incluyen en Git y tienen términos propios que deben revisarse
para uso comercial. Los scores son señales heurísticas, no probabilidades.
Consulta [docs/tracking-reid.md](docs/tracking-reid.md) para fórmulas,
licencias, límites y protocolo de evaluación.

La ejecución técnica del clip disponible procesa 958 fotogramas de 3840×2160
en CPU con el método híbrido. No hay un partido completo ni Ground Truth humano;
por tanto no se declaran métricas de precisión ni mejora de identificación.
La lectura es secuencial y la galería está limitada, pero checkpoints de
reanudación del estado de ByteTrack/ReID todavía no están implementados. No se
ha probado procesamiento de 90 minutos.

La herramienta de conversión de video de revisión necesita PyAV, instalado
mediante el extra opcional `annotation`:

```powershell
python -m pip install -e ".[annotation]"
```

## Seguimiento de balón

El comando `matchvision-track-ball` procesa el balón por separado del tracking
de jugadores. Utiliza el protocolo `Detector`, admite checkpoints
especializados con procedencia declarada y permite especificar IDs de clase.
La salida distingue detecciones del modelo, posiciones predichas durante
oclusiones breves y fotogramas sin detección. Las predicciones no son
mediciones; las coordenadas permanecen en píxeles de imagen.

```powershell
matchvision-track-ball data\videos\amateur-football-clip.webm `
  --output outputs\ball-tracking `
  --model yolo11n.pt `
  --device cpu `
  --confidence 0.05
```

El modelo COCO es solo un baseline genérico: no se afirma que localice de forma
fiable un balón pequeño o borroso. La implementación temporal no sustituye un
detector entrenado para fútbol, ni usa optical flow o embeddings. Ver
[docs/ball-tracking.md](docs/ball-tracking.md) para el protocolo, las
limitaciones, licencias y comandos para un checkpoint personalizado.

Evalúa un MOT sin ground truth (solo genera indicadores descriptivos):

```powershell
matchvision-analyze-tracking path\to\predictions.txt `
  --frame-count 958 `
  --output outputs\tracking-analysis\run-name
```

Ejecuta pruebas:

```powershell
python -m pytest
```

Valida la configuración del benchmark:

```powershell
matchvision-validate-config --config configs/benchmark.yaml
```

El evaluador recibe archivos de predicciones ya generados; no descarga datos ni
ejecuta inferencia. Consulta [docs/experiment-runbook.md](docs/experiment-runbook.md)
para los manifiestos y comandos reproducibles.

Los videos y datasets, anotaciones y resultados generados deben guardarse bajo
`data/` y `outputs/`; los pesos deben quedar en `models/`. Estas rutas y los
archivos de video/checkpoint se excluyen de Git para evitar publicar medios,
datos personales o binarios pesados. Se conservan el código, pruebas,
documentación, configuraciones y manifiestos de procedencia que no contengan
rutas o datos privados. Registra versión y checksum de cada modelo.

## Próximas etapas

1. Completar la anotación humana y evaluar ReID contra el mismo ground truth.
2. Evaluar la clasificación de equipos sobre video real con correcciones
   humanas registradas.
3. Comparar detectores y trackers en las mismas secuencias y hardware.
4. Incorporar segmentación de cancha e integrar la homografía manual existente
   con correspondencias verificadas, sin afirmar distancias reales sin escala
   física validada.
5. Obtener datos y pesos autorizados para evaluar/fine-tunear el detector de
   balón; añadir compensación de movimiento y visualización táctica 2D.
6. Crear interfaz de puntos de cancha y validar la homografía manual con
   referencias reales; después abordar estimación de líneas/cámara móvil.
7. Diseñar métricas espaciales/físicas que distingan mediciones, estimaciones
   e inferencias y validar en partidos no vistos.
8. Documentar licencias y optimizar CPU/GPU.

La primera API de calibración manual se documenta en
[docs/pitch-calibration.md](docs/pitch-calibration.md); sus ejemplos de puntos
son ilustrativos y no constituyen una calibración del video disponible.
