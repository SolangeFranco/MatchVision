# Continuidad de tracking y ReID por apariencia

## Auditoría del proyecto e implementación

El pipeline existente ofrece los protocolos reemplazables `Detector` y
`MultiObjectTracker`, inferencia YOLO11n, ByteTrack aislado por clase,
procesamiento fotograma a fotograma, exportación MOT, MP4 anotado, TrackEval
y metadatos de ejecución. El baseline está preservado en
`outputs/video-runs/8ec67506-93cc-4f17-b80f-d98faabea23f.*`.

Los ID ByteTrack son temporales y locales al video. No había una galería de
apariencia retenida, asociaciones entre tracklets expirados ni exportación de
trayectorias con ID persistentes. El MOT estándar tampoco conserva clases, así
que no permite reconstruir conteos de jugadores o identidades de equipo. Este
video no tiene ground truth.

La nueva capa es aditiva: no cambia detecciones YOLO ni ByteTrack.
`IdentityAssigner` consume observaciones y recortes de los fotogramas fuente y
propone ID persistentes locales al video. Se conserva el formato de ejecución;
el modo ReID opcional añade CSV de trayectorias y JSON de indicadores
descriptivos.

## Alternativas de tracking y ReID

| Método | Señales de asociación | Ventajas | Costos para este proyecto |
|---|---|---|---|
| ByteTrack | Movimiento/IoU y asociaciones de detecciones de confianza alta/baja | Baseline existente de bajo costo; integración sencilla con el detector | No usa embeddings de apariencia para reconectar un ID temporal nuevo con uno anterior |
| BoT-SORT | Asociación tipo ByteTrack, compensación opcional del movimiento de cámara y ReID | Alternativa disponible en Ultralytics; la compensación puede ayudar con cámara móvil | No hay motivo para reemplazar el baseline fijo sin comparación controlada; no infiere identidades entre cortes arbitrarios y ReID necesita un embedding adecuado |
| DeepSORT | Movimiento Kalman y descriptor de apariencia aprendido | Diseño conocido que puede ayudar durante oclusiones breves | La implementación original documenta descriptor MARS de TensorFlow 1.5 y checkpoint separado; revisar runtime legado, procedencia/términos y dominio de personas. El repositorio declara GPL-3.0 |
| OSNet / Torchreid | Embeddings aprendidos y distancia coseno | Modelo compacto de apariencia multiescala con opciones preentrenadas | Los dominios publicados no son este video aéreo; checkpoint, dominio, costo y dependencia requieren revisión; no hay validación específica de fútbol |
| Capa actual de ID persistentes | Histograma HSV + textura/forma HOG, posición predicha, escala, gap y confianza; asignación húngara global | CPU, sin pesos nuevos, transparente/reproducible; puede abstenerse en vez de forzar enlaces débiles | Descriptor artesanal, no embedding semántico; uniformes, recortes pequeños, pose y jugadores parecidos pueden confundirlo. Solo propone asociaciones, no reconoce identidades verificadas |

Referencias principales:

- La [guía de tracking de Ultralytics](https://docs.ultralytics.com/modes/track/)
  documenta ByteTrack, BoT-SORT y su configuración.
- El [repositorio original DeepSORT](https://github.com/nwojke/deep_sort)
  describe el descriptor aprendido, el checkpoint MARS y la licencia GPL-3.0.
- [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) describe
  opciones person-ReID preentrenadas. Su licencia MIT no cubre automáticamente
  pesos ni datos de entrenamiento.
- El [paper OSNet](https://arxiv.org/abs/1905.00953) describe características
  multiescala; sus resultados publicados no corresponden a este clip.

No se descargó un checkpoint/dataset alternativo ni se entrenó. Las
dependencias instaladas OpenCV, NumPy y SciPy bastan para el descriptor y la
asignación. `configs/reidentification.yaml` fija parámetros que quedan
registrados en cada ejecución.

## Comportamiento actual de la asociación

1. ByteTrack mantiene ID temporales independientes por clase.
2. El adaptador extrae histograma HSV de torso y descriptor HOG normalizado de
   cada recorte de persona elegible. El score combina ambas señales; el color
   solo no supera los gates de asociación.
3. Solo considera identidades anteriores dentro del máximo de fotogramas
   configurado y fuera de la ventana de protección de ID activo. Los
   candidatos también deben superar gates de movimiento predicho, escala,
   apariencia, score total y ambigüedad.
4. Una asignación húngara exige correspondencia uno a uno para tracklets que
   comienzan en el mismo fotograma. Ante evidencia débil/ambigua, se crea otro
   ID persistente en lugar de forzar una fusión.
5. Una heurística global de cambio de escena en escala de grises elimina las
   asociaciones temporales activas y bloquea enlaces entre épocas. Se abstiene
   entre planos, sin afirmar identidad entre cámaras.
6. Otras clases reciben ID persistentes locales, pero no se asocian mediante
   ReID. Los ID no son globales entre videos o partidos.

El score no está calibrado y **no es una probabilidad**. El umbral actual es
0.82; aumentó desde 0.76 después de una inspección visual pequeña que encontró
un par dudoso, pero no se calibró con identidades. La clase predeterminada 0 es
COCO `person` e incluye árbitros; no identifica equipos. Detecciones breves o
borrosas pueden no aportar textura suficiente.

## Análisis baseline: indicadores descriptivos

```powershell
python -m matchvision.tracking.analysis_cli `
  outputs\video-runs\8ec67506-93cc-4f17-b80f-d98faabea23f.txt `
  --frame-count 958 `
  --output outputs\tracking-analysis\baseline-byte-track
```

Observado en el MOT guardado de 10 columnas:

- 958 fotogramas decodificados, 4.847 observaciones, 81 ID temporales.
- Observaciones por ID: media 59.84, mediana 17, mínimo 1 y máximo 545.
- 19 ID tienen como máximo 5 observaciones y 33 como máximo 10.
- Media de 5.06 observaciones por fotograma incluyendo todos; mediana 5 en
  fotogramas no vacíos; máximo 11; 42 sin tracks exportados.
- Al dividir observaciones de un ID en cada gap de al menos un fotograma se
  obtienen 431 segmentos de presencia consecutiva (350 más que los 81 ID),
  350 gaps internos y 1.380 posiciones de fotograma ausentes.

Son descripciones de archivos de tracks, no métricas de precisión. Sin
anotaciones no se distinguen fallos del detector, oclusiones, política del
tracker y errores de identidad. No son cambios de ID, fragmentación de
benchmark, MOTA, IDF1, HOTA ni evidencia de que hubiera 81 personas distintas.

El analizador escribe CSV por ID, JSON y PNG de detecciones por fotograma y
distribución de observaciones. No puede inferir clases a partir del MOT
existente.

Se inspeccionaron como recortes ampliados dos propuestas de score alto y tres
de score bajo en `outputs/tracking-analysis/reid-review-examples.png` y
`outputs/tracking-analysis/reid-low-score-review.png`: cuatro parecían
compatibles y una fue inconclusa. Es una muestra pequeña y seleccionada, no
una auditoría ni estimación de falsos merges. La ejecución final conservadora
usa 0.82.

## Comparación reproducible

Procesa el mismo video, checkpoint YOLO, filtro de clases, parámetros
ByteTrack y dispositivo; añade solo la asociación de identidad:

```powershell
matchvision-track-video `
  data\videos\amateur-football-clip.webm `
  --output outputs\video-runs `
  --device cpu `
  --persistent-identities `
  --reid-config configs\reidentification.yaml
```

Cada ejecución crea un ID nuevo (no sobrescribe baseline) y:

- `*_annotated.mp4`: recuadro, clase, score, ID temporal `T`, persistente `P`,
  estado/score y traza de color persistente.
- `*_trajectories.csv`: fotograma, ID temporal/persistente, clase, caja,
  confianza, estado y score de asociación opcional.
- `*_tracking_stats.json`: observaciones por ID, cantidad de ID, tracklets
  unidos, gaps, fotogramas vacíos y eventos ReID.
- Predicción MOT y JSON habituales con parámetros, hashes, versiones y tiempo.
- Nuevas ejecuciones registran segundos y milisegundos medios por fotograma
  para decodificación, detector, tracker, ReID y render/exportación. Es tiempo
  de pared por etapa; el resultado incluye la sincronización que haga el
  adaptador del detector.

Compara `tracking_stats.json` y CSV por track del baseline y ReID. Menos
tracklets o más observaciones por ID son proxies estructurales: una fusión
incorrecta también puede mejorarlos. Revisa el video y etiqueta cruces,
salidas, reapariciones y cortes. Sin ground truth independiente no se puede
afirmar mejora de precisión.

## Rendimiento observado y límites de escalado

El clip disponible tiene 958 fotogramas decodificables, 29,97 FPS, resolución
3840×2160 y duración aproximada de 32 segundos; no existe un video de partido
completo en los recursos auditados. En la ejecución CPU persistente
`4052ed58-6c37-4a2c-b62a-920dfe1e700b`, el metadato registra 315,57 s para 958
fotogramas (3,04 FPS), incluyendo decodificación, detección, tracking, ReID y
escritura/render del video. Esto equivale a unas 9,9 veces más tiempo que la
duración fuente. Una extrapolación lineal a 90 minutos sería alrededor de
14,8 horas en la misma configuración, no una garantía de duración.

Los resultados históricos guardan solo el tiempo total, no desglosan el
detector, tracker, ReID ni la codificación. Las nuevas ejecuciones instrumentan
esas etapas; las propuestas ReID generadas de nuevo también guardan tiempos de
preparación, extracción de evidencias/características y scoring/agrupación. El
JSON v3 histórico no tiene esos campos y no se modifica retroactivamente.
Tampoco se registraron uso/peak de RAM o VRAM; no se puede afirmar uso de GPU
ni identificar el cuello de botella concreto de la ejecución histórica. No se
aplicó una optimización algorítmica sin perfil por etapa y sin Ground Truth.

El pipeline procesa fotogramas secuencialmente y la agregación de estadísticas
de identidad ahora es incremental: ya no conserva un diccionario por cada
observación del video. Los datos de salida y el estado del modelo siguen
requiriendo recursos según la duración, resolución y número de tracks.
ByteTrack mantiene internamente el estado Kalman/listas de tracks y la capa
ReID mantiene galerías, velocidades, escenas y asignaciones; ninguno expone un
formato de estado versionado. `*_tracking_stats.json` y `*_trajectories.csv`
dependen de activar la capa de identidad persistente (`--persistent-identities`):
resumen asignaciones, tracklets, huecos y observaciones persistentes. Sin esa
capa, el pipeline general de detección/tracking sigue funcionando, pero ambos
campos de ruta son `null` y no se crea un archivo de estadísticas de identidad.

Para evitar repetir inferencia YOLO tras una interrupción, se puede indicar
`--checkpoint-dir outputs\video-checkpoints\partido-01`. Se guardan segmentos
atómicos de detecciones, vinculados por hashes al video, configuración y pesos
del detector. Al reanudar, el pipeline vuelve a leer el video desde el inicio
y reproduce las detecciones almacenadas para reconstruir ByteTrack y ReID; no
serializa sus estados mutables ni continúa archivos de salida parciales. La
carpeta debe conservarse y no reutilizarse con otra configuración. Esta caché
no prueba rendimiento de 90 minutos ni permite continuar desde un fotograma
intermedio sin reproducir el tracking previo.

## Límites actuales

- El clip es una toma aérea amateur de 32 segundos, no una transmisión.
- Sin ground truth etiquetado no se cuantifican cambios de ID,
  reidentificaciones correctas, fusiones falsas, HOTA, MOTA o IDF1.
- El detector de cambio de plano inicia ID nuevos entre cortes; no resuelve
  ReID entre cámaras de una transmisión.
- HOG/HSV no son embeddings aprendidos y pueden fallar con recortes pequeños,
  uniformes, cambios de iluminación o jugadores parecidos.
- COCO `person` incluye árbitros; la asignación de equipo/rol no forma parte de
  este módulo ReID.
- Los descriptores pueden reducir los FPS en CPU; mide cada ejecución en vez
  de estimarlo desde ByteTrack.

## Comparación opcional con OSNet-AIN

Se conserva `hsv_hog` como método predeterminado y referencia. El extractor
opcional usa `osnet_ain_x1_0` de Torchreid con pesos preentrenados en MSMT17:
2,17 millones de parámetros, entrada 256×128 y descriptor de 512 valores.
El modelo AIN usa normalización por instancia para mejorar la robustez de
apariencia, pero MSMT17 no representa la cámara aérea ni el dominio de este
clip. La similitud coseno no es una probabilidad ni acredita una identidad.

Instala el extra local y descarga el checkpoint fijado, que se valida por
tamaño y SHA-256 antes de guardarse en `models/`:

```powershell
python -m pip install -e ".[reid]"
matchvision-download-reid-model
```

El código de [Torchreid](https://github.com/KaiyangZhou/deep-person-reid) y
la implementación OSNet está bajo MIT; el [paper OSNet](https://arxiv.org/abs/1905.00953)
describe la extracción multiescala. Los pesos se obtienen del
[repositorio kaiyangzhou/osnet en Hugging Face](https://huggingface.co/kaiyangzhou/osnet);
el checkpoint fijado tiene
SHA-256 `8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f`.
La tarjeta del modelo declara MIT para los pesos. El checkpoint y el dataset
no se redistribuyen con el código. Revisa las condiciones del dataset MSMT17
y la licencia/atribución de los pesos antes de uso comercial; la licencia del
código no resuelve los derechos del dataset.
El extractor usa `torch.load(..., weights_only=True)`, BGR→RGB, redimensionado
256×128, normalización ImageNet y normalización L2 del embedding. `auto`
selecciona CUDA, MPS o CPU; un dispositivo explícito no disponible produce
error.

Para comparar en el mismo video y la misma configuración de detección/tracking:

```powershell
matchvision-track-video data\videos\amateur-football-clip.webm `
  --output outputs\video-runs `
  --device cpu `
  --persistent-identities `
  --reid-method hybrid `
  --reid-config configs\reidentification.yaml
```

Los métodos configurables son:

| Método | Señal visual | Lectura |
|---|---|---|
| `hsv_hog` | 0,5 × intersección HSV + 0,5 × coseno HOG, ambos acotados a [0,1] | Baseline preservado; mantiene los gates HSV/HOG existentes |
| `embedding` | Media de hasta las tres similitudes coseno más altas de la galería; transformación `(coseno + 1) / 2` a [0,1] | OSNet-AIN; comparación de apariencia, no probabilidad |
| `hybrid` | `embedding_weight` × señal OSNet + (1 − `embedding_weight`) × señal HSV/HOG | Fusión explícita de señales transformadas a [0,1] antes de aplicar el peso |

Los tres métodos usan además los términos existentes de movimiento predicho,
escala y confianza del detector, gates temporales y asignación Húngara uno-a-uno.
Por ello, un score final de una variante no debe compararse como si fuera una
probabilidad con el de otra. Los umbrales son iniciales y no están calibrados
con Ground Truth. La galería guarda como máximo cinco embeddings por identidad
y muestrea tracks activos cada diez fotogramas por defecto; los tracks nuevos
se codifican inmediatamente. Ajusta `embedding_*`, `embedding_weight` y
`minimum_visual_similarity` en YAML, o elige el método con `--reid-method`.

El asistente de revisión también admite `--reid-method hybrid` y escribe
generaciones v4 sin modificar los JSON históricos v3. Cada propuesta incluye
scores separados `hsv_hog`, `embedding` e `hybrid` cuando se dispone del
checkpoint; el método seleccionado solo controla el orden. El anotador local
muestra las tres puntuaciones como heurísticas y mantiene intactas las
decisiones humanas y Ground Truth.

La extracción se hace en lotes acotados en memoria, pero ByteTrack y la capa
de identidad mantienen estado solo en memoria y el pipeline todavía no guarda
checkpoints reanudables. Una interrupción requiere iniciar una ejecución nueva;
el resultado parcial no se trata como salida completa. Los fotogramas se leen
secuencialmente y no se acumula el video en RAM, pero no se ha probado un
partido de 90 minutos ni se afirma rendimiento de partido completo. Tampoco se
registran aún picos de RAM/VRAM.

La comparación de HSV/HOG, embeddings y modo híbrido sobre un mismo clip es
solo una prueba técnica si no hay Ground Truth independiente. Para evaluar
calidad, fija umbrales y pesos en entrenamiento/validación separados por
partido; reserva partidos completos e identidades no vistos para prueba. No
dividas apariciones casi idénticas del mismo partido entre entrenamiento y
prueba, no uses propuestas automáticas para crear la referencia y congela el
Ground Truth antes de medir. En la actualidad no hay identidades humanas
verificadas, así que no hay resultados válidos de mejora, mAP o Top-k.

### Primera ejecución híbrida en el clip local

La corrida `09664dcb-e536-4644-830a-caa914bd0aae` completó los 958 fotogramas
del video a 3840×2160 con YOLO11n + ByteTrack y OSNet en CPU. Se conservaron
los resultados históricos; esta ejecución escribió archivos nuevos bajo
`outputs/video-runs/`. Fue una comparación operativa con la corrida HSV/HOG
`4052ed58-6c37-4a2c-b62a-920dfe1e700b`: ambos tienen el mismo hash de entrada,
dispositivo CPU, detector/confianza/tamaño y configuración/versionado de
ByteTrack.

| Corrida | Método | Tiempo total | Throughput | Tiempo ReID |
|---|---|---:|---:|---:|
| `4052ed58-6c37-4a2c-b62a-920dfe1e700b` | HSV/HOG | 315,57 s | 3,04 FPS | No desglosado históricamente |
| `5620a07c-0f95-4b56-83ec-a92c1d9ad206` | Embedding | 549,90 s | 1,74 FPS | 179,92 s |
| `09664dcb-e536-4644-830a-caa914bd0aae` | Híbrido | 649,98 s | 1,47 FPS | 244,84 s |

Las tres corridas procesaron 958 fotogramas y produjeron 81 tracks temporales.
HSV/HOG generó 66 ID persistentes y 15 enlaces; embedding, 44 ID y 37 enlaces;
híbrido, 42 ID y 39 enlaces. Son conteos descriptivos, no mediciones de
identidad correcta: todas las métricas de evaluación quedaron `null` por
ausencia de Ground Truth. No se puede concluir que embedding o híbrido sean
más precisos. Las corridas profundas tardaron 1,74× y 2,06× el baseline,
respectivamente, en la misma máquina.

En la corrida híbrida, detección tomó 250,49 s, tracking 2,77 s, decodificación
30,22 s y render/exportación 121,56 s. Esta corrida no separó el tiempo de
extracción de embeddings, HSV/HOG y asociación; las nuevas ejecuciones
incluyen esos subtotales en `timings.reidentification_detail` (extracción,
inferencia OSNet, HSV/HOG y asociación). No se midieron RAM ni VRAM; CUDA no
está disponible y los tres métodos corrieron en CPU. No se extrapola a partidos
de 90 minutos.
