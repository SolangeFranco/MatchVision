# Plan de evaluación de datasets y modelos

Estado: diseño de investigación y experimentos. No se descargó ningún dataset,
no se entrenó ni evaluó un modelo y este documento no informa resultados
locales de benchmark. Las fuentes y condiciones de acceso se revisaron el
2026-11-06; confirma la release y sus términos cuando solicites acceso.

## Decisión resumida

Usar **SoccerNet-Tracking** como benchmark principal para la primera
comparación de detección y MOT dentro de una secuencia. Es específico de fútbol,
proporciona videos originales, cajas/ID y objetos que incluyen el balón, y
dispone de protocolo basado en HOTA.

No cubre todo el producto. Las anotaciones MOT estándar no identifican clase o
equipo, y los clips separados no documentan identidades enlazadas a lo largo
del partido. Complementar con:

1. **SoccerNet ReID** para recuperación por apariencia y correspondencias de
   acción/repetición.
2. **SoccerNet Game State Reconstruction (GSR)** para rol, lado de equipo,
   número de camiseta y cancha.
3. **SportsMOT** como comprobación adicional de dominio MOT deportivo, sujeto
   a sus términos no comerciales.

Ninguno de estos datasets demuestra persistencia global de identidad entre
cortes arbitrarios a lo largo de un partido.

## Comparación de datasets

| Dataset | Video/anotaciones | Qué puede evaluar | Brecha importante | Acceso y términos |
|---|---|---|---|---|
| [SoccerNet-Tracking](https://github.com/SoccerNet/sn-tracking) | Descripción oficial: 12 partidos completos, 200 clips anotados de 30 s y una mitad completamente anotada. CSV estilo MOT: fotograma, ID, caja, ancho/alto, confianza y campos de compatibilidad. | Detección, asociación y tracking de objetos de una secuencia; incluye jugadores, árbitros y balón. Usa HOTA/TrackEval. | La tarea oficial no considera clase y el CSV no tiene atributo de equipo. No documenta ID que unan clips/cámaras. No infieras GT entre planos a partir de ID de secuencias independientes. | El kit indica `pip install SoccerNet` y `SoccerNetDownloader.downloadDataTask(task="tracking", split=[...])`. SoccerNet sirve video protegido mediante Hugging Face; requiere cuenta, aprobación y NDA para datos restringidos. Verifica versión y acceso antes de descargar. No redistribuir. El repositorio tracking no muestra licencia de código clara; pide aclaración antes de reutilizar código más allá de inspección. |
| [SoccerNet ReID](https://github.com/SoccerNet/sn-reid) | Recortes de jugadores y correspondencias/identidades de vistas de acción y repetición; el README del challenge informa 340.993 miniaturas de 400 partidos. | Embeddings/recuperación por apariencia entre vistas relacionadas con una acción, con su protocolo mAP/rank-1. | Las etiquetas son por acción; no identifican a una persona entre acciones independientes o en todo el partido. No es ground truth de identidad global y los recortes no son anotaciones de detector full-frame. | Acceso por flujo Hugging Face/NDA de SoccerNet. El kit publica código MIT, pero eso no licencia videos/dataset. |
| [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate) | Clips y anotaciones JSON por frame/detección, con ID y atributos de rol, número, lado del equipo y coordenadas de cancha. El proyecto describe 200 clips; usar release v1.3 o posterior según solicita el repo. | Localización, rol/equipo/número y cancha con protocolo GS-HOTA y kit oficial. | No sustituye enlaces de identidad global entre cámaras. Equipo es lado relativo a cámara, no club canónico; no es tarea de tracking de balón. | Descarga oficial de SoccerNet/Hugging Face; videos protegidos requieren aprobación/NDA. Código del repositorio GPL-3.0; derechos de video/dataset son independientes. |
| [SportsMOT](https://github.com/MCG-NJU/SportsMOT) | 240 clips de básquet, voleibol y fútbol; anotaciones MOTChallenge por imagen/caja/ID. El README indica media de 485 frames y clips cortados para evitar cambios de plano. | Comparación adicional de tracking de jugadores y dominio deportivo; publica splits y scripts de evaluación/TrackEval. | Tarea de jugadores que excluye espectadores, árbitros y entrenadores; sin balón ni equipo. No evalúa ReID entre planos. | Requiere unirse a la competencia CodaLab. El README declara CC BY-NC 4.0 y prohíbe redistribuir. Tratar uso no comercial como restricción; confirmar términos. |

### Interpretación correcta de las anotaciones

- SoccerNet-Tracking trata los objetivos como conjunto de objetos; el CSV MOT
  no separa clases para métricas específicas de jugador, árbitro o balón.
  Aun así permite evaluar cajas/ID según el benchmark oficial.
- Un ID de secuencia no es identidad global de persona. Hasta confirmar la
  semántica exacta de una release, asumir que solo es válido dentro de ella.
- SoccerNet ReID evalúa vistas relacionadas con una acción/repetición; sus
  etiquetas no persisten entre acciones.
- GSR aporta rol/equipo/número/cancha en otra tarea, pero no agrega enlaces
  multi-cámara a SoccerNet-Tracking.
- Para precisión/recall/AP de balón y detección por clase se necesitan clases
  verificadas en la release o un subset con licencia documentada. No uses las
  propias predicciones del modelo como ground truth.

## Alternativas de detector

La primera comparación usa pesos genéricos públicos e inferencia; mide
transferencia a fútbol, no el sistema futuro ajustado. Mapear COCO
(`person`, `sports ball`) al esquema del proyecto no significa que `person`
distinga jugadores y árbitros.

| Familia | Motivo para incluirla | Costos y límites |
|---|---|---|
| YOLO (Ultralytics) | Detector one-stage conocido, referencia práctica de velocidad/precisión y sencilla inferencia de video. | Pesos COCO no son específicos de fútbol. Jugadores lejanos y balón quizá requieran fine-tuning, alta resolución o tiles. Código AGPL-3.0 con alternativa comercial del proveedor; revisar peso/licencia y despliegue. |
| RT-DETR (PaddleDetection) | Detector Transformer en tiempo real; alternativa arquitectónica a YOLO con opciones COCO. | Medir throughput, memoria y recall de balón pequeño en hardware real; velocidad publicada depende de hardware/runtime. Código Apache-2.0; revisar cada checkpoint por separado. |
| Faster R-CNN (Torchvision) | Baseline COCO de dos etapas conocido para medir si el recall justifica latencia. | Menos adecuado para tiempo real/partido completo; persisten límites de objetos pequeños y clases. Código BSD; revisar dependencias y pesos. |

No ordenes modelos usando FPS/AP de proveedores en dispositivos/datasets
distintos. Fija runtime, frames y resolución, y reporta inferencia y throughput
end-to-end por separado.

## Alternativas de tracker

Todos reciben **las mismas detecciones cacheadas** por ejecución para aislar
asociación de cambios del detector.

| Tracker | Rol | Límites |
|---|---|---|
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | Baseline rápido de movimiento/IoU; asociación de detecciones de confianza baja relevante ante pérdidas/oclusiones. | Sin ReID de apariencia por defecto; no resuelve desapariciones largas/cortes. Código MIT. |
| [BoT-SORT](https://github.com/NirAharon/BoT-SORT) | Comparar compensación del movimiento de cámara y ReID opcional contra ByteTrack. Separar configuración sin ReID y con ReID fijo. | Compensación para movimiento dentro de una vista, no para establecer identidad entre planos de transmisión no relacionados. Código MIT. |
| [MixSort](https://github.com/MCG-NJU/MixSort) | Candidato de asociación deportiva, probado por autores en SportsMOT. | Sus resultados publicados no son resultados del proyecto ni prueba de ReID global; verificar licencias y dependencias antes. |

SoccerNet-Tracking publica HOTA 75.61 para ganador de 2023 y 42.38 para su
baseline. Son **resultados publicados del benchmark**, no una comparación
directa YOLO/ByteTrack/BoT-SORT ni resultados medidos aquí. MixSort también
publica resultados SportsMOT de sus autores; no compararlos numéricamente con
SoccerNet.

## Alternativas deportivas de ReID

1. **Protocolo SoccerNet ReID**: opción inicial de dominio cercano; entrenar o
   evaluar embeddings con splits oficiales e informar métricas por acción.
   Respetar semántica de ID y restricciones.
2. **OSNet**: baseline ligero de persona (por ejemplo, con Torchreid), útil
   para embeddings de tracklet, pero puede enfocarse demasiado en ropa/fondo
   genéricos y confundir compañeros.
3. **Transformer ReID (p. ej. TransReID)**: mayor capacidad para probar señales
   finas; más cómputo y sin asumir generalización fuera de dominio.
4. **MixSort**: método de asociación MOT, no benchmark de identidad global de
   partido.
5. **OCR de dorsal + atributos de equipo/camiseta**: evidencia independiente
   cuando el dorsal es legible. OCR suele fallar en resolución de transmisión;
   color de equipo solo reduce candidatos y no distingue compañeros.

Para cualquier experimento entre planos, extrae tracklets/embeddings, calibra
scores y genera enlaces candidatos con confianza. Mide recuperación solo en
pares acción/repetición con etiquetas y conserva el resultado
desconocido/abstención. Es una prueba local, no identidad global. Para evaluar
un partido se requieren enlaces ground truth entre planos no relacionados.

## Licencias, acceso y procedencia

- **Acceso no equivale a licencia.** Los videos SoccerNet están protegidos:
  solicitar acceso en Hugging Face, aceptar NDA y esperar aprobación. La
  licencia del código no otorga derechos sobre el video.
- **SportsMOT** declara CC BY-NC 4.0 y prohíbe redistribución; no es opción
  predeterminada para uso comercial.
- Licencias de código consultadas: ByteTrack MIT; BoT-SORT MIT; kit SoccerNet
  ReID MIT; repositorio GSR GPL-3.0; Ultralytics AGPL-3.0 con alternativa
  comercial; PaddleDetection Apache-2.0. En la revisión no se halló licencia
  explícita en SoccerNet-Tracking. Revisar submódulos, pesos, checkpoints y
  términos vigentes antes de distribuir/comercializar.
- **Descarga**: no espejar ni versionar medios. SoccerNet mediante paquete
  oficial/Hugging Face tras aprobación; SportsMOT vía CodaLab y términos
  aceptados. Confirmar release, split, versión de anotación y uso permitido.
- Antes de bajar pesos, registrar modelo/checkpoint, URL, versión/commit,
  SHA-256, licencia, datos de preentrenamiento, inferencia y uso previsto.

Esto es un resumen técnico, no asesoría legal. Si una licencia de
dataset/código/checkpoint falta o es ambigua, solicitar confirmación escrita o
revisión legal; no asumir permiso.

## Protocolo reproducible

### A. Fijar datos

1. Solicita acceso autorizado y obtén release oficial de videos/anotaciones
   SoccerNet-Tracking. No descargues el corpus antes de aprobar acceso,
   almacenamiento y condiciones.
2. Inventaría los datos locales y crea un manifiesto pequeño y versionado con
   split, partido/secuencia, rutas relativas, frames, resolución, FPS,
   release/version y SHA-256. No incluyas medios.
3. Usa solo train/validation públicos etiquetados para ajustes. Separa por
   **partido**, no por frames/clips del mismo encuentro. Usa la ruta oficial
   de test oculto para claims de challenge.
4. Si la release no ofrece validación etiquetada, reserva 20% de partidos
   oficiales de train con semilla fija; nunca repartas frames del mismo
   partido entre train y validation.
5. Para primera comparación pequeña, elige las primeras cinco secuencias
   elegibles de partidos distintos por ID canónico y registra la selección.
   Conserva todos los frames para MOT. Un smoke test puede usar primeros 300,
   identificado como smoke-only; no reportes como métrica final. Ejecuta split
   completo antes de elegir sistema.
6. Registra versión de datos/split, commits, Python/PyTorch/OpenCV,
   GPU/controlador y semillas en cada ejecución.

### B. Experimento de detector

1. Compara checkpoints fijos de YOLO, RT-DETR y Faster R-CNN, documentando
   preentrenamiento COCO, versiones y hashes. No entrenes/fine-tune en esta
   etapa.
2. Usa mismos frames, aspecto, letterbox, mapeo de clases y postprocesamiento.
   Ajusta confianza/NMS solo en validation y conserva clase/score por caja.
3. Mide precisión/recall/AP solo para categorías presentes en etiquetas.
   Informa aparte detección de cajas agregada si las etiquetas no distinguen
   clases. Si el release no etiqueta el balón, su AP es **no medible**.
4. Informa métricas por clase/tamaño cuando sea posible, latencia solo
   detector, memoria máxima, FPS end-to-end y hardware. Mide al menos tres
   ejecuciones tras warm-up; no incluyas carga de modelo en throughput estable.

### C. Experimento de tracking

1. Cachea detecciones una vez por detector/configuración. Entrega mismas cajas
   y scores a ByteTrack, BoT-SORT sin ReID y BoT-SORT con checkpoint ReID
   documentado. Añade MixSort solo cuando estén resueltas licencias y
   dependencias.
2. Ajusta cada tracker en validation con el mismo presupuesto y congela los
   parámetros antes de test.
3. Evalúa con conversión/protocolo oficial SoccerNet y TrackEval: HOTA, DetA,
   AssA, MOTA, IDF1, cambios de ID y fragmentación. Añade, si se admite, un
   ensayo con detecciones GT para aislar la asociación.
4. Informa por secuencia/split y promedio macro por secuencia. No unas ID entre
   clips separados salvo que ground truth oficial defina el enlace.
5. Mantén separados cortes y ReID entre planos. Sin GT global entre planos,
   informa precisión/recall solo sobre subset de correspondencias etiquetadas,
   no como IDF1 MOT estándar.

### D. ReID y límites de otras tareas

- Evalúa embeddings en el protocolo de retrieval por acción de SoccerNet ReID
  (mAP/rank-1 o métricas oficiales actuales), separado de MOT.
- Evalúa rol/equipo/dorsal/cancha en GSR con GS-HOTA; no mezcles sus scores con
  SoccerNet-Tracking.
- Describe SportsMOT como dominio MOT sin cambios de plano; no lo uses para
  validar balón o ReID entre planos.
- Para métricas sólidas de balón por clase e identidad entre planos, primero
  establece ground truth etiquetado y derechos adecuados.

### Registro de resultados

No hay resultados locales de este benchmark. Registra en una futura ficha
hashes de manifiesto/modelo/pesos, configuración, hardware, runtime, fecha y
métricas reales. No copies valores de papers o leaderboards.

| Ejecución | Manifiesto | Detector/checkpoint | Tracker/config | HOTA | MOTA | IDF1 | ID | AP/recall | FPS | Estado |
|---|---|---|---|---:|---:|---:|---:|---|---:|---|
| — | — | — | — | — | — | — | — | — | — | No ejecutado |

## Evaluación local independiente

`matchvision-benchmark` compara cuatro ejecuciones completas sobre el mismo
video y el mismo Ground Truth humano congelado: YOLO+ByteTrack, HSV/HOG,
OSNet y el modo híbrido. Conserva los JSON de ejecución, sus CSV de
detecciones/tracks y el archivo congelado; el benchmark valida hashes,
configuración común del detector/ByteTrack y número de fotogramas. No modifica
las anotaciones ni sobrescribe informes existentes.

Prepara un mapa explícito entre etiquetas del detector y tipos humanos, por
ejemplo `{"person":"player","sports ball":"ball"}`. No asumas que `person`
separa árbitros y jugadores. Luego ejecuta:

```powershell
matchvision-benchmark `
  --annotations data\annotations\partido-congelado.json `
  --video data\videos\partido.mp4 `
  --class-map configs\clases-evaluacion.json `
  --baseline outputs\runs\baseline\run.json `
  --hsv-hog outputs\runs\hsv-hog\run.json `
  --osnet outputs\runs\osnet\run.json `
  --hybrid outputs\runs\hybrid\run.json `
  --output outputs\benchmarks\partido.json `
  --dataset-kind real_match `
  --iou-threshold 0.5 `
  --appearance-retrieval `
    --reid-sample-interval 10 `
    --reid-maximum-observations 500 `
    --reid-similarity-threshold 0.82 `
    --reid-embedding-weight 0.60 `
    --osnet-weights models\osnet_ain_x1_0.pth
```

`--dataset-kind` es una declaración del operador, no verifica que sea un
partido ni que las etiquetas sean correctas. Usa `synthetic` para datos
sintéticos; usa `real_match` solo para una secuencia real anotada y revisada.
El informe guarda la declaración, hashes, modelo/configuración de cada
ejecución, hardware, tiempos, métricas y CSV comparativo. Los tiempos de
pipeline se toman de cada ejecución, no se vuelven a inferir durante la
evaluación.

La detección se evalúa por clase con emparejamiento uno-a-uno y el umbral IoU
configurado. Aumentar el umbral exige cajas más superpuestas para considerarlas
aciertos y normalmente reduce TP, elevando FN/FP. Tracking usa TrackEval
(`HOTA`, `MOTA`, `IDF1`, cambios y fragmentación) con IoU 0.5, fijo por la
implementación actual; los archivos MOT no incluyen clase, así que esas
métricas son globales y no deben confundirse con detección por clase.

`--appearance-retrieval` calcula, por separado, Rank-1, mAP y asociaciones de
similitud HSV/HOG, OSNet e híbridas sobre cajas humanas, sin depender de
detecciones ni de IDs del tracker. Rank-1/mAP son métricas del descriptor,
no del pipeline de reidentificación completo. Las consultas sin una aparición
positiva en la galería se excluyen del promedio de ranking. Los pares de
asociación comparan apariciones en fotogramas distintos; el umbral de similitud
no es una probabilidad calibrada. Los escenarios de oclusión, cruce,
reaparición y cambio de escena solo se evalúan si el anotador registró eventos
humanos y existe una aparición anterior. Un cambio de escena no demuestra
continuidad de identidad entre cámaras físicas.

Para acotar costo y memoria, el retrieval limita por defecto el cálculo a 500
apariciones, distribuidas determinísticamente y de forma aproximadamente
uniforme por identidad; prioriza las apariciones marcadas con eventos humanos.
El informe registra candidatas, apariciones incluidas/excluidas y fotogramas
seleccionados. Cambia `--reid-maximum-observations` solo si el hardware permite
el costo cuadrático de comparar todas las apariciones.

OSNet e híbrido requieren `--osnet-weights` con el checkpoint verificado.
Si no se indica, sus métricas de apariencia se registran como no disponibles;
no se sustituyen por valores aproximados. Sin `--appearance-retrieval`,
Rank-1/mAP se registran como no evaluados. La tabla comparativa separa esas
métricas de descriptor de las asociaciones de trayectorias obtenidas con
TrackEval. El umbral de similitud y el peso híbrido son configurables y se
registran en la salida; el peso solo afecta al método híbrido.

También se puede evaluar solo la recuperación por apariencia:

```powershell
matchvision-evaluate-reid-retrieval `
  --annotations data\annotations\partido-congelado.json `
  --video data\videos\partido.mp4 `
  --methods hsv_hog embedding hybrid `
  --osnet-weights models\osnet_ain_x1_0.pth `
  --sample-interval 10 `
  --maximum-observations 500 `
  --output outputs\benchmarks\reid-apariencia.json
```

No hay un resultado real de benchmark hasta ejecutar el flujo con un conjunto
de partido anotado por personas y congelado antes de consultar resultados. Las
pruebas automatizadas usan datos sintéticos y validan contratos/métricas; no
demuestran precisión deportiva.

## Referencias

- [Acceso SoccerNet/NDA](https://www.soccer-net.org/data)
- [SoccerNet en Hugging Face](https://huggingface.co/SoccerNet)
- [Kit SoccerNet-Tracking](https://github.com/SoccerNet/sn-tracking)
- [Reglas SoccerNet-Tracking](https://github.com/SoccerNet/sn-tracking/blob/main/ChallengeRules.md)
- [Kit ReID y semántica](https://github.com/SoccerNet/sn-reid)
- [Licencia de código ReID](https://raw.githubusercontent.com/SoccerNet/sn-reid/main/LICENSE)
- [Reglas SoccerNet ReID](https://github.com/SoccerNet/sn-reid/blob/main/ChallengeRules.md)
- [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate)
- [Licencia GSR](https://raw.githubusercontent.com/SoccerNet/sn-gamestate/main/LICENSE)
- [SportsMOT: datos/términos/formato](https://github.com/MCG-NJU/SportsMOT)
- [ByteTrack](https://github.com/FoundationVision/ByteTrack)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
- [MixSort y resultados informados por autores](https://github.com/MCG-NJU/MixSort)
- [Torchreid / OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- [TransReID](https://github.com/damo-cv/TransReID)
- [Licencia Ultralytics](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)
- [Licencia PaddleDetection](https://raw.githubusercontent.com/PaddlePaddle/PaddleDetection/release/2.9/LICENSE)
- [Licencia Torchvision](https://raw.githubusercontent.com/pytorch/vision/main/LICENSE)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
