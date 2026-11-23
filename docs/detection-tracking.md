# Primera implementación de detección y tracking

## Selección inicial de modelos

La implementación inicial usa Ultralytics YOLO11n preentrenado en COCO y la
implementación ByteTrack de Ultralytics. Es un baseline ejecutable y conocido;
los protocolos `Detector` y `MultiObjectTracker` permiten reemplazarlos.

El modelo COCO devuelve clases genéricas `person` y `sports ball` (el adaptador
filtra esas categorías cuando están en el mapa de clases). **No distingue
jugadores de árbitros, no clasifica equipos por camiseta ni reconoce
específicamente el fútbol.** El balón puede perderse o localizarse mal por su
tamaño y desenfoque. No se ha establecido su recall. No se deben corregir estas
limitaciones cambiando el nombre de las detecciones a mano ni infiriendo el
equipo desde la clase COCO. Se puede proporcionar un checkpoint especializado
con `--model`, pero sus etiquetas solo se deben interpretar tras verificar
anotaciones y procedencia.

| Alternativa | Ventajas | Límites y decisión |
|---|---|---|
| YOLO11n COCO + ByteTrack (baseline elegido) | Pesos preentrenados, API de inferencia de video y configuración sencilla. | Etiquetas genéricas de persona/objeto deportivo; el balón puede ser demasiado pequeño y COCO no identifica árbitros. Sirve para validar el flujo, no como detector deportivo final. |
| Checkpoint YOLO específico de fútbol (por ejemplo, un modelo comunitario de Roboflow Sports) | Puede localizar mejor jugadores/escenas; las clases personalizadas podrían incluir roles. | La procedencia, licencia de datos, definición de clases, calidad de etiquetas y términos comerciales dependen del artefacto. Un listado público de datos no prueba que exista un checkpoint permisivo ni etiquetas de árbitro. Revisar cada artefacto antes de elegirlo. |
| RT-DETR con PaddleDetection | Familia de detectores alternativa y opciones preentrenadas COCO. | Requiere integrar runtime/preprocesamiento y medir latencia; los pesos genéricos mantienen las mismas limitaciones de clase. Posponer hasta tener split de evaluación y benchmark de hardware. |
| Faster R-CNN con Torchvision | Referencia conocida de dos etapas para comparar precisión. | La latencia puede no ser adecuada para partidos completos y las etiquetas genéricas siguen limitando. No es la primera opción en tiempo real. |
| BoT-SORT | API con compensación de movimiento de cámara y ReID opcional. | Tiene más configuración/dependencias; la compensación no resuelve cortes arbitrarios de transmisión. Añadir detrás del mismo protocolo tras comparar con ByteTrack. |

### Procedencia y licencias

- **Modelo**: `yolo11n.pt` oficial de Ultralytics, preentrenado en COCO. La
  biblioteca lo resuelve/descarga solo si una ejecución solicita inferencia.
  Registrar ruta y SHA-256 del checkpoint en el JSON de ejecución. Fuentes:
  [modelos Ultralytics](https://github.com/ultralytics/ultralytics) y
  [licencia](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- **Código**: el repositorio de Ultralytics es AGPL-3.0; el proveedor ofrece
  una licencia comercial separada. Revisar términos de código y checkpoint
  para el uso previsto antes de distribuir o comercializar. Este proyecto no
  concede esos derechos.
- **Tracker**: ByteTrack se integra desde Ultralytics y usa su configuración;
  queda sujeto a la misma revisión de licencia.
- **No se ha ajustado ningún modelo** y no se versionan pesos. El JSON guarda
  argumento, versiones, hash del checkpoint cuando está disponible y
  parámetros.
- En esta etapa no había pesos COCO descargados ni video de partido en el
  workspace. La primera inferencia con el modelo por defecto requiere acceso a
  Internet para descargarlo o un checkpoint oficial previamente almacenado.

## Procesamiento y resultados

`matchvision-track-video INPUT --output OUTPUT_DIR` procesa un fotograma a la
vez y escribe un archivo de predicción MOT de 10 columnas, un MP4 anotado y
un JSON con procedencia del modelo/tracker, tiempo y FPS. El adaptador mantiene
una instancia ByteTrack independiente por clase para evitar asociar, por
ejemplo, una persona con el balón; después asigna ID únicos en la salida.

Para comparar con ground truth MOT local:

```powershell
matchvision-track-video path\to\match.mp4 `
  --output outputs\video-runs `
  --ground-truth path\to\gt.txt
```

El pipeline llama al mismo evaluador TrackEval de HOTA/MOTA/IDF1 usado en los
benchmarks. No pases anotaciones de otra secuencia: se comprueban rangos de
fotogramas y las métricas corresponden solo a los archivos proporcionados. Sin
ground truth, el JSON no contiene métricas de precisión.

Para un checkpoint personalizado, especifica toda su procedencia:

```powershell
matchvision-track-video path\to\match.mp4 `
  --model models\football.pt `
  --model-source "URL, proveedor y revisión exacta del artefacto" `
  --model-license "Licencia exacta del checkpoint" `
  --pretrained-on "Dataset de entrenamiento documentado"
```

El comando rechaza checkpoints personalizados sin esos datos. Guarda en el
JSON el modelo, origen, licencia, dataset de preentrenamiento, SHA-256 local
cuando la biblioteca lo expone y versiones de runtime.

Se puede elegir CPU con `--device cpu`; CUDA acepta `--device 0` u otro valor
compatible con Ultralytics. No se promete tiempo real: tiempo transcurrido y
FPS medidos quedan registrados por ejecución.

## Límites actuales de identidad y categorías

- Los ID son locales a un video procesado continuamente; no persisten entre
  archivos, cambios de plano, repeticiones o cámaras.
- `person` de COCO incluye árbitros, pero no indica si es jugador, árbitro,
  portero o entrenador.
- `sports ball` no es una clase específica de fútbol y el balón de transmisión
  suele ser pequeño o borroso. Este baseline no promete un recall fiable.
- ByteTrack aislado por clase reduce asociaciones entre clases, pero no
  resuelve ReID tras oclusiones largas o cortes de cámara.
- El evaluador CSV de SoccerNet MOT es agnóstico a clases y no valida
  detecciones específicas de árbitro o balón.

El siguiente paso de selección es obtener video autorizado y secuencias
etiquetadas de validación, inspeccionar licencias y etiquetas de cada
artefacto, y comparar detectores/trackers sobre las mismas secuencias y
hardware.

El seguimiento del balón tiene un flujo independiente y especializado para
asociación/predicción temporal: consulta
[ball-tracking.md](ball-tracking.md). El pipeline YOLO + ByteTrack general no
debe interpretarse como detector confiable del balón de fútbol.

## Requisitos para una ejecución local de video

1. Video de partido legible (`.mp4`, `.mkv` o `.avi`) cuyo uso esté autorizado.
   El workspace actual también contiene el clip amateur documentado en
   `experiment-run-2026-11-08.md`.
2. Acceso a Internet para descargar `yolo11n.pt` en la primera ejecución o un
   checkpoint Ultralytics ya almacenado. Un modelo personalizado necesita
   origen, licencia y dataset de preentrenamiento en el CLI.
3. Para métricas de precisión, anotaciones MOT de esa secuencia exacta con la
   misma numeración de fotogramas y formato de 9/10 columnas. Sin ellas solo
   se informan procesamiento y archivos de salida, no precisión de tracking.
