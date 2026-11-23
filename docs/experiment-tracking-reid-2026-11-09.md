# Experimento ByteTrack + ID persistentes/ReID

## Propósito y auditoría

El baseline ya separaba detección (`Detector`), tracking temporal
(`MultiObjectTracker`), procesamiento de fotogramas, exportación MOT,
visualización y evaluación TrackEval. Los ID ByteTrack eran locales al video;
no había asociación entre un tracklet nuevo y una persona previa. El MOT no
conserva clases y no había etiquetas humanas de identidad para este clip.

La implementación mantiene YOLO11n y ByteTrack, y añade un `IdentityAssigner`
opcional entre el tracking temporal y las exportaciones. No se descargó ningún
checkpoint/dataset ni se entrenó un modelo. Alternativas, fuentes, límites y
umbral están documentados en [tracking-reid.md](tracking-reid.md).

## Ejecución final reproducible

ID de ejecución: `4052ed58-6c37-4a2c-b62a-920dfe1e700b`.

- **Entrada:** `data/videos/amateur-football-clip.webm`; SHA-256, atribución y
  licencia están en el JSON y en [el experimento del clip original](experiment-run-2026-11-08.md).
- **Detector/tracker:** YOLO11n oficial COCO + Ultralytics ByteTrack.
- **Dispositivo/runtime:** CPU; Python 3.12.10, Ultralytics 8.4.204, OpenCV
  4.14.0, NumPy 2.5.4. CUDA no estaba disponible.
- **ReID:** [configs/reidentification.yaml](../configs/reidentification.yaml):
  clase COCO 0, gap máximo 150 fotogramas, protección de ID activo 8,
  umbral de asociación 0.82, margen de ambigüedad 0.06, asignación uno a uno,
  descriptores HOG+HSV y gates de movimiento/escala. ReID entre escenas está
  desactivado.
- **Fotogramas:** 958 decodificados y procesados, igual que el baseline.
- **Observaciones:** 4.847, igual que el baseline; todas las predicciones
  exportadas fueron COCO `person` (clase 0). No se exportó ningún track del
  balón.
- **ID ByteTrack temporales:** 81.
- **ID persistentes propuestos:** 66; 15 propuestas de reidentificación y 14
  ID persistentes que reúnen más de un tracklet temporal.
- **Scores de asociación:** 15 propuestas; media 0.8375, mínimo 0.8224. No
  son probabilidades.
- **Duración por observaciones:** media 73.44 y mediana 17.5 por ID
  persistente, frente a media 59.84 y mediana 17 por ID temporal baseline.
  Alcance en fotogramas: media 102.38 y mediana 24.5, frente a 76.88 y 21.
- **Tracks cortos:** 16 ID persistentes tienen como máximo 5 observaciones
  (baseline: 19); 27 tienen como máximo 10 (baseline: 33).
- **Tiempo:** 315.57 segundos; 3.036 FPS procesados. Es una sola ejecución
  CPU, no un benchmark controlado.
- **Cortes detectados:** 0.
- **Métricas de precisión:** no calculadas; no existe ground truth.

También se ejecutó el umbral exploratorio 0.76 (`7d9f9808-27c6-47c8-8d43-b31fa7626339`):
44 relinks propuestos y 37 ID persistentes. Se revisaron cinco pares
seleccionados como recortes: cuatro parecían visualmente compatibles y uno fue
inconcluso. Esa muestra no es una auditoría sistemática ni hay etiquetas de
identidad, así que el umbral final subió a 0.82 para favorecer abstención. La
ejecución exploratoria se preserva y no es la configuración final.

## Baseline frente a proxies finales

| Observación | ByteTrack baseline | ID persistente |
|---|---:|---:|
| Fotogramas procesados | 958 | 958 |
| Observaciones | 4.847 | 4.847 |
| ID de salida | 81 temporales | 66 persistentes |
| Media de observaciones por ID | 59.84 | 73.44 |
| Mediana de observaciones por ID | 17 | 17.5 |
| Alcance medio en fotogramas | 76.88 | 102.38 |
| Alcance mediano en fotogramas | 21 | 24.5 |
| ID con ≤5 observaciones | 19 | 16 |
| Segmentos consecutivos de presencia | 431 | 431 |
| Fotogramas sin detección exportada | 42 | 42 |

El menor número de ID y tracks descriptivos más largos indican que la capa
asoció algunos tracklets ByteTrack. **No demuestran que los enlaces sean
correctos.** Los segmentos consecutivos totales no disminuyeron; continúan
habiendo huecos de detección. `excess_segments_over_ids` es 350 en baseline y
365 en la salida final porque resta menos ID de los mismos 431 segmentos; no
se debe interpretar como una métrica estándar de fragmentación.

Throughput medido: baseline 2.794 FPS, exploratorio 2.902 FPS, final 3.036 FPS.
Son mediciones observadas de ejecuciones individuales, no una mejora de
velocidad atribuible a ReID; no se ha caracterizado variación ni warm cache.

## Archivos generados

Se comprobó que las salidas finales no estuvieran vacías, coincidieran con los
SHA-256 del JSON y que el video anotado decodificara los 958 fotogramas a
3840×2160 y 29.97 FPS.

- MP4 anotado: `outputs/video-runs/4052ed58-6c37-4a2c-b62a-920dfe1e700b_annotated.mp4`
- MOT con ID persistentes: `outputs/video-runs/4052ed58-6c37-4a2c-b62a-920dfe1e700b.txt`
- Trayectorias temporales/persistentes: `outputs/video-runs/4052ed58-6c37-4a2c-b62a-920dfe1e700b_trajectories.csv`
- Estadísticas proxy: `outputs/video-runs/4052ed58-6c37-4a2c-b62a-920dfe1e700b_tracking_stats.json`
- Parámetros, versiones, tiempos y hashes: `outputs/video-runs/4052ed58-6c37-4a2c-b62a-920dfe1e700b.json`
- Estadísticas baseline y ReID: `outputs/tracking-analysis/baseline-byte-track/` y
  `outputs/tracking-analysis/persistent-reid-threshold-082/`
- Recortes cualitativos: `outputs/tracking-analysis/reid-review-examples.png` y
  `outputs/tracking-analysis/reid-low-score-review.png`
- Vista previa anotada: `outputs/tracking-analysis/persistent-annotated-preview.png`

El video muestra cajas, clase, score, ID persistente de color consistente y
trazas cortas. El CSV conserva ID temporales/persistentes, estado y score de
asociación.

## Límites y criterio de éxito

La capa propone 15 asociaciones por encima del umbral heurístico y exporta
artefactos reproducibles; **no** demuestra mayor precisión de identidad real
ni menos fusiones falsas. La pequeña muestra visual no sustituye ground truth.
No se ejecuta ReID entre cortes arbitrarios de cámara. El clip es una sola
toma amateur desde arriba y no representa una transmisión profesional. COCO
`person` mezcla jugadores y árbitros; los descriptores pueden confundir
uniformes similares o recortes pequeños/desenfocados.

La precisión sigue pendiente: requiere anotaciones humanas o revisión
sistemática de fusiones falsas, cruces, oclusiones, salidas/reapariciones y
cambios de escena, y después métricas de identidad. Hasta entonces los ID son
propuestas prudentes locales al video, no identidades verificadas.
