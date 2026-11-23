# Ejecución de prueba YOLO11n + ByteTrack

Esta ejecución de humo comprueba decodificación de video, detección,
seguimiento local, visualización y exportación MOT en un clip breve de fútbol
amateur con licencia abierta. No es un benchmark ni una afirmación de
precisión.

## Fuente y atribución

- **Obra:** [Amateurfußball Torschuss von oben](https://commons.wikimedia.org/wiki/File:Amateurfu%C3%9Fball_Torschuss_von_oben.webm)
- **Autor:** Maximilian Schönherr
- **Licencia:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)
- **Archivo local:** `data/videos/amateur-football-clip.webm`
- **SHA-256 de entrada:** `42e9d2a0468a7fc08622eccabea2bd640324f9550d1909f99ead0e49a1bfeac0`

El video anotado añade recuadros, etiquetas e ID del tracker; la fuente no se
modificó. Para distribuir una adaptación, incluye atribución, enlace a la
licencia, indicación de cambios y cumple las condiciones share-alike de CC BY-SA
4.0.

## Configuración

- **ID de ejecución:** `8ec67506-93cc-4f17-b80f-d98faabea23f`
- **Detector:** YOLO11n oficial de Ultralytics preentrenado en COCO
- **SHA-256 del checkpoint:** `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`
- **Tracker:** Ultralytics ByteTrack
- **Runtime:** Python 3.12.10, Ultralytics 8.4.204, OpenCV 4.14.0, NumPy 2.5.4
- **Dispositivo:** CPU (CUDA no disponible)
- **Tamaño de imagen:** 1280
- **Confianza:** 0.25
- **ByteTrack:** umbral de activación 0.25, buffer de pérdida 30 fotogramas,
  umbral de asociación 0.8
- **Clases:** COCO `person` (0) y `sports ball` (32)
- **Ground truth:** no se proporcionó; no se calcularon métricas de precisión

## Resultados observados

- **Fotogramas procesados:** 958. Los metadatos de fuente declaran 959; OpenCV
  decodificó 958 y se procesaron todos los disponibles.
- **Detecciones exportadas:** 4.847 filas MOT.
- **ID de track únicos:** 81.
- **Fotogramas con al menos una caja:** 916; la última fila MOT corresponde al
  fotograma 935 porque no hubo tracks emitidos en los siguientes.
- **Tiempo de procesamiento:** 342.82 segundos.
- **Throughput medio:** 2.79 fotogramas procesados por segundo.
- **Métricas:** no disponibles; no se proporcionó ground truth coincidente.

Se volvió a abrir el MP4 anotado y se decodificaron los 958 fotogramas a
3840×2160 y 29.97 FPS. La exportación MOT tiene 10 columnas y fotogramas entre
1 y 935. Se verificaron los hashes del registro JSON y de ambas salidas.

## Artefactos

- Video anotado: `outputs/video-runs/8ec67506-93cc-4f17-b80f-d98faabea23f_annotated.mp4`
- Predicciones MOT: `outputs/video-runs/8ec67506-93cc-4f17-b80f-d98faabea23f.txt`
- Registro de ejecución: `outputs/video-runs/8ec67506-93cc-4f17-b80f-d98faabea23f.json`

## Límites

Es un clip amateur de 32 segundos grabado desde arriba, no una transmisión
profesional. `person` de COCO no distingue jugadores de árbitros ni equipos.
`sports ball` no es una clase específica de fútbol y la ejecución no establece
una detección fiable del balón. Los ID ByteTrack son locales a este video y no
resuelven ReID entre cortes o archivos. El FPS depende del clip, CPU y
configuración; no es un benchmark general.
