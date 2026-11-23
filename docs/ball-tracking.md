# Detección y seguimiento del balón

## Alcance e interfaz

`matchvision-track-ball` reutiliza el protocolo `Detector` del proyecto y
permite ejecutar un detector Ultralytics compatible sobre cada fotograma. El
CLI acepta checkpoints personalizados, IDs de clase para el balón y CPU/GPU.
El postprocesador `BallTracker` se ocupa exclusivamente del balón y no asigna
ni modifica identidades de jugadores.

El seguimiento conserva un segmento mientras las detecciones puedan asociarse
con la posición y velocidad recientes. Filtra cajas inválidas, demasiado
grandes o extremadamente alargadas; durante una pérdida breve extrapola la
posición con velocidad amortiguada; al reaparecer, intenta asociarla dentro de
una región de búsqueda proporcional al tiempo sin detecciones. Si la pérdida
supera el límite, crea un segmento nuevo en la detección siguiente. No usa
apariencia, optical flow ni aprendizaje para inventar una observación durante
una oclusión.

Los estados de cada fotograma son:

- `detectada`: centro y caja proceden de una salida del detector; la confianza
  se conserva como confianza del detector, no como probabilidad calibrada.
- `predicha`: extrapolación temporal; no es una detección ni ground truth y no
  tiene confianza del detector.
- `sin_deteccion`: no hay una posición confiable disponible.

Las coordenadas se expresan en píxeles de la imagen de entrada. No representan
posición sobre la cancha, velocidad física ni distancia real.

## Ejecución

Con los pesos COCO ya instalados:

```powershell
matchvision-track-ball `
  data\videos\amateur-football-clip.webm `
  --output outputs\ball-tracking `
  --device cpu `
  --confidence 0.05 `
  --image-size 1280
```

También se puede usar `python -m matchvision.ball.cli` con los mismos
argumentos. Para un checkpoint especializado, primero verifica sus clases,
procedencia, licencia de los pesos y licencia/derechos del dataset. A
continuación declara la procedencia en la ejecución y selecciona la clase
correcta si no se llama `ball`, `football`, `soccer ball` o `sports ball`:

```powershell
matchvision-track-ball `
  data\videos\amateur-football-clip.webm `
  --output outputs\ball-tracking `
  --model models\football-ball.pt `
  --model-source "URL, organización y revisión exacta" `
  --model-license "Licencia comprobada de estos pesos" `
  --pretrained-on "Dataset y versión comprobados" `
  --classes 0 `
  --ball-class-ids 0
```

El adaptador del detector rechaza pesos personalizados sin esos tres datos de
procedencia. `--classes` limita la inferencia del detector; `--ball-class-ids`
elige cuáles de las clases devueltas por ese modelo representan el balón.
Comprueba siempre los IDs contra el mapa de clases del checkpoint.

Cada ejecución crea un CSV y su manifiesto JSON bajo la carpeta indicada. Los
nombres son únicos; el procesamiento rechaza sobrescribir salidas existentes.
El manifiesto registra hashes SHA-256 del video y CSV, configuración, clase,
procedencia declarada, pesos/checksum cuando están disponibles, conteos por
estado y limitaciones.

## Decisión de modelos, investigación y licencias

El adapter es agnóstico al modelo, pero el checkpoint `yolo11n.pt` incluido
como baseline es el modelo general de Ultralytics preentrenado en COCO. Su
`sports ball` es una clase genérica, no una garantía de localizar el balón de
fútbol pequeño, borroso o parcialmente ocluido. Reducir el umbral puede
recuperar candidatos de baja confianza, pero también aumenta falsos positivos;
el seguimiento temporal no puede convertir un falso positivo en una detección
correcta.

Se revisaron implementaciones públicas de entrenamiento YOLOv8 y de
asociación temporal basada en embeddings para fútbol, entre ellas
[soccer_balls_tracking](https://github.com/godSaveDaniele/soccer_balls_tracking)
y [tracking-soccerball-yolov8](https://github.com/anujsainarain/tracking-soccerball-yolov8).
Son pistas metodológicas —modelos finos sobre datos de fútbol, búsqueda
temporal local y embeddings—, no checkpoints validados para este clip. No se
halló una licencia verificable de los pesos/datos en las páginas consultadas;
por tanto no se descargaron ni incorporaron artefactos. Sus resultados
publicados no se trasladan a este proyecto.

El código de Ultralytics se distribuye bajo AGPL-3.0 con una licencia
Enterprise alternativa para ciertos despliegues comerciales. Verifica los
términos actuales aplicables al código, los pesos, sus datos de entrenamiento
y tu producto; esta implementación no concede derechos comerciales.

Fuentes consultadas:

- [Repositorio Ultralytics](https://github.com/ultralytics/ultralytics) y
  [página de licencias](https://www.ultralytics.com/license).
- [SoccerNet Game State Reconstruction](https://github.com/SoccerNet/sn-gamestate),
  framework/dataset de investigación de tracking y atributos de cancha; no se
  usó ni descargó en esta etapa.
- Repositorios públicos de balón citados arriba. No se copiaron código, pesos
  ni datos de esos repositorios.

## Límites y validación requerida

La asociación actual usa distancia y velocidad en píxeles de imagen. En una
cámara móvil, paneos, zoom y cortes aparentes desplazan toda la escena y
pueden invalidar la predicción. Detecciones múltiples de objetos blancos,
líneas, jugadores o publicidad pueden confundirse con el balón. El sistema no
usa segmentación de cancha, homografía, compensación de movimiento, optical
flow ni embeddings.

Las pruebas unitarias usan detecciones sintéticas para comprobar asociación,
oclusiones breves, recuperación, expiración de segmentos y exportación. No
miden precisión de detección. Para una evaluación real se necesita un conjunto
representativo de fotogramas de partidos, anotaciones humanas del centro/caja
del balón y métricas por clase (por ejemplo, precision/recall, AP con umbral
pequeño y error de centro). Deben separarse clips de entrenamiento y evaluación
por partido; no se debe ajustar el umbral sobre el conjunto final.

El siguiente paso de esta etapa es obtener datos y pesos cuyo uso esté
autorizado, etiquetar una muestra de nuestro video o de una fuente deportiva
con licencia clara, y comparar COCO frente a un detector fino bajo las mismas
condiciones. Hasta entonces, los conteos de inferencia son observaciones del
modelo, no resultados de exactitud.

## Comprobación en video disponible

Se ejecutó una comprobación corta con el video amateur local, solo para
verificar la integración real del detector y el exportador:

- Video fuente SHA-256:
  `42e9d2a0468a7fc08622eccabea2bd640324f9550d1909f99ead0e49a1bfeac0`.
- Checkpoint local preexistente `yolo11n.pt`, SHA-256:
  `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`;
  Ultralytics `8.4.204`, inferencia CPU, `imgsz=960`, confianza `0.05`,
  clases COCO `person` y `sports ball`.
- Fotogramas fuente 351–366: los 16 fotogramas se reescalaron para la prueba
  temporal a 1920×1080.
- Resultado: **0 detecciones de balón, 0 predicciones y 16 fotogramas sin
  detección**. No se generaron etiquetas ni se interpretó este resultado como
  precisión del modelo. Los archivos temporales se eliminaron.

Esta comprobación confirma que el flujo procesa video real, pero también deja
claro que el baseline COCO no produce una trayectoria utilizable en esta
muestra. Para continuar se requiere un detector especializado validado y
etiquetas humanas; no se deben completar esos fotogramas manualmente a partir
de la salida del modelo.
