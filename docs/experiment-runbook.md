# Guía de preparación y evaluación de experimentos

Esta guía valida datos locales ya disponibles. La inferencia local de
detección/tracking está descrita en [detection-tracking.md](detection-tracking.md).
No descarga datasets ni entrena modelos. Comprueba los términos y el acceso
vigentes antes de obtener datos.

## Instalación y validación

Instala dependencias declaradas y la revisión fijada de TrackEval:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,vision,annotation]"
matchvision-validate-config --config configs/benchmark.yaml
```

El extra `annotation` proporciona PyAV para crear copias de revisión H.264.
Si solo se va a validar configuración y manifiestos, basta con
`.[dev,vision]`.

La configuración del workspace selecciona `.venv\Scripts\python.exe`. Si VS
Code tiene seleccionado otro intérprete, usa **Python: Select Interpreter** y
abre después una terminal integrada nueva.

Para validar también un manifiesto de secuencias:

```powershell
matchvision-validate-config --config configs/benchmark.yaml --manifest data/annotations/soccernet_tracking_manifest.json
```

El YAML del benchmark se valida con el JSON Schema del paquete. El manifiesto
tiene su propio esquema y comprobaciones cruzadas: cada partido debe aparecer
en un único split y cada ID de secuencia debe ser único. Puede exigir y
verificar SHA-256 de video/anotaciones. Las rutas son relativas al manifiesto;
los medios no se copian ni modifican.

## Manifiesto de secuencias

El esquema formal está en
[`src/matchvision/schemas/sequence-manifest.schema.json`](../src/matchvision/schemas/sequence-manifest.schema.json).
Crea un JSON con una entrada por secuencia anotada:

```json
{
  "manifest_version": 1,
  "dataset": "SoccerNet-Tracking",
  "release": "indicar-la-version-autorizada",
  "sequences": [
    {
      "game_id": "game-001",
      "sequence_id": "sequence-001",
      "split": "validation",
      "video_path": "../raw/game-001/sequence-001.mp4",
      "annotation_path": "gt/sequence-001.txt",
      "frame_count": 750,
      "fps": 25,
      "width": 1280,
      "height": 720,
      "video_sha256": "<64 caracteres hexadecimales>",
      "annotation_sha256": "<64 caracteres hexadecimales>"
    }
  ]
}
```

Los hashes son marcadores de posición: reemplázalos por valores reales. No
versiones medios ni manifiestos que revelen contenido sujeto a acceso o
licencia. La separación se hace por **partido**: todas las secuencias de un
partido deben pertenecer al mismo split (`train`, `validation`, `test` o
`challenge`).

## Anotaciones y predicciones

El evaluador admite filas MOTChallenge/SoccerNet de 9 o 10 columnas:

```text
frame,id,x,y,width,height,confidence,unused,unused[,unused]
```

El fotograma empieza en 1 y las cajas usan `x,y,width,height` desde la esquina
superior izquierda. Los valores deben ser finitos, las dimensiones positivas
y no puede haber más de una caja por ID y fotograma. Los índices deben caber
en `frame_count`. El evaluador usa IoU de cajas sin clase en los umbrales de
TrackEval; estas anotaciones no generan métricas de balón/equipo por clase.

Pon un archivo de predicción `<sequence_id>.txt` por secuencia seleccionada en
un directorio. Usa el mismo formato MOT y registra versiones del detector y
tracker que generaron las predicciones sobre ese video exacto.

## Ejecutar una evaluación reproducible pequeña

Primero prepara un manifiesto autorizado, anotaciones y predicciones. La
configuración selecciona `validation` y requiere hashes por defecto:

```powershell
matchvision-evaluate `
  --config configs/benchmark.yaml `
  --manifest data/annotations/soccernet_tracking_manifest.json `
  --predictions data/predictions/bytetrack `
  --output outputs/benchmarks `
  --limit 5 `
  --detector-version "model-name@checkpoint-or-sha256" `
  --tracker-version "ByteTrack@commit-or-config"
```

`--limit` elige las primeras N secuencias en orden estable
`(game_id, sequence_id)` dentro del split configurado. Es una evaluación de
humo, no un benchmark final. Sin `--limit` se procesa todo el split. Si falta
un archivo, falla un hash, hay fuga entre splits o una fila MOT no es válida,
la ejecución aborta con un error explícito.

Cada ejecución crea:

- `<experiment-id>.json`: configuración exacta/hash, release del dataset y
  hash del manifiesto, secuencias/hashes de entrada, checkpoint/parámetros,
  versiones, HOTA/MOTA/IDF1 por secuencia y macro, cambios de ID, tiempos y
  entorno de software.
- `<experiment-id>.csv`: scores y tiempos por secuencia.

Los resultados son append-only y se escriben de forma atómica. Los promedios
macro pesan cada secuencia por igual; no deben presentarse como la agregación
oficial del challenge SoccerNet a menos que reproduzcan ese protocolo.

## Métricas y procedencia

HOTA, CLEAR/MOTA e Identity/IDF1 se calculan con las clases oficiales de
[TrackEval](https://github.com/JonathonLuiten/TrackEval) en el commit fijado en
`pyproject.toml`. El evaluador prepara similitudes IoU por fotograma y llama a
esas clases; no vuelve a implementar sus fórmulas. Un shim de compatibilidad
provee alias NumPy que se eliminaron después de publicar esa revisión.

El paquete de métricas TrackEval puede emitir una advertencia opcional por
falta de `pycocotools` al cargar métricas BURST no relacionadas. HOTA/MOTA/IDF1
MOT no lo requieren y este evaluador no usa BURST.

## Bloqueos externos actuales

- Acceso autorizado a SoccerNet, aceptación de términos, release verificado y
  manifiesto de splits.
- Archivos de predicción de detector/tracker generados localmente y con
  versiones registradas.
- En esta etapa aún no se implementa inferencia para benchmark, evaluación de
  balón por clase ni comparación de detectores/trackers. No se reportarán
  scores hasta tener predicciones y ejecutar la evaluación.
