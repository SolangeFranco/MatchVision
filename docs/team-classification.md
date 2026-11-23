# Agrupamiento visual de equipos

## Alcance

`matchvision.teams.UniformTeamClassifier` agrupa apariciones de tracks en dos
equipos usando el color de la camiseta. Es un estimador visual no supervisado:
no usa el color para fusionar tracks, no identifica a una persona, no predice
el rol de jugador/árbitro y no interpreta `equipo_a` como local ni `equipo_b`
como visitante. Los ID proporcionados por el pipeline solo son claves para
agrupar apariciones.

El método extrae una región aproximada del torso (20–80 % del ancho y 18–58 %
de la altura del recuadro), calcula un histograma HSV de tono/saturación y un
histograma de brillo, y promedia las muestras espaciadas de cada ID. Un
agrupamiento determinista de dos centros propone los equipos. Se abstiene si
faltan muestras, los grupos son demasiado parecidos, una asignación queda
ambigua o un color es atípico. La puntuación de confianza es un margen de
distancia entre histogramas; **no es una probabilidad**.

El recorte no es segmentación semántica y puede incluir fondo, piel o
uniformes de portero/árbitro. Iluminación, sombras, cámaras y camisetas
similares pueden cambiar los histogramas. No se usa un modelo entrenado ni se
afirma precisión hasta probarlo contra etiquetas humanas.

## Ejecutar sobre tracks existentes

Genera las trayectorias usando el pipeline habitual. El módulo procesa el
video original y los recuadros guardados; no vuelve a ejecutar detección ni
tracking:

```powershell
python -m matchvision.pipeline.cli `
  data\videos\amateur-football-clip.webm `
  --output outputs\video-runs `
  --device cpu `
  --persistent-identities
```

Selecciona el CSV de trayectorias del nuevo resultado y una ruta de salida
nueva para la clasificación:

```powershell
python -m matchvision.teams.cli `
  --video data\videos\amateur-football-clip.webm `
  --trajectories outputs\video-runs\<run-id>_trajectories.csv `
  --identity-column persistent_id `
  --output outputs\team-classification\<run-id>_team_assignments.csv
```

Al instalar el proyecto con `pip install -e .`, también queda disponible el
comando `matchvision-classify-teams` con los mismos argumentos. El programa
escribe un CSV por ID de entrada y un manifiesto JSON con hashes, parámetros,
estados y limitaciones. Se niega a sobrescribir cualquiera de los dos
archivos.

Estados:

- `assigned`: propuesta con separación suficiente frente al otro centro.
- `ambiguous`: no hay margen suficiente para decidir con seguridad.
- `outlier`: apariencia alejada de ambos centros; revisar posibles árbitros,
  porteros, oclusiones o recuadros deficientes.
- `insufficient_samples`: faltan observaciones utilizables del ID.
- `insufficient_data`: no hay suficientes apariciones por grupo o los colores
  observados no permiten separar dos grupos.
- `manual`: etiqueta corregida explícitamente por una persona; no tiene
  puntuación estimada.

Un estado no asignado no significa árbitro. El sistema no tiene una clase de
árbitro confiable en el baseline COCO.

## Correcciones humanas

Prepara un archivo de overrides nuevo; no edites los resultados previos:

```json
{
  "schema_version": 1,
  "identity_column": "persistent_id",
  "assignments": {
    "12": "equipo_a",
    "37": "equipo_b"
  }
}
```

Vuelve a ejecutar el comando incluyendo `--overrides
outputs\team-classification\correcciones.json`. Una corrección manual prevalece
sobre la estimación y queda identificada como `manual` en el CSV. Para usar
ID de tracks temporales en lugar de las propuestas de ID persistente, elige
`--identity-column temporal_track_id` y crea el archivo de correcciones con el
mismo valor en `identity_column`.

Revisa el video alrededor de los casos ambiguos/atípicos antes de corregir. No
asignes un equipo basándote únicamente en un color detectado en un recuadro
borroso, pequeño u ocluido. Si los dos uniformes se parecen o no hay evidencia
suficiente, deja la propuesta sin equipo hasta revisarla.

## Validación y estado de los resultados

Las pruebas unitarias cubren dos colores claramente distintos, uniformes
indistinguibles, correcciones manuales, muestreo espaciado, validación de
recuadros, exportación y protección contra sobrescritura. Son pruebas de
comportamiento con imágenes sintéticas; **no demuestran precisión en partidos
reales**.

Se procesó el video local y las trayectorias preservadas para verificar el
flujo real (no para medir precisión):

- 958 fotogramas del video; 4.847 observaciones; 66 ID persistentes de entrada.
- 65 ID tuvieron al menos un recorte de apariencia utilizable y 41 alcanzaron
  las tres muestras mínimas.
- 26 propuestas se asignaron automáticamente: 23 a `equipo_a` y 3 a
  `equipo_b`. Otras 5 quedaron ambiguas, 10 se marcaron atípicas y 25 tuvieron
  muestras insuficientes.
- No se aplicaron correcciones manuales y no se calcularon métricas de
  exactitud. La distribución 23/3 es muy desequilibrada y **no debe
  interpretarse como una identificación fiable de los equipos**; requiere
  revisión humana antes de usarse.
- Manifiesto y checksum del CSV:
  `outputs/team-classification/4052ed58-6c37-4a2c-b62a-920dfe1e700b_team_assignments_v3.csv.json`
  y SHA-256 `864a61c733d0522131eda59234e70773686e5bef75dd0792bf6a1af7b80a2250`.

El informe también registra el SHA-256 del video y del CSV de trayectorias
usados. Estas cantidades son salidas del algoritmo, no ground truth. Para
validar la clasificación hay que revisar identidades visuales, registrar
correcciones y comparar con etiquetas humanas; no se deben presentar las
asignaciones actuales como resultados correctos.

El baseline YOLO11n/COCO solo detecta `person` y puede mezclar jugadores y
árbitros. El archivo MOT no conserva rol ni equipo. No se descargan pesos,
datasets ni modelos externos para este módulo.
