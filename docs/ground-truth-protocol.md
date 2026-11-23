# Protocolo de Ground Truth humano

Estado: implementación funcional; todavía no existe un conjunto humano real
congelado para el partido.

## Video seleccionado

La primera captura disponible y verificable es
`data/videos/amateur-football-clip.webm`:

- resolución: 3840 × 2160;
- FPS: 29.97002997002997;
- fotograma informado por el contenedor: 959;
- fotogramas decodificados por OpenCV: 958;
- duración aproximada: 31.96 segundos;
- SHA-256:
  `42e9d2a0468a7fc08622eccabea2bd640324f9550d1909f99ead0e49a1bfeac0`.

Se usa el número decodificado, 958, como límite temporal. El clip está documentado
como material amateur y no sustituye Ground Truth de una transmisión completa.

## Muestreo reproducible

Generar el plan sin crear etiquetas:

```powershell
matchvision-create-annotation-plan `
  --video data\videos\amateur-football-clip.webm `
  --output outputs\ground-truth\amateur-football-sampling-plan-v1.json `
  --interval-frames 30 `
  --dense-window-seconds 2 `
  --dense-window-count 3
```

El plan actual contiene:

- muestreo temporal cada 30 fotogramas;
- una ventana densa al inicio, una central y una final;
- 207 fotogramas planificados;
- ventanas densas de 60 fotogramas consecutivos:
  67–126, 449–508 y 832–891;
- hash del plan y del video.

El plan solo selecciona fotogramas. Las identidades y cajas se anotan
manualmente y nunca se generan a partir de las predicciones.

## Casos difíciles

Los casos difíciles deben confirmarse por inspección humana. Se pueden añadir
solo después de observarlos:

```text
FRAME:crossing:Los jugadores A y B se cruzan.
FRAME:partial_occlusion:El jugador queda parcialmente oculto.
FRAME:complete_occlusion:No es visible durante varios fotogramas.
FRAME:exit_reentry:Sale del encuadre y vuelve a entrar.
FRAME:scene_change:Existe un cambio visible de plano o cámara.
FRAME:zoom_change:La escala cambia por un acercamiento o alejamiento.
FRAME:similar_uniforms:Las camisetas tienen colores visualmente similares.
```

Nunca se debe registrar un caso difícil basándose únicamente en un score ReID,
un gap del tracker o una propuesta heurística.

## Estados de anotación

Cada fotograma planificado debe quedar en uno de estos estados:

- `pending`: pendiente de revisión;
- `reviewed`: revisado exhaustivamente;
- `not_applicable`: no se puede usar con una justificación registrada.

Las observaciones individuales pueden ser:

- `confirmed`: identidad humana resuelta;
- `unknown`: identidad no identificable;
- `pending`: identidad ambigua que requiere revisión posterior;
- `ignored`: el recuadro o límite no es fiable y no entra en evaluación.

Una identidad confirmada debe ser estable entre fotogramas. Si no puede
demostrar visualmente que dos apariciones son la misma persona, use identidad
desconocida o pendiente; no invente un ID.

## Reglas de cajas

- `player`: jugador de campo.
- `goalkeeper`: portero, aunque se recomienda conservar una identidad separada
  por persona.
- `referee`: árbitro.
- `ball`: balón completo, no una línea ni una mancha cercana.
- `other`: otra entidad que no corresponde a las categorías anteriores.

Dibuje desde el borde visible del objeto hasta el borde visible. Los recuadros
parcialmente visibles deben seguir el objeto visible; no extrapole partes
completamente ocultas por otras personas. Una caja puede ir fuera de una
ventana 2D solo si se cumple el límite real de la imagen; el validador rechaza
coordenadas fuera del fotograma.

No puede haber dos objetos con la misma identidad en un fotograma. Los IDs de
anotación deben ser únicos dentro de cada fotograma; pueden reutilizarse entre
fotogramas solo si representan el mismo recuadro seguido manualmente.

## Eventos de trayectoria

Los eventos válidos son `appearance`, `disappearance`, `occlusion_start`,
`occlusion_end`, `crossing` y `camera_change`. Deben atribuirse a una
identidad humana confirmada y incluir notas cuando ayuden a interpretar el
evento.

Un hueco de ByteTrack no demuestra por sí solo una salida. Un inicio de
oclusión requiere observar la pérdida visual y un fin de oclusión requiere
observar la separación posterior.

## Control de calidad

Auditar un archivo humano:

```powershell
matchvision-quality-control-ground-truth `
  --annotations identidad-ground-truth.json `
  --sampling-plan outputs\ground-truth\amateur-football-sampling-plan-v1.json `
  --output outputs\ground-truth\quality-report.json
```

El informe detecta:

- campos obligatorios y esquemas inválidos;
- IDs duplicados o inconsistentes;
- tipos contradictorios de una identidad;
- cajas fuera de límites;
- fotogramas pendientes o sin revisión;
- estados no aplicables sin justificación;
- observaciones desconocidas, ambiguas o ignoradas;
- eventos incompatibles;
- plan y video con hashes o metadatos distintos;
- ventanas densas sin cobertura evaluable.

La CLI devuelve código distinto de cero si el conjunto no está listo.

## Congelación

Solo puede congelarse una exportación que tenga:

- al menos los fotogramas planificados correctamente revisados;
- identidades resueltas en las ventanas incluidas;
- un plan vinculado al mismo video;
- una huella interna de integridad.

```powershell
matchvision-freeze-identity-set `
  --input identidad-ground-truth.json `
  --output outputs\ground-truth\amateur-football-ground-truth-frozen-v1.json
```

La congelación no corrige mala calidad humana. Después de congelar, no se debe
modificar el archivo ni las anotaciones. Los JSON congelados deben conservarse
separados de predicciones, métricas, videos y soportes automáticos.
