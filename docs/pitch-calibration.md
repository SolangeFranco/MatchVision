# Calibración geométrica de cancha

## Alcance implementado

`matchvision.pitch.fit_pitch_calibration` ajusta una homografía mediante
correspondencias 2D entre píxeles de una imagen y puntos del plano de cancha.
Usa `cv2.findHomography` con RANSAC, conserva la máscara de inliers y calcula
el RMSE de reproyección en las unidades de cancha declaradas. La API puede
proyectar puntos y usar el centro inferior de una caja como aproximación al
punto de contacto de un jugador con el suelo.

La calibración es manual, está asociada a una imagen de tamaño y vista
concretos, y modela un plano. No detecta líneas ni esquinas automáticamente,
no decide dimensiones reglamentarias, no corrige movimiento de cámara y no
interpela entre planos. No hay un anotador visual ni un CLI de calibración
todavía; se prepara un conjunto de correspondencias verificadas como datos de
entrada del módulo.

## Uso desde Python

Selecciona al menos cuatro referencias visibles y no colineales. Cada punto de
imagen debe corresponder al mismo punto de la cancha en coordenadas 2D:

```python
from matchvision.pitch import fit_pitch_calibration

calibration = fit_pitch_calibration(
    image_points=[
        (320, 180),
        (1510, 205),
        (1800, 860),
        (180, 900),
        (1000, 500),
    ],
    pitch_points=[
        (0, 0),
        (105, 0),
        (105, 68),
        (0, 68),
        (52.5, 34),
    ],
    image_size=(1920, 1080),
    units="m",
    scale_verified=False,
    ransac_reprojection_threshold=1.0,
)

position = calibration.project_player_footpoint((900, 400, 940, 520))
report = calibration.to_dict()
```

Los valores de ejemplo son **ilustrativos**, no referencias medidas del video
del proyecto ni una calibración válida para este. Sustitúyelos por coordenadas
revisadas del fotograma y dimensiones comprobadas del terreno. El umbral RANSAC
y `reprojection_rmse` se expresan en las unidades de cancha indicadas. El
índice de referencia opcional es cero-based y ayuda a documentar el fotograma
calibrado.

## Interpretación y límites

- `scale_verified=False` significa que las coordenadas proyectadas usan las
  unidades declaradas, pero la escala física no está confirmada. No se deben
  reportar metros, velocidad o distancia física como mediciones válidas.
- `scale_verified=True` solo debe usarse cuando quien prepara las
  correspondencias verificó las dimensiones físicas y su correspondencia con
  el plano del suelo; el software no lo verifica por sí mismo.
- El RMSE describe el ajuste a los puntos **inlier usados en el ajuste**. Un
  error pequeño no demuestra que los puntos manuales sean correctos ni que la
  cancha real corresponda a las dimensiones elegidas.
- Jugadores de pie pueden tener el pie oculto por oclusión o desenfoque; el
  centro inferior de una caja no garantiza contacto real con el suelo. No
  proyectes el centro del torso como posición de cancha.
- Para cámara móvil se necesita estimar/actualizar una homografía por vista o
  por fotograma y detectar cortes. No reutilices una matriz estática después
  de paneos, zoom o cambio de plano.
- Puntos de balón elevados, jugadores saltando y otros objetos fuera del suelo
  no obedecen la misma homografía plana.

## Validación pendiente

Las pruebas usan geometría sintética para comprobar transformaciones,
correspondencias inconsistentes, rechazo RANSAC, errores de reproyección,
límites de imagen y proyección del punto inferior. No hay anotaciones de puntos
de cancha verificadas para el video local, por lo que todavía no existe una
validación real de precisión geométrica. El paso siguiente es crear una
herramienta de selección/revisión de referencias sobre el fotograma, registrar
sus dimensiones y procedencia, y comparar puntos proyectados con referencias
reservadas que no se usaron para ajustar la matriz.
