# Decisiones de diseño

Reglas que sigue el código y que las pruebas verifican. Los números de ítem (#22, #40, etc.) son los de la bitácora del equipo; el porqué de cada decisión y su evidencia están en el README ("Decisiones técnicas y su justificación").

- **Solo scripts `.py`**: no se usan notebooks (`.ipynb`) para el código
  core del pipeline. Notebooks solo se permiten como exploración puntual
  fuera de `src/`, nunca versionados.
- **Bounding boxes estandarizados**: toda coordenada de detección o
  anotación se representa como `[x_min, y_min, x_max, y_max]` en píxeles
  absolutos de la imagen original.
- **Enfoque híbrido obligatorio**: `src/preprocessing/` (OpenCV) siempre
  precede a `src/models/` (YOLO26), en este orden: corrección de imagen
  (dewarping y luego CLAHE, `correct_image()`) -> segmentación automática
  de góndola/repisas ->
  recorte de ROI. La segmentación trabaja sobre la imagen ya corregida.
  No se reemplaza el pre-procesamiento clásico por el modelo de inferencia.
- **ROI automática**: el sistema segmenta la góndola y sus repisas en
  `src/preprocessing/`, antes de YOLO26. Ningún usuario dibuja ni configura
  ROI por cámara, así que el código no debe depender de una ROI cargada de
  configuración. Flujo: `correct_image()` -> `segment_gondola()` ->
  `crop_roi(imagen_corregida, segmentacion.roi)`. Las detecciones vuelven
  a la imagen original con `RoiCrop.to_original()` y luego
  `CorrectedImage.to_original()`.
- **Niveles de repisa híbridos** (decidido 2026-10-01, ítem #22): los
  niveles que mide la disponibilidad salen de `shelf_levels()`
  (`src/postprocessing/shelf_levels.py`), que combina las filas de
  productos detectados (sus bases marcan la repisa aunque esté inclinada)
  con `segmentacion.shelf_lines` (aportan las repisas vacías), como
  franjas inclinadas. Flujo: detecciones en coordenadas de la imagen
  corregida (`RoiCrop.to_original()`) -> `shelf_levels(cajas,
  [astuple(l) for l in segmentacion.shelf_lines], segmentacion.roi)` ->
  `compute_availability(cajas, niveles)`. `GondolaSegmentation.shelves`
  (rectángulos horizontales) queda para recortar la ROI, no para medir:
  con la cámara en ángulo mezcla repisas vecinas.
- **Profundidad constante**: toda góndola se asume de **25 cm** de
  profundidad (valor fijo, dentro del rango real de 20-30 cm). No se
  estima profundidad (sin modelos de profundidad ni visión estéreo): el
  análisis es 2D sobre la vista frontal. Si un cálculo necesita la
  profundidad (volumen del Nivel 1), usa esa única constante global de
  25 cm, nunca un valor por góndola o por cámara.
- **Detector YOLO26** (`ultralytics>=8.4.0`, pesos `yolo26*.pt`). En
  góndolas densas/apiladas inferir con `nms=False` (activa la cabeza sin
  NMS; por defecto Ultralytics aplica NMS igual), `max_det` >= 1000 (el
  default 300 trunca góndolas llenas) e `imgsz=1280`. Lo mismo al entrenar
  y validar (`nms=False`, `max_det=1000`): si no, Ultralytics mide la
  cabeza con NMS y elige `best.pt` con otro modelo. La inferencia pasa por
  `ProductDetector` (`src/models/detector.py`), que ya fija estos valores;
  el entrenamiento, por `scripts/train_detector.py` (ver
  `docs/training.md`).
- **Pipeline modular**: `preprocessing -> models -> postprocessing ->
  reporting`. Cada etapa vive en su propio paquete y no debe importar
  directamente detalles internos de otra etapa; se comunican mediante
  estructuras de datos simples (arrays, dicts, dataclasses).
- **Umbral de quiebre de stock**: se alerta cuando el espacio **libre** es
  `>= 30%` del frente de la góndola (disponibilidad `<= 70%`; MVP Nivel
  2, ver README). El espacio libre se mide como cobertura a lo largo de
  cada repisa (en X, sobre las franjas de `shelf_levels()`): huecos de
  al menos un ancho de producto, sin medir la altura de los productos,
  en la góndola acotada a los productos detectados
  (`src/postprocessing/availability.py`, decidido 2026-09-30).
- **Quiebre por góndola completa** (cambio de alcance del 2026-10-19,
  indicado por el profesor): la alerta y el % de espacio libre que el
  sistema **informa** (JSON, vista, reportes, alertas, app web) son solo
  los de la góndola completa. Las repisas de `shelf_levels()` siguen
  siendo el método interno para medir (sumar la cobertura a lo largo de
  cada repisa), pero no se informa alerta ni porcentaje por repisa. Ver
  ítem #40.
- **Licencias de datos**: SKU-110K es solo para uso académico y no
  comercial, y su licencia prohíbe distribuir, mostrar a terceros o
  **modificar** sus fotos (crear obras derivadas). Se usa para entrenar
  y evaluar (confirmado por el profesor el 2026-10-19), nunca para
  generar imágenes sintéticas (vistas en ángulo,
  distorsión de lente) ni para demos ante Walmart u otros externos: para
  eso, datasets que lo permitan (Kaggle supermarket-shelves CC0, SHARD y
  Roboflow CC BY 4.0). Ver `src/datasets/sources.py` y
  `docs/datasets.md`.
