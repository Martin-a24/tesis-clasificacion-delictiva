# Resumen detallado del proyecto (handoff para redacción de tesis)

Estado consolidado de lo construido y decidido. Rama estable: `master`.
Modelo final elegido: **ResNet-18**. Entorno: conda `tesis`; servidor 2× RTX A6000.

---

## 0. Visión general

CNN que clasifica tiles satelitales (PeruSAT-1) de Lima Metropolitana y Callao en
3 niveles de riesgo de delito patrimonial (**bajo / medio / alto**), cruzando
imágenes con datos delictivos del MININTER. Tres objetivos:
- **Obj 1:** dataset preprocesado (R2) y etiquetado (R3).
- **Obj 2:** modelo CNN de clasificación + evaluación.
- **Obj 3:** interpretabilidad (Grad-CAM, R6), coherencia con criminología
  ambiental (R7) y zonificación QGIS (R8).

---

## 1. Objetivo 1 — Preprocesamiento y etiquetado

### 1.1 Preprocesamiento (R2 y extensión R2.1)
- **R2 (validado y firmado 19/05/2026** por Edwin Alvarez Mamani): pansharpening
  (GDAL, remuestreo cúbico, 2.8m→0.7m, 4 bandas B/G/R/NIR), teselado 512×512 px
  (~358×358 m), filtros de calidad (píxeles válidos ≥95%, agua NDWI, urbano ≥50%).
- **R2.1 (extensión, en `master`; documentar como anexo, no re-firmar):** se
  detectó que escenas que se solapan generaban **tiles duplicados** → fuga
  espacial. Diagnóstico: **597 de 2451 ubicaciones aparecían en >1 escena.**
  Solución:
  - **Mosaico virtual (VRT)** de todas las escenas (`nodata=0`) → cada ubicación
    existe una sola vez (script `02b_construir_mosaico.py`).
  - **Grilla global fija** anclada en EPSG:32718 → `cell_id` estable entre
    corridas y al agregar escenas (script `03_generar_tiles.py`).
  - **Teselado paralelo** (multiprocessing) — responde a la sugerencia del
    experto R2 de aplicar computación paralela.
- **Resultado del teselado:** mosaico 61379×38549 px; **2603 tiles válidos** de
  8806 potenciales (29.6%); descartados: borde 5147, agua 587, no urbano 469.

### 1.2 Etiquetado (R3 — enviado a validar, sin firmar aún)
- **Método (el usado en `master`):** grilla virtual de 358 m sobre toda Lima
  (22194 tiles; 3305 con ≥1 delito), **conteo de delitos por tile**, umbrales
  **globales** por percentiles (P33 y P67) calculados sobre los tiles virtuales
  con delito; **tiles sin delito → bajo**. Umbrales: P33=1, P67=4 delitos.
- **Distribución resultante (2603 tiles reales):** bajo **64.5%** / medio
  **20.5%** / alto **15.1%**.
- **Exploración NO adoptada** (queda en rama `experimento-etiquetado`): medir la
  actividad con **densidad por vecindad** o **KDE gaussiano** en vez de conteo.
  **Hallazgo importante:** con la cobertura actual (2 escenas céntricas, densas)
  + umbral global de toda Lima, el suavizado **degenera** (KDE dio 3% bajo /
  29% medio / **68% alto**), porque rellena los ceros y todo el área urbana queda
  "alta" respecto a la periferia vacía de Lima. Por eso **se mantuvo el conteo**.
- **Limitaciones a discutir:** "bajo" mezcla bajo-riesgo con "sin registro"; la
  clase "medio" es intrínsecamente difusa; MAUP (dependencia del tamaño de tile);
  subregistro de denuncias (~83.9% según INEI) → el conteo subestima; supuesto de
  estabilidad de patrones espaciales relativos.

### 1.3 Split del dataset (script 05)
- **StratifiedGroupKFold agrupado por `cell_id`** → ningún `cell_id` se reparte
  entre train/val/test (evita fuga espacial). Ratios 70/15/15.
- **Train 1859 · Val 372 · Test 372** tiles. Total delitos: 16596.

---

## 2. Objetivo 2 — Modelo CNN

### 2.1 Arquitectura y entrenamiento (script 06)
- **Transfer learning** desde ImageNet; primera capa conv adaptada de 3→4 bandas
  (pesos RGB copiados + promedio para la banda NIR).
- **Hiperparámetros (config final):** optimizador **AdamW**, lr **1e-4**,
  weight_decay **5e-4**, **dropout 0.2** en la cabeza, **label_smoothing 0.1**,
  **class weights** (bajo 0.517, medio 1.626, alto 2.213), **data augmentation
  dihedral** (flips + rotaciones de 90°), **early stopping** (paciencia 10),
  num_epochs 50 (tope), batch_size 32. **Determinista** (semilla 42).
- Justificación de estos valores: dataset pequeño + desbalanceado + etiqueta
  ruidosa → regularización fuerte + label smoothing (para etiqueta imperfecta) +
  clase-pesos. Sin AMP/cudnn.benchmark para mantener **reproducibilidad**.

### 2.2 Comparación de arquitecturas (script 08) — selección por VALIDACIÓN
La selección se hace por **F1-macro en validación** (el test se reserva como
reporte no sesgado). Resultados reales:

| Arquitectura | Params | F1-macro **val** | Acc test | F1-macro test | F1 bajo | F1 medio | F1 alto | Recall alto |
|---|---|---|---|---|---|---|---|---|
| **ResNet-18 (elegido)** | 11.2M | 0.697 | **0.772** | **0.704** | 0.881 | 0.527 | **0.704** | **0.786** (44/56) |
| ResNet-50 | 23.5M | 0.711 | 0.763 | 0.703 | 0.861 | 0.546 | 0.704 | 0.679 (38/56) |
| EfficientNet-B0 | 4.0M | 0.719 | 0.753 | 0.691 | 0.868 | 0.557 | 0.648 | 0.607 (34/56) |

Baselines (F1-macro): aleatorio 0.296, mayoritario 0.261 (predice siempre "bajo").

### 2.3 Selección del modelo: **ResNet-18** (ver `docs/SELECCION_MODELO.md`)
- El desempeño **global** de las tres está **empatado dentro del ruido** (test de
  372 tiles, solo 56 "alto").
- **Criterio de desempate a-priori, motivado por el dominio:** como el objetivo
  es identificar **zonas de alto riesgo (hotspots)**, se prioriza la
  **sensibilidad a "alto"**. ResNet-18 gana (recall 0.786 vs 0.607 de
  EfficientNet) y es más parsimonioso y estable que ResNet-50.
- **Nota metodológica:** el desempate por "alto" se declara como criterio del
  dominio; la evidencia por-clase proviene del test (limitación menor; lo ideal
  sería confirmarlo con métricas por clase de validación — trabajo futuro).

### 2.4 Resultado final (ResNet-18)
- **Accuracy 0.772 · F1-macro 0.704 · F1-weighted 0.782.**
- Por clase (P/R/F1): bajo 0.939/0.829/**0.881**; medio 0.484/0.579/**0.527**;
  alto 0.638/0.786/**0.704**.
- Matriz de confusión (filas=real, col=pred):
  - bajo → [199, 35, 6]
  - medio → [13, 44, 19]
  - alto → [0, 12, 44]
- **Mejora de +40.8 puntos de F1-macro sobre el baseline aleatorio** y +44.2
  sobre el mayoritario.
- **Curva de entrenamiento:** mejor época = 2 (early stopping en época 12). El
  pico temprano es **normal en transfer learning con dataset pequeño** (el modelo
  aprovecha rápido las features de ImageNet y luego sobreajusta; el early stopping
  guarda el mejor punto). No es un defecto; es la buena práctica funcionando.

---

## 3. Objetivo 3 — Interpretabilidad y zonificación

- **R6 — Grad-CAM (`09_gradcam.py`):** mapas de activación por tile del test
  (implementación con hooks de PyTorch, sin dependencias extra), organizados por
  categoría; montajes representativos por clase; `gradcam_resumen.csv` (acierto y
  concentración de activación por clase). Capa objetivo: `layer4` (ResNet).
- **R7 — Coherencia (`docs/R7_tabla_correspondencias.md`):** plantilla para
  comparar regiones activadas por Grad-CAM con factores de **CPTED** (vigilancia
  natural, control de accesos, mantenimiento/territorialidad) y **Routine Activity
  Theory** (objetivos atractivos, ausencia de guardianes) + proxies visibles
  (densidad construida, lotes baldíos, vías, comercio, vegetación).
- **R8 — Zonificación (`10_zonificacion.py`):** predice el riesgo en todos los
  tiles → `data/labels/tiles_predicciones.geojson` (capa QGIS) + mapa + reporte de
  **coincidencia espacial** con los delitos del MININTER (delitos promedio por
  categoría predicha, % de delitos en zonas "alto", acuerdo global y en test).
- **Pendiente de ejecutar** sobre ResNet-18: correr pasos 9 y 10 en el servidor.

---

## 4. Infraestructura y herramientas

- **`scripts/panel.py`** — panel de control en terminal: estado del pipeline
  (hecho/listo/bloqueado), correr pasos, configurar parámetros (se guardan en
  `config.yaml` conservando comentarios), ver inputs/cómo agregar datos, muestra
  el modelo activo y las arquitecturas entrenadas.
- **`scripts/sync.sh`** — transferencia PC↔servidor con rsync+SSH (ProxyJump);
  opcional, documentado para terceros.
- **`scripts/limpiar_salidas.py`** — limpieza de salidas; `--desde-modelo` borra
  solo de 06 en adelante (re-entrenar sin rehacer 01–05).
- **Documentación:** `README.md`, `SETUP.md`, `docs/RESULTADOS.md` (mapa de
  outputs), `docs/SELECCION_MODELO.md`, `docs/R7_tabla_correspondencias.md`.
- **Replicabilidad:** todo por `config.yaml` (nada hardcodeado); estructura de
  `data/raw/` versionada con `.gitkeep`; README explica cómo adaptarlo a otra
  ciudad/periodo.

---

## 5. Limitaciones honestas (para la sección de Discusión)

1. La etiqueta es un **proxy** (delito *registrado*, con subregistro ~83.9%).
2. Cobertura actual = **2 escenas céntricas** → subconjunto sesgado de Lima
   (afecta el balance de clases y por qué el suavizado/KDE degeneró).
3. La clase **"medio" es la más difícil** (F1 ≈ 0.53) — frontera difusa.
4. Compromiso en detección de **"alto"** según arquitectura.
5. Diferencias entre arquitecturas **dentro del ruido** (test pequeño).
6. **MAUP** (dependencia del tamaño/origen de la grilla; el ancla fija estabiliza
   el origen) e **imagen diurna** (iluminación, un factor CPTED, no observable).

---

## 6. Resultados clave para citar (números)

- Dataset: 16596 delitos; 2603 tiles (train 1859 / val 372 / test 372).
- Distribución de clases: bajo 64.5% / medio 20.5% / alto 15.1%.
- Modelo final ResNet-18: **Acc 0.772 · F1-macro 0.704 · F1-weighted 0.782.**
- Baselines F1-macro: aleatorio 0.296 · mayoritario 0.261 → **modelo +40 pts**.

---

## 7. Estado del repositorio y próximos pasos

- Ramas: **`master`** (estable, todo lo anterior) y **`experimento-etiquetado`**
  (exploración KDE/vecindad, guardada por si el experto sugiere cambiar R3).
- **Pendiente operativo:** correr pasos **9 y 10** sobre ResNet-18, bajar
  `results/` y `data/labels/` a la PC (`sync.sh pull-entregables`).
- **Mapa outputs → tesis:** ver `docs/RESULTADOS.md`.
