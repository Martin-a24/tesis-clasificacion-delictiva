# R7 — Tabla de correspondencias: Grad-CAM vs criminología ambiental

**Objetivo 3 / R7.** Analizar la coherencia entre las regiones que el modelo
activa (Grad-CAM, R6) y los factores ambientales documentados en la literatura
(CPTED — *Crime Prevention Through Environmental Design*; RAT — *Routine
Activity Theory*).

## Cómo llenar esta tabla

1. Corre `python scripts/09_gradcam.py` (genera los mapas y montajes).
2. Revisa, por categoría de riesgo:
   - `results/gradcam/montage_<categoria>.png` (muestras representativas),
   - `results/gradcam/<categoria>/*.png` (todos los tiles),
   - `results/gradcam/gradcam_resumen.csv` (dónde se concentra la activación).
3. Identifica **patrones visuales recurrentes** dentro de las regiones resaltadas
   (lo que el modelo "miró") y asócialos a un factor de la teoría.
4. Marca si la asociación es **coherente** (lo esperado por la teoría) o
   **divergente** (el modelo mira algo que la teoría no predice, o ignora algo
   que sí predice). Cita 2–3 tiles de ejemplo.

> Nota de honestidad metodológica: Grad-CAM muestra *correlación* aprendida, no
> causalidad. La etiqueta es delito **registrado** (proxy de riesgo). Documenta
> ambas limitaciones al interpretar.

## Tabla (plantilla — completar con la evidencia)

| Factor (teoría) | Descripción | Proxy visible en imagen satelital (0.7 m) | ¿Aparece en regiones activadas? (categoría) | Coherente / Divergente | Tiles de ejemplo |
|---|---|---|---|---|---|
| **CPTED — Vigilancia natural** | Espacios visibles, con "ojos en la calle" | Calles anchas, ventanas a la vía, plazas abiertas | _por completar_ | _por completar_ | _por completar_ |
| **CPTED — Control de accesos** | Delimitación de entradas/salidas | Rejas, calles cerradas, condominios amurallados | | | |
| **CPTED — Mantenimiento / territorialidad** | Cuidado del espacio (teoría ventanas rotas) | Lotes baldíos, basura, techos deteriorados, áreas verdes descuidadas | | | |
| **CPTED — Iluminación** (limitado en imagen diurna) | Visibilidad nocturna | (Poco observable en imagen diurna; documentar como limitación) | | | |
| **RAT — Objetivos atractivos** | Concentración de bienes/personas | Mercados, centros comerciales, paraderos, alta densidad | | | |
| **RAT — Ausencia de guardianes** | Falta de control formal/informal | Zonas descampadas, periferia, baja densidad construida | | | |
| **RAT — Convergencia / flujo** | Cruce de víctimas y ofensores | Vías principales, nodos de transporte | | | |
| **Tejido urbano — Densidad / informalidad** | Morfología del asentamiento | Trama irregular, autoconstrucción, alta densidad de techos | | | |
| **Tejido urbano — Vías y conectividad** | Estructura vial | Avenidas, intersecciones, accesibilidad | | | |
| **Tejido urbano — Vegetación / espacios abiertos** | Cobertura vegetal | Parques, áreas verdes, descampados | | | |

## Síntesis a redactar (a partir de la tabla)

- **Coherencias principales**: factores donde Grad-CAM coincide con la teoría.
- **Divergencias**: dónde el modelo mira algo inesperado (¿artefacto, sesgo de
  la etiqueta, o hallazgo?).
- **Limitaciones**: subregistro de denuncias, etiqueta como proxy, imagen diurna,
  resolución, cobertura del dataset.
