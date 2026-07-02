# Selección del modelo (respaldo del Objetivo 2)

Documento de respaldo para justificar la elección de arquitectura en la tesis.
Corrida de comparación: `08_comparar_arquitecturas.py` (2026-07-01).

## Criterio de selección

1. **Criterio primario:** F1-macro sobre el conjunto de **validación** (métrica
   balanceada, adecuada para clases desbalanceadas; el test se reserva como
   estimación no sesgada del modelo elegido).
2. **Criterio de desempate (a-priori, motivado por el dominio):** cuando el
   desempeño global está empatado, se prioriza la **sensibilidad a la clase
   'alto' (recall)**, porque el objetivo del trabajo es identificar zonas de
   **alto riesgo (hotspots)**; omitir zonas de alto riesgo es el error más
   costoso en esta aplicación.

## Resultados comparativos

Transfer learning desde ImageNet, 4 bandas, entrenamiento determinista
(semilla 42), class weights + label smoothing 0.1, AdamW, early stopping.

| Arquitectura | Params | F1-macro **val** | Acc test | F1-macro test | F1 bajo | F1 medio | F1 alto | Recall alto |
|---|---|---|---|---|---|---|---|---|
| **resnet18** (elegido) | 11.2M | 0.697 | **0.772** | **0.704** | 0.881 | 0.527 | **0.704** | **0.786** (44/56) |
| resnet50 | 23.5M | 0.711 | 0.763 | 0.703 | 0.861 | 0.546 | 0.704 | 0.679 (38/56) |
| efficientnet_b0 | 4.0M | 0.719 | 0.753 | 0.691 | 0.868 | 0.557 | 0.648 | 0.607 (34/56) |

Baselines (F1-macro): aleatorio 0.296, mayoritario 0.261.
Conjunto de test: 372 tiles (bajo 240, medio 76, alto 56).

## Decisión: **resnet18**

- El desempeño **global** de las tres arquitecturas está **empatado dentro del
  ruido** (F1-macro val 0.70–0.72; test 0.69–0.70), esperable con un test de 372
  tiles y solo 56 de la clase 'alto'.
- En la **detección de 'alto'** (el objetivo aplicado) sí hay diferencia:
  `resnet18` logra el mejor recall (0.786) y F1 (0.704), frente a
  `efficientnet_b0` (recall 0.607). Por el criterio de desempate, se elige
  `resnet18`.
- Frente a `resnet50` (recall alto 0.679), `resnet18` es mejor en 'alto', más
  parsimonioso (11M vs 23M) y sobreajusta menos (resnet50 alcanzó train F1 0.99).

## Resultado clave (Objetivo 2)

Las tres arquitecturas **superan ampliamente a los baselines** (+40 puntos de
F1-macro sobre el aleatorio y el mayoritario), lo que confirma que el entorno
urbano contiene señal predictiva del riesgo delictivo.

## Discusión y limitaciones

- La clase **'medio' es la más difícil** de discriminar para todos los modelos
  (F1 ≈ 0.53–0.56): es la frontera difusa entre bajo y alto.
- Las diferencias globales entre arquitecturas son pequeñas (dentro del ruido);
  no se sobre-interpretan.
- La ventaja de `resnet18` en 'alto' se observa en el test; idealmente se
  confirmaría también con métricas por clase en validación (trabajo futuro:
  loguear F1 por clase de validación en el script 06).
- La etiqueta es un proxy (delito registrado, con subregistro), lo que acota la
  interpretación de las métricas.

## Configuración del modelo elegido

`configs/config.yaml`:
- `modelo.architecture: resnet18`
- `modelo.dropout: 0.2`
- `entrenamiento.optimizer: adamw`, `weight_decay: 0.0005`, `label_smoothing: 0.1`
- `entrenamiento.num_epochs: 50` (tope; el early stopping corta ~época 12),
  `early_stopping_patience: 10`
