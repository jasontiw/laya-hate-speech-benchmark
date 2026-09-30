# Benchmark Local de Detección de Discurso de Odio

[English](README.md) | **Español**

Un experimento pequeño, local y reproducible que mide cómo rinde **Laya** en la
detección de discurso de odio y lo compara contra clasificadores clásicos y
especializados sobre el **mismo dataset público** y las **mismas filas de test**.

Pregunta de investigación:

> ¿Cómo rinde Laya en un dataset conocido de discurso de odio frente a otros
> clasificadores locales, con los mismos ejemplos y métricas — y qué **formulación de
> Laya** (choice de 3 clases, choice binario, `noul`, definiciones explícitas) da el
> mejor equilibrio recall / precisión / coste?

Es un benchmark de investigación, **no** un sistema de moderación en producción. Se
reduce a un comando y a un conjunto de artefactos generados.

---

## El hallazgo sobre el que está construido

Los modelos no se ordenan simplemente por calidad: **discrepan sobre qué cuenta como
discurso de odio**. Sobre las mismas filas de test, un modelo es preciso y casi nunca
dispara; otro dispara a menudo y captura mucho más odio a un coste alto de precisión.
La accuracy agregada esconde esto, así que el benchmark reporta métricas hate-vs-rest,
PR-AUC, un punto de operación elegido en validación, coverage, calibración e intervalos
bootstrap junto a la tabla habitual.

---

## Modelos comparados

Nueve configuraciones, de tres enfoques conceptualmente distintos:

| Clave | Modelo | Tarea | Régimen de entrenamiento | ¿Entrenado con el train de Davidson? |
| --- | --- | --- | --- | --- |
| `majority` | Baseline de clase mayoritaria | 3 clases | Baseline | No |
| `tfidf` | TF-IDF + Regresión Logística | 3 clases | Supervisado (ajustado aquí) | Sí |
| `tfidf_balanced` | Igual, `class_weight="balanced"` | 3 clases | Supervisado (ajustado aquí) | Sí |
| `laya` | Laya, choice 3 clases — **L1** | 3 clases | Zero-shot | No |
| `laya_semantic` | Laya, choice 3 clases, definiciones explícitas — **L4** | 3 clases | Zero-shot | No |
| `laya_binary` | Laya, choice de 2 opciones odio/no-odio — **L2** | odio vs resto | Zero-shot | No |
| `laya_noul` | Laya, `noul` P(odio) — **L3** | odio vs resto | Zero-shot | No |
| `laya_finetuned` | Laya, pregunta L1, **fine-tuned** | 3 clases | Supervisado (RLCD, ajustado aquí) | Sí |
| `hatexplain` | `Hate-speech-CNERG/bert-base-uncased-hatexplain` | 3 clases | Preentrenado externo | No |

`laya_finetuned` es **opt-in** (`run_benchmark.py --include-finetuned`): es un checkpoint
de ~0,8 GB producido localmente en la fase 2A, así que un clon limpio sigue reproduciendo
el benchmark zero-shot con un solo comando. Ver [docs/finetuning.md](docs/finetuning.md).

> **Este benchmark compara enfoques de clasificación end-to-end bajo sus regímenes de
> entrenamiento naturales; no es una comparación controlada arquitectura-vs-arquitectura.**

En particular, `tfidf` se ajusta con el split de entrenamiento in-domain mientras que
`laya` y `hatexplain` se usan tal como se publican. Una puntuación mayor de un modelo
supervisado es esperable y no es evidencia de que su arquitectura sea mejor.

Dataset: **Davidson** (`tdavidson/hate_speech_offensive`), 24.783 tweets en inglés,
tres clases — `hate speech` (0), `offensive language` (1), `neither` (2).

---

## Inicio rápido

Requiere Python 3.10+ y (para la build de CUDA) una GPU NVIDIA.

```powershell
cd laya-hate-speech-benchmark

# 1. Entorno — instala primero la build de PyTorch que corresponda a tu hardware
uv venv --python 3.12
uv pip install torch --index-url https://download.pytorch.org/whl/cu126   # o .../cpu
uv pip install -r requirements.txt

# 2. Reproduce todo
.\.venv\Scripts\python.exe run_benchmark.py
```

En macOS/Linux usa `.venv/bin/python run_benchmark.py`.

La primera ejecución descarga el dataset (~2,5 MB) y los checkpoints (Laya English,
HateXplain BERT); después quedan en la caché de Hugging Face.

### Flags útiles

```bash
python run_benchmark.py --list-models               # imprime las claves configuradas y sale
python run_benchmark.py --limit 200                 # smoke test (marcado como tal)
python run_benchmark.py --models laya,laya_binary   # solo algunas configuraciones
python run_benchmark.py --include-finetuned          # + la fila de Laya fine-tuned (fase 2A)
python run_benchmark.py --config my_experiment.yaml # configuración alternativa
```

Cada ejecución se gobierna desde [`config.yaml`](config.yaml); ese archivo se copia tal
cual en `results/experiment_config.json`.

---

## Artefactos de salida

```
results/
├── dataset_metadata.json          # fuente, SHA-256, filas, distribución de clases
├── experiment_config.json         # config + hardware/runtime + revisiones resueltas
├── split.json                     # semilla, conteos y los ids exactos train/validación/test
├── predictions.csv                # una fila por tweet, todas las predicciones + probabilidades
├── metrics.json                   # métricas, IC, puntos de operación, calibración, coverage
├── model_comparison.csv           # la tabla comparativa principal
├── operating_points.csv           # decisión natural vs umbral elegido en validación
├── coverage.csv                   # coverage / accuracy / recall de odio vs confianza
├── validation_scores.csv          # scores de odio por modelo en validación (sin texto de tweets)
├── error_analysis.csv             # errores agrupados por categoría (gold -> predicho)
├── confusion_matrix_<model>.png
├── pr_curve_hate_vs_rest.png      # precisión/recall hate-vs-rest, una línea por modelo
└── report.md

report/
└── benchmark_report.md            # el entregable final
```

`benchmark_report.md` cubre las 16 secciones del PRD y, dentro de ellas, el material de
la v1.1: regímenes de entrenamiento, IC bootstrap, variantes binarias, latencia con
batching, coverage, calibración y una lista de huecos conocidos.

---

## Metodología

- **Manejo del dataset.** Descarga programática, verificación por SHA-256, nunca se
  edita, se conservan las etiquetas originales.
- **Split.** Estratificado por clase y agrupado para que los duplicados no crucen
  fronteras: **train / validación / test**. La validación se recorta del lado de
  entrenamiento, así el test no cambia y los resultados siguen siendo comparables entre
  corridas. El normalizador, en orden:

  ```python
  def normalize(text):
      value = text
      value = value.lower()                                  # minúsculas
      value = re.sub(r"https?://\S+|www\.\S+", " ", value)   # URLs fuera
      value = re.sub(r"@\w+", " ", value)                    # @menciones fuera
      value = re.sub(r"^\s*rt\b[:\s]+", "", value, re.I)     # "rt" inicial fuera
      value = re.sub(r"[^\w\s]", " ", value)                 # puntuación fuera (opcional; apagado por defecto)
      value = re.sub(r"\s+", " ", value).strip()             # espacios colapsados y recortados
      return value
  ```

- **Formulaciones de Laya.** Choice de 3 clases, choice binario, `noul` y un prompt con
  definiciones explícitas, todas zero-shot. Ninguna se selecciona usando el test.
- **Mapeo de HateXplain.** Su `id2label` se lee de la config del modelo y se mapea a las
  etiquetas canónicas (`hate speech -> hate speech`, `offensive -> offensive language`,
  `normal -> neither`). El mapeo crudo queda registrado en el reporte.
- **Métricas.** Accuracy, macro F1, precision/recall/F1 por clase y matrices de confusión
  para los modelos de 3 clases; precision/recall/F1 hate-vs-rest con FP/FN para todos.
- **Métrica de ranking.** **PR-AUC (average precision)** para odio vs resto. Con ~5,8 %
  de positivos, ROC-AUC es halagadora y la accuracy la domina la clase mayoritaria.
- **Punto de operación.** El umbral sobre `P(hate speech)` se **elige en validación** y se
  aplica sin cambios al test; el umbral óptimo en test se registra solo como cota superior.
- **Inferencia.** Donde hay ruta batcheada (`predict_batch` de Laya, HateXplain, TF-IDF),
  las predicciones salen de una pasada batcheada y **la latencia por ítem se mide aparte,
  sobre una muestra**, así throughput y latencia nunca se mezclan. Ambas rutas se
  contrastan y se reporta su acuerdo de etiquetas (los tamaños de batch pueden cambiar el
  resultado en coma flotante).
- **Incertidumbre.** **IC bootstrap** al 95 % (percentil) para accuracy, macro F1, hate F1
  y PR-AUC. Describe variabilidad, no significancia entre modelos (los splits son
  compartidos, así que la comparación es emparejada).
- **Calibración y coverage.** Brier, ECE y bins de fiabilidad para el score de odio,
  medidos **en validación**; más una tabla de coverage que muestra accuracy y recall de
  odio tras descartar las respuestas menos confiadas.
- **Presupuesto de tokens.** Tokens medios/máximos del estado y número de filas truncadas
  por modelo.

---

## Estructura del proyecto

```
laya-hate-speech-benchmark/
├── config.yaml               # ajustes del experimento (única fuente de verdad)
├── run_benchmark.py          # punto de entrada: python run_benchmark.py
├── requirements.txt
├── pyproject.toml
├── data/{raw,processed}/     # ignorado por git
├── docs/PRD.md               # la especificación que esto implementa
├── scripts/
│   ├── run_benchmark.py      # envoltorio fino (mismo punto de entrada)
│   ├── build_laya_finetune_data.py  # fase 2A: split -> JSONL de fine-tuning
│   ├── finetune_laya.py      # fase 2A: fine-tune RLCD en un solo dispositivo
│   └── render_report.py      # re-renderiza el reporte sin re-ejecutar modelos
├── src/
│   ├── config.py             # config tipada, etiquetas, tareas, regímenes
│   ├── dataset.py            # descarga, verificación, normalización, split agrupado 3 vías
│   ├── environment.py        # introspección de hardware/runtime
│   ├── metrics.py            # clasificación, ranking, punto de operación, bootstrap
│   ├── calibration.py        # Brier, ECE, bins de fiabilidad
│   ├── evaluation.py         # orquestación
│   ├── reporting.py          # gráficos + reporte Markdown
│   └── models/
│       ├── base.py           # interfaz: task, predict_one, predict_batch
│       ├── majority.py
│       ├── tfidf.py
│       ├── laya_model.py     # Laya: variantes zero-shot + montaje del checkpoint fine-tuned
│       └── hatexplain.py
└── results/, report/         # generados, ignorados por git
```

### Desviaciones respecto a la estructura propuesta en el PRD

- `src/models/laya.py` se llama `laya_model.py`, para que nunca pueda ensombrecer al
  paquete `laya` instalado al importar.
- El dataset se descarga como el CSV original de los autores en lugar de vía
  `huggingface/datasets`, para poder hashearlo directamente.
- El único `TF-IDF + LR` y el único prompt de Laya del PRD pasaron a ser **variantes**
  (dos ajustes de pesos; cuatro formulaciones de Laya), porque compararlos separa una
  propiedad del modelo de una propiedad de su prior de entrenamiento o de su prompt.
- El fine-tuning (fase 2A) está implementado como paso **opt-in**, documentado en
  [docs/finetuning.md](docs/finetuning.md); Detoxify **no** está implementado.

---

## Fase 2A — fine-tuning de Laya (opt-in)

El apartado 24 del PRD difiere el fine-tuning a la fase 2A. Aquí está implementado como
un pipeline de dos pasos que produce un checkpoint local, que el arnés evalúa después
exactamente igual que cualquier otra fila:

```powershell
.\.venv\Scripts\python.exe scripts\build_laya_finetune_data.py
.\.venv\Scripts\python.exe scripts\finetune_laya.py `
    --data data\processed\laya_finetune_train.jsonl `
    --eval-data data\processed\laya_finetune_validation.jsonl
.\.venv\Scripts\python.exe run_benchmark.py --include-finetuned
```

El fine-tuning reutiliza la pregunta L1 tal cual, entrena solo con el split de train
(16.130 filas, RLCD, 4 épocas) y nunca toca una fila de test. El test se puntúa una sola
vez, con `run_benchmark.py`, después del entrenamiento.

Por tanto se reportan dos regímenes de entrenamiento de Laya — **zero-shot** (los
checkpoints publicados) y **supervisado fine-tuned** (este local) — y nunca se mezclan.
El reporte lleva la dirección predicha antes de la corrida, para juzgarla contra la
medición en vez de recordarla a posteriori. Protocolo completo, coste medido y límites
honestos: [docs/finetuning.md](docs/finetuning.md).

---

## Huecos conocidos

| Área | Estado |
| --- | --- |
| Dataset, split agrupado, reproducibilidad | Hecho |
| Métricas de 3 clases, matrices de confusión, latencia | Hecho |
| Hate-vs-rest con PR-AUC e IC bootstrap | Hecho |
| Split de validación; umbrales elegidos ahí | Hecho |
| Inferencia batcheada con throughput separado | Hecho |
| Formulaciones binarias (choice, `noul`) | Hecho |
| Calibración medida (Brier, ECE, fiabilidad) | Hecho |
| Ajuste de temperaturas + calibración del score de odio (Platt/isotonic) | Hecho |
| Generalización a un segundo dataset | **Falta** |
| Codificación cualitativa de errores | Solo mecánica |
| Fine-tuning de Laya con el train | Hecho (fase 2A), opt-in con `--include-finetuned` |

### Bug de upstream encontrado por este benchmark

`laya.calibrate.records_from_labeled` llama a `agent._forward()` directamente, pero solo
`predict` / `predict_batch` llevan `@torch.no_grad()`. En un checkpoint cuyos parámetros
requieren gradiente, construir los records de calibración falla con
`RuntimeError: Can't call numpy() on Tensor that requires grad` (laya 0.3.22).
`src/laya_calibration.py` envuelve la llamada en `torch.no_grad()` como workaround que
no altera el comportamiento; el arreglo corresponde a upstream.

### Recalibración

Se calibran **dos cosas distintas**, y el benchmark mide ambas porque **no** son la misma
cantidad:

1. **El mapa de temperaturas de Laya** (`src/laya_calibration.py`), ajustado en validación
   y guardado como `results/calibration_<model>.json`. Calibra la confianza de la
   respuesta que Laya *eligió*. Medido: eso lo arregla (ECE held-out 0,089 → 0,026 en la
   variante binaria) y apenas mueve `P(hate speech)`.
2. **Un mapa monótono sobre el score de odio** (`src/score_calibration.py`): Platt scaling
   e isotonic regression, ajustados en validación y medidos en test. Esta es la cantidad
   que un detector umbraliza, y **es la que funciona**: reduce el ECE del score de odio de
   Laya en aproximadamente un orden de magnitud.

Ambos mapas son monótonos, así que ninguno puede cambiar una **predicción**. Lo que difiere es
lo que le cuestan al ranking:

- **Platt scaling** es estrictamente monótono, así que el PR-AUC y todas las métricas de
  clasificación son idénticas antes y después (medido: delta 0, hasta el último decimal).
- **Isotonic regression** es solo no-decreciente: fusiona miles de scores distintos en unas
  pocas docenas de bloques, aparecen empates, y `average_precision` cae unos puntos porque no
  puede ordenar dentro de un empate. El mejor ECE, a costa de un ranking más grueso.

La recalibración compra confianza en el número, no rendimiento de detección — y el reporte
**mide** la invariancia en vez de afirmarla.

Para reutilizar un mapa ajustado dentro de Laya:

```python
agent = laya.load("convaiinnovations/laya", calibration="results/calibration_laya.json")
```

Los mapas del score de odio son números planos (dos coeficientes para Platt, dos arrays
para isotonic), así que `src/score_calibration.py` los aplica sin dependencias extra.

---

## Contenido del repositorio y qué se deja fuera a propósito

Versionado: el código fuente, `config.yaml`, ambos README, `docs/PRD.md`, la licencia y
un conjunto pequeño de resultados **agregados** (`results/metrics.json`,
`model_comparison.csv`, `operating_points.csv`, `coverage.csv`,
`report/benchmark_report.md`).

No versionado, y por qué:

| Ruta | Motivo |
| --- | --- |
| `data/`, `results/predictions.csv`, `results/error_analysis.csv` | contienen el **texto crudo de los tweets**, que es discurso de odio. Se regeneran con `python run_benchmark.py` y se re-verifican por SHA-256. |
| `.venv/`, y los PNG más allá del reporte | regenerables |
| `results/summary.json` | regenerable; se usa solo para re-renderizar el reporte |

**Los errores representativos salen redactados por defecto.** El reporte lista las
categorías de error y los ids de ejemplo, pero omite el texto del tweet, porque el
reporte es el artefacto con más probabilidad de publicarse. Pon
`output.error_examples_in_report: full` para un reporte local, y re-renderiza sin
re-ejecutar ningún modelo:

```bash
python scripts/render_report.py
```

---

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).
