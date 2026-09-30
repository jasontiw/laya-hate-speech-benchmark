# Benchmark Local de Detección de Discurso de Odio

[English](README.md) | **Español**

Un experimento pequeño, local y reproducible que mide cómo rinde **Laya** en la
detección de discurso de odio y lo compara contra clasificadores clásicos y
especializados sobre el **mismo dataset público** y las **mismas filas de test**.

Pregunta de investigación:

> ¿Cómo rinde Laya en un dataset conocido de discurso de odio frente a otros
> enfoques locales de clasificación, cuando todos los modelos se evalúan sobre los
> mismos ejemplos y con las mismas métricas?

Es un benchmark de investigación, **no** un sistema de moderación en producción.
Se reduce a un comando y a un conjunto de artefactos generados.

---

## Modelos comparados

| Clave | Modelo | ¿Entrenado aquí? | Probabilidades |
| --- | --- | --- | --- |
| `majority` | Baseline de clase mayoritaria | se ajusta en el train | no |
| `tfidf` | TF-IDF + Regresión Logística | se ajusta en el train | sí |
| `laya` | Laya (pregunta `choice` zero-shot) | no | sí |
| `hatexplain` | `Hate-speech-CNERG/bert-base-uncased-hatexplain` | no | sí |

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

En macOS/Linux usa `.venv/bin/python run_benchmark.py`, o simplemente `python` dentro
de un entorno virtual activado.

La primera ejecución descarga el dataset (~2,5 MB) y los checkpoints (Laya English,
HateXplain BERT); después quedan en la caché de Hugging Face.

### Flags útiles

```bash
python run_benchmark.py --limit 200                  # smoke test rápido (marcado como "no es un resultado del benchmark")
python run_benchmark.py --models tfidf,laya          # solo algunos modelos
python run_benchmark.py --config my_experiment.yaml  # configuración alternativa
python run_benchmark.py --list-models                # imprime las claves de modelo y sale
```

Cada ejecución se gobierna desde [`config.yaml`](config.yaml); ese archivo se copia
tal cual en `results/experiment_config.json`.

---

## Artefactos de salida

```
results/
├── dataset_metadata.json     # fuente, SHA-256, filas, distribución de clases
├── experiment_config.json    # config + hardware/runtime + revisiones resueltas
├── split.json                # semilla, conteos y la lista exacta de ids de test
├── predictions.csv           # una fila por tweet de test, predicciones + probabilidades
├── metrics.json              # métricas completas por modelo
├── model_comparison.csv      # la tabla comparativa del PRD
├── error_analysis.csv        # errores agrupados por categoría (gold -> predicho)
├── confusion_matrix_<model>.png
└── report.md

report/
└── benchmark_report.md       # el entregable final
```

`benchmark_report.md` tiene las 16 secciones que exige el PRD (resumen ejecutivo,
dataset, montaje, modelos, metodología, resultados generales y de odio, matrices de
confusión, latencia, análisis de errores, análisis de Laya, limitaciones, conclusiones,
instrucciones de reproducción y referencias).

---

## Metodología

- **Manejo del dataset.** Descarga programática, verificación por SHA-256, nunca se
  edita, se conservan las etiquetas originales.
- **Split.** División estratificada fija 80/20 con semilla determinista. Los tweets que
  normalizan a la misma cadena quedan en el mismo lado, así un tweet duplicado no puede
  filtrarse de train a test. La normalización se usa *solo* para detectar duplicados —
  los modelos siempre ven el texto crudo.
- **Pregunta a Laya.** La detección se plantea como una pregunta `choice` con las tres
  etiquetas y sus descripciones. Sin fine-tuning.
- **Mapeo de HateXplain.** Su `id2label` se lee de la config del modelo y se mapea a las
  etiquetas canónicas (`hate speech -> hate speech`, `offensive -> offensive language`,
  `normal -> neither`). El mapeo crudo queda registrado en el reporte.
- **Métricas.** Accuracy, macro F1, macro precision/recall, precision/recall/F1 por clase,
  matrices de confusión, y precision/recall/F1 específicos de odio con conteos de
  falsos positivos y negativos.
- **Latencia.** Tiempo de inferencia por ítem con warm-up, reportado como total, media,
  p50, p95 y throughput. El tiempo de carga/`fit` del modelo se reporta aparte.
- **Reproducibilidad.** Misma config → mismo split (se registra el SHA-256 de la lista
  de ids de test). Las revisiones de modelo se pueden fijar en `config.yaml`.

---

## Estructura del proyecto

```
laya-hate-speech-benchmark/
├── config.yaml               # ajustes del experimento (única fuente de verdad)
├── run_benchmark.py          # punto de entrada: python run_benchmark.py
├── requirements.txt
├── pyproject.toml
├── data/{raw,processed}/     # ignorado por git
├── scripts/run_benchmark.py  # envoltorio fino (mismo punto de entrada)
├── src/
│   ├── config.py             # config tipada + etiquetas canónicas
│   ├── dataset.py            # descarga, verificación, normalización, split agrupado
│   ├── environment.py        # introspección de hardware/runtime
│   ├── metrics.py            # métricas de clasificación y latencia
│   ├── evaluation.py         # orquestación
│   ├── reporting.py          # gráficos de confusión + reporte Markdown
│   └── models/
│       ├── base.py           # interfaz común
│       ├── majority.py
│       ├── tfidf.py
│       ├── laya_model.py     # Laya zero-shot
│       └── hatexplain.py
└── results/, report/         # generados, ignorados por git
```

### Desviaciones respecto a la estructura propuesta en el PRD

- `src/models/laya.py` se llama `laya_model.py`. Un módulo llamado literalmente `laya.py`
  dentro de este paquete puede ensombrecer al paquete `laya` instalado al importar; el
  renombre elimina esa ambigüedad. El resto sigue la estructura propuesta.
- El dataset se descarga como el CSV original de los autores en lugar de vía
  `huggingface/datasets`, para poder hashear directamente el archivo descargado.
- El fine-tuning, Detoxify y las extensiones de fase 2 **no** están implementados: esto
  es solo el MVP.

---

## Criterios de aceptación

| AC | Dónde se cumple |
| --- | --- |
| AC1 dataset descargado + verificado | `src/dataset.py`, `results/dataset_metadata.json` |
| AC2 split determinista persistido | `src/dataset.py`, `results/split.json` |
| AC3 Laya procesa todo el test | `src/models/laya_model.py` |
| AC4 predicciones TF-IDF + LR | `src/models/tfidf.py` |
| AC5 predicciones HateXplain | `src/models/hatexplain.py` |
| AC6 los cuatro modelos, todas las métricas | `src/metrics.py`, `results/metrics.json` |
| AC7 precision/recall/F1 de odio | sección 8 del reporte |
| AC8 matriz de confusión por modelo | `results/confusion_matrix_*.png` |
| AC9 latencia registrada | `results/metrics.json` (`latency`), sección 10 del reporte |
| AC10 todas las predicciones en un archivo | `results/predictions.csv` |
| AC11 reporte Markdown generado | `report/benchmark_report.md` |
| AC12 ejecución reproducible | semilla fija + SHA-256 de ids de test registrado |

---

## Publicar en GitHub

El proyecto es autocontenido. El entorno virtual, el dataset descargado y todos los
artefactos generados están ignorados por git, así que un clon limpio queda pequeño y
los reconstruye con un solo comando.

```bash
git init
git add .
git commit -m "feat: benchmark local de discurso de odio comparando Laya con baselines"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/<tu-repo>.git
git push -u origin main
```

Para publicar igualmente un resultado generado concreto (por ejemplo el reporte de una
corrida real):

```bash
git add -f report/benchmark_report.md results/model_comparison.csv
git commit -m "docs: añade el reporte del benchmark de <fecha>"
```

---

## Licencia

Apache-2.0. Ver [LICENSE](LICENSE).
