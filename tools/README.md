# `tools/` — Scripts utilitarios standalone

Scripts que NO son parte del codebase de training (eso vive en `code/`) pero
ayudan a preparar data, analizar artefactos o automatizar setup.

Característica común: **portables**, sin dependencias raras, ejecutables fuera
de la conda env de training. Se corren tanto en local (Windows / macOS) como
en la VM Azure.

## Scripts disponibles

### `extract_webshop_specs.py`

Extrae las specs estructuradas del dataset WebShop (productos + atributos +
opciones + price_upper + categoría) en un JSON limpio para análisis y para
serialización en el prompt del teacher de PIAR.

**Input**: `items_human_ins.json` (bajado por `code/agent_system/.../setup.sh -d all`).

**Output**: JSON con metadata + análisis estadístico + N ejemplos serializados.

**Uso**:

```bash
python tools/extract_webshop_specs.py \
    --items-human-ins /path/to/webshop/data/items_human_ins.json \
    --out experiments/E000/webshop-specs.json \
    --n-examples 10
```

Para extracción completa (todos los ~12K goals):

```bash
python tools/extract_webshop_specs.py \
    --items-human-ins /path/to/items_human_ins.json \
    --out experiments/E000/webshop-specs-full.json \
    --include-all-goals
```

**Post-pivot 2026-06**: además de la spec monolítica (brazo A1), el script
descompone cada goal en **componentes g_i** con wrapper w_i (N.1/N.8, formato
de [`research/synthesis/pi-webshop.md`](../research/synthesis/pi-webshop.md)
§2–§5) y serializa el bloque del residual R(t). El output incluye análisis de
K por goal y longitudes de value por tipo. Smoke-tested con fixture sintético
(2026-06-12; `uv run --no-project python tools/extract_webshop_specs.py ...`
funciona sin deps).

**Refs**: #17, decisiones C.3 (template A1), C.5 (spec como primary PI),
N.1/N.8 (componentes + canonicalización). Plan de análisis en
[`research/notes/webshop-specs-analysis-plan.md`](../research/notes/webshop-specs-analysis-plan.md).

**Estado del dataset (2026-06-12)**: el Google Drive oficial está con "Quota
exceeded" — descarga bloqueada también desde esta máquina (además del proxy).
Mirror en búsqueda; ver comentario en #17.

Para usar las funciones directamente en el código de PIAR (cuando se
implemente `compute_piar_step_reward` / el belief tracker), importar desde acá:

```python
from tools.extract_webshop_specs import (
    decompose_goal_components,
    serialize_residual_block,
    serialize_spec_for_teacher_prompt,
)
```

(O copiar las funciones a `code/istar/piar_step_reward.py` si se quiere
autonomía de `code/` respecto a `tools/`.)
