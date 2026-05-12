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

**Refs**: #17, decisión C.3 (template), C.5 (spec como primary PI). Plan de
análisis en [`research/notes/webshop-specs-analysis-plan.md`](../research/notes/webshop-specs-analysis-plan.md).

Para usar `serialize_spec_for_teacher_prompt()` directamente en el código de
PIAR (cuando se implemente `compute_piar_step_reward`), importar desde acá:

```python
from tools.extract_webshop_specs import serialize_spec_for_teacher_prompt
```

(O copiar la función a `code/istar/piar_step_reward.py` si se quiere autonomía
de `code/` respecto a `tools/`.)
