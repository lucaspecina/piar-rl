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

### `figura1_scoring.py`

Harness de scoring de la **Figura 1** (pre-registro congelado
[`research/synthesis/figura1-prereg.md`](../research/synthesis/figura1-prereg.md),
#23). Sobre un JSONL de rollouts congelados computa las tres lecturas
(r_fwd full-golden, r_bwd, r_fwd residual con τ = percentil por tipo),
clasifica acciones con las reglas determinísticas del prereg §3, y emite
las confirmatorias P1'/P2'/P3/P5, el control P4 y las descriptivas §4.3
(incl. N.14 ventana-2 vs full y las dos variantes de wrapper para P5).
Reusa `decompose_goal_components` / `serialize_residual_block` del extractor.

**Input** (`--rollouts`): JSONL, una línea por episodio:
`{"goal": {<goal dict de runtime, CON asin y name>}, "turns": [{"action", "observation"}, ...], "score": float}`.
La observación del turno t es la respuesta del env a la acción t (incluir la
observación final). `clicked_asin` opcional por turno (solo si el texto del
click no es el ASIN). Contrato completo en el docstring del script.

**Día-1 en la VM** (run real, prereg §2 — Qwen2.5-7B, ~200 episodios):

```bash
# 1) run real (gatea P1'/P2'/P3/P5; --history both agrega el reporte N.14)
PYTHONUTF8=1 uv run --with torch --with transformers --no-project \
    python tools/figura1_scoring.py \
    --rollouts experiments/E001/rollouts.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --device auto --history both \
    --out experiments/E001/figura1_scores.json

# 2) control P4 (shuffled-golden) + verdict contra el run real
PYTHONUTF8=1 uv run --with torch --with transformers --no-project \
    python tools/figura1_scoring.py \
    --rollouts experiments/E001/rollouts.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct --device auto --history both \
    --shuffled --baseline-json experiments/E001/figura1_scores.json \
    --out experiments/E001/figura1_scores_shuffled.json
```

(Shakedown previo con `--model Qwen/Qwen2.5-1.5B-Instruct`; los números que
cuentan son los del 7B. Ablation de length-norm: `--length-norm`.)

**Smoke test en CPU** (corrido 2026-06-12 en Windows, sin GPU — fixture
sintético de 4 episodios en `tools/testdata/figura1_smoke.jsonl`):

```bash
# asserts de mecánica (spans frontera wrapper→valor, clasificación, gating,
# derangement, stats) — solo tokenizer, sin pesos:
PYTHONUTF8=1 uv run --with torch --with transformers --no-project \
    python tools/figura1_scoring.py --self-test --model Qwen/Qwen2.5-0.5B-Instruct

# end-to-end con modelo chico:
PYTHONUTF8=1 uv run --with torch --with transformers --no-project \
    python tools/figura1_scoring.py \
    --rollouts tools/testdata/figura1_smoke.jsonl \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --history both \
    --batch-size 4 --out tools/testdata/figura1_smoke_scores.json
```

**Decisiones de implementación donde el prereg era ambiguo** (lista corta —
detalle en el docstring; revisar antes del run real):

1. **P4**: clasificación contra el goal VERDADERO, scores con el goal rotado
   (derangement cíclico seeded). P4 cubre la familia {P1', P2', P3}; el AUC
   de P5 bajo shuffle se reporta descriptivo (que el belief siga al contexto
   bajo shuffle ES el mecanismo, no leakage).
2. **P1' gatea sobre r_fwd full-golden** (el residual está condicionado a
   P5/τ; se reporta como secundario). PASS = contraste de producto Y de
   opción con dirección correcta y p < α/3; clase vacía → N/A.
3. Tests **one-sided** (las predicciones son direccionales); Bonferroni
   α = 0.05/3 sobre {P1', P2', P3}.
4. **τ_tipo** = percentil `--tau-percentile` (default 50) sobre TODOS los
   b_i(t) pooled (t=0..T, todos los episodios) por tipo, del propio run.
5. **Full-golden usa el mismo formato de bloque por líneas** que el residual
   (todos los K componentes) — aisla el gating como única variable; el
   template OPSD `[GOLDEN SPEC]` queda para el brazo A1 de training.
6. Los prompts son una **reconstrucción reducida** del template real (el
   contrato no trae obs inicial ni admissible actions): header + transcript
   de pares (acción, obs) + tail. Ventana-2 = últimos 2 pares (el pipeline
   real muestra además la obs anterior a la acción más vieja — media
   observación de desvío). b_i(0) = solo la instrucción. r_fwd puntúa la
   acción pelada como mensaje del assistant (sin scaffold `<think>/<action>`;
   el contrato no trae thinks).
7. **P5** incluye filas t=0 (label "no aparecido"). Matching = forma canónica
   verbatim case-insensitive (regla congelada): para `price` ("X.XX dollars")
   casi nunca matchea ("$X.XX" en las páginas) → tipo sin positivos, se
   excluye del AUC y se loggea un conteo descriptivo con "$X.XX" (no gatea).
8. "Observaciones previas" excluye la instrucción (no es una observación).
9. Variante primaria de wrapper = `--wrapper-variant` (default `prefill`,
   alimenta r_bwd/gating); la secundaria se computa siempre para P5 salvo
   `--skip-secondary-wrapper`.
10. Clicks de producto: ASIN del texto del click (regex `^[A-Za-z0-9]{10}$`)
    vs `goal['asin']`; fallback `clicked_asin` loggeado; fallback match por
    `goal['name']`. Opciones: lowercase + `normalize_color` (importada del
    engine vendoreado en `code/`, sin duplicar COLOR_SET).

**Pendiente**: checkpointing por lotes del scoring (prereg §7 lo pide para
la VM spot; el run de ~200 episodios es corto y re-corrible — agregar si el
wall-clock real lo justifica).

**Refs**: #23 (prereg), #17 (g_i), `pivot-2026-06.md` §4/§8,
`pi-webshop.md` §2–§5/§8, `piar-implementation-points.md` §7.

### `vm_setup.sh`

Setup día-1 de la VM (`lp-gpu-h100-x2-spot`) en UN comando, idempotente
(re-correrlo tras un fallo parcial es seguro — cada paso skipea si ya está).
Instala Miniforge + envs `piar` (3.12, training) y `webshop` (3.10, env +
training encima, según `code/README.md`), baja los datasets por el **mirror
HF** (el Drive oficial está muerto) + el **índice Lucene pre-construido**
(ahorra horas de indexing), baja Qwen2.5-7B/1.5B con symlinks compatibles
con los scripts del fork, y corre verificaciones finales (GPU count, import
del env, self-test del harness). ~60-80 GB, 1-2 h.

```bash
# desde la máquina local: prender + auto-shutdown (una vez) + ssh
az vm start -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot
az vm auto-shutdown -g RG-IAF-YTEC-poc-int -n lp-gpu-h100-x2-spot --time 0600
# en la VM:
git clone https://github.com/lucaspecina/piar-rl.git && cd piar-rl
git checkout pivot-2026-06
tmux new -s setup
bash tools/vm_setup.sh 2>&1 | tee ~/vm_setup.log
```

Supuestos no verificables desde Windows marcados en el script:
`grep VERIFICAR-EN-VM tools/vm_setup.sh` (usuario admin, CUDA del host,
ABI de flash-attn, subdir del índice). **Escrito en seco (2026-06-12), sin
probar en Ubuntu real** — el primer run se supervisa por SSH.

**Refs**: #16 (checklist día-1 + configs B1/B2 pre-registradas en comments).
