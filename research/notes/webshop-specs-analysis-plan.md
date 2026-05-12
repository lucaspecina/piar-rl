# Plan de análisis — WebShop privileged context specs

> **Qué es esto:** el plan para qué responder cuando `tools/extract_webshop_specs.py`
> corra contra el dataset real. Define **qué buscar, qué decidir, qué bloquea fase 4**.
>
> **Status:** plan a ejecutar. Bloqueado por descarga del dataset
> (`code/agent_system/environments/env_package/webshop/webshop/setup.sh -d all`).
>
> **Refs:** #17 (issue), C.3 + C.5 (decisiones), piar-implementation-points.md §3.

---

## 1. Por qué este análisis bloquea fase 4

Decisión C.5 (2026-05-11): la spec estructurada del producto target es el
**primary privileged context** del teacher de PIAR (en lugar de trayectorias
humanas, descripción NL, o ASIN literal). Razón documentada:
- Cubre el 100% de las tareas (12K vs 1.6K para human demos).
- Es rica sin ser trivialmente copiable.
- Viene gratis en el dataset, no requiere generación.

**Pero la decisión se tomó sin ver las specs reales.** Antes de fase 4 hay que
confirmar empíricamente que la spec es lo suficientemente rica como para que el
log-ratio teacher-vs-student tenga señal medible. Si las specs son pobres
(promedio 1 atributo, opciones vacías), PIAR puede no tener margen.

Este análisis cierra esa duda con datos, antes de gastar GPU-días en fase 3+4.

## 2. Preguntas concretas a responder

### 2.1 Cobertura

- **¿Cuántos goals totales se extraen?** Esperado: ~12K para `items_human_ins.json` full, ~1K para small.
- **¿Cuántos goals se skipean por falta de attributes?** El código original
  (`goal.py:31-33`) skipea estos. Si el % es alto, hay menos data efectiva.

### 2.2 Riqueza de las specs

- **Distribución de # de attributes por goal**. Pregunta clave: ¿el goal mediano
  tiene 1, 2, 3 o más atributos?
  - **Hipótesis si mediana ≤ 1**: specs pobres. PIAR puede no tener margen.
    Considerar fallback a trayectorias humanas sobre el subset de 1.6K.
  - **Hipótesis si mediana ≥ 2**: specs ricas suficientes. C.5 se sostiene.
- **Distribución de # de options por goal**. `goal_options` puede estar vacío.
  - **Si > 50% tiene options vacías**: bajar peso de options en el template, no
    incluirlas si no hay.

### 2.3 Token budget

- **Distribución de longitud (en tokens estimados) del bloque GOLDEN SPEC**
  serializado.
  - **Si p99 > 500 tokens**: revisar si `max_prompt_length=4096` aguanta el
    prompt original (~3000 tokens de history multi-turn) + golden + response.
    Si no: subir `max_prompt_length` a 5120 o filtrar goals con spec demasiado larga.
  - **Si p99 < 200 tokens**: tranquilos, no es un problema.
- **¿La longitud crece con # de attributes?** Lineal o irregular?

### 2.4 Diversidad

- **Top categorías**: ¿está balanceado o concentrado? Si una categoría tiene
  > 30% de los goals, hay riesgo de overfitting per-categoría.
- **ASINs únicos vs goals totales**: cada ASIN puede tener múltiples
  instructions humanas. ¿Cuántas instrucciones promedio por producto?

### 2.5 Setup para ablations C.5 (D.9)

- **Subset de 1.6K con trayectorias humanas** (decisión C.5): identificar qué
  goals tienen `instructions` populated y guardar la lista para correr la
  ablation "spec sola" vs "spec + traj humana" sobre el subset.
- **Goals con `asin` único** (para D.9 shuffled-golden control): hace falta
  saber que goals son distinguibles entre sí. ASINs únicos por goal facilita
  la rotación.

---

## 3. Criterios de éxito del análisis

Después de correr el extractor, **el plan se considera resuelto si**:

1. ✅ El JSON output existe y tiene el campo `analysis` populado.
2. ✅ El campo `n_goals` está en el rango esperado (12K full / 1K small).
3. ✅ Hay al menos **1 ejemplo de spec rica** entre los 10 random sample
   (manualmente revisado: la spec se "lee" como algo accionable, no como
   campos sueltos).
4. ✅ Las decisiones de §4 abajo están tomadas (con datos, no con hipótesis).

---

## 4. Decisiones que el análisis destraba

### 4.1 ¿Spec estructurada como primary se sostiene? (Confirmar C.5)

- **Sí, si**: mediana ≥ 2 attrs Y ≥ 30% tiene options Y p99 token length < 500.
- **Pivot a "spec + traj humana" sobre subset 1.6K, si**: mediana < 2 attrs Y
  trayectorias humanas razonablemente disponibles.
- **Pivot a "instruction_text rica" (NL), si**: spec pobre Y trayectorias
  humanas también pobres. Implica reescribir C.5.

### 4.2 ¿`max_prompt_length` actual aguanta?

- **Sí, si**: p99 del bloque GOLDEN SPEC < 400 tokens (margen 100 tokens sobre
  prompt original ~3500).
- **No, si**: p99 > 500 tokens. Subir `max_prompt_length` a 5120 en `run_webshop.sh`.

### 4.3 ¿Hace falta filtering por categoría?

- **No, si**: categoría top < 30%.
- **Sí, si**: categoría top > 50%. Considerar stratified sampling.

### 4.4 ¿La función `serialize_spec_for_teacher_prompt` está bien diseñada?

Revisar 5-10 ejemplos serializados manualmente:
- ¿Se lee como prosa útil o como un dump de keys=values?
- ¿El delimitador `[GOLDEN SPEC]` se confunde con tokens reales del chat
  template de Qwen2.5? (Sospechado: improbable, pero verificar visualmente.)
- ¿Falta algún campo crítico (e.g., `price_upper` cuando no es None)?

Si hay problemas, ajustar `serialize_spec_for_teacher_prompt` y re-correr.

---

## 5. Cómo correr

### 5.1 Pre-requisito: bajar el dataset

```bash
# En la VM (idealmente, una vez) o local con conda env compatible:
cd code/agent_system/environments/env_package/webshop/webshop
./setup.sh -d all  # baja items_shuffle.json + items_ins_v2.json + items_human_ins.json
# Carpeta resultante: code/agent_system/environments/env_package/webshop/webshop/data/
```

Tamaño aproximado del download: ~150-200 MB para `-d all`.

### 5.2 Correr el extractor

```bash
python tools/extract_webshop_specs.py \
    --items-human-ins code/agent_system/environments/env_package/webshop/webshop/data/items_human_ins.json \
    --out experiments/E000/webshop-specs.json \
    --n-examples 15
```

Tiempo esperado: <1 minuto. No requiere GPU. No requiere conda env de webshop
(spaCy, faiss, etc.) — el script es puro Python stdlib.

### 5.3 Revisar el output

1. Leer el `metadata` y `analysis` del JSON. Confronto contra las hipótesis
   de §2 + criterios de §3.
2. Leer los 15 ejemplos manualmente. Vibe check: ¿se lee razonable?
3. Tomar las decisiones de §4.
4. Updatear `design-decisions.md` C.5 si se sostiene; reescribir si no.
5. Documentar resultados al cerrar #17 (comment en el issue + link al JSON).

---

## 6. Salida esperada del análisis (template de reporte)

Cuando se cierre #17, postear comment con esta estructura:

```markdown
## Resultados extracción WebShop specs (`experiments/E000/webshop-specs.json`)

### Cobertura
- Total goals extraídos: NNNN (de M productos cargados)
- Goals skipeados: NNN (X.X%, sin attributes)

### Riqueza
- Attributes/goal — mediana: N, mean: N.N, p99: N
- Options/goal — mediana: N, mean: N.N
- Specs ricas (≥2 attrs + ≥1 opt): X.X%
- Specs pobres (<2 attrs): X.X%

### Token budget
- Longitud serializada del bloque GOLDEN SPEC — mediana: N tokens, p99: N tokens
- ¿Excede max_prompt_length=4096 combinado con prompt original? SÍ/NO

### Decisiones tomadas
- C.5 sostenida / pivot a alternativa X
- max_prompt_length: mantener 4096 / subir a 5120
- Filtering categórico: SÍ/NO

### Próximo paso bloqueado
- Si C.5 sostenida: avanzar a fase 4 implementación PIAR (#16).
- Si pivot: reescribir C.5 antes de fase 4.
```

---

## 7. Riesgos identificados al planificar

- **El dataset puede tener corruption o estructura inesperada** vs lo que
  asumimos en el script. Mitigación: el script defensive-codes con `.get(key, default)`
  y reporta `skipped`. Si > 5% se skippean: investigar.
- **`get_human_goals` original aleatoriza `price_upper`** desde un `PRICE_RANGE`
  filtrado por precio real del producto. Nuestro extractor pone `None` porque
  no carga precios. **Esto es OK**: el price_upper "real" es el que el env
  computará al instanciar, no algo que dependa de nuestra extracción. El template
  del teacher omite price_upper si es None.
- **Goals duplicados**: posible que el mismo ASIN aparezca con múltiples
  instrucciones humanas idénticas. Si > 10% son duplicados, deduplicar para
  D.9 shuffled (necesitamos diversidad en el shuffle).
