# PI de WebShop — componentes g_i, wrappers y formato canónico

> **Entregable de [#17](https://github.com/lucaspecina/piar-rl/issues/17)**
> (post-pivot: N.1 componentes + N.8 canonicalización). Define la
> descomposición exacta de la información privilegiada de WebShop en
> componentes fácticos verificables, el wrapper textual `w_i` de cada uno
> (para computar el belief `b_i`), y la serialización del residual `R(t)`
> al prompt del scorer.
>
> **Status:** diseño cerrado sobre la estructura del goal verificada en
> código ([`goal.py:48-58`](../../code/agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py));
> la validación empírica sobre el dataset real sigue bloqueada por la
> descarga (`setup.sh -d all`) — ver
> [`webshop-specs-analysis-plan.md`](../notes/webshop-specs-analysis-plan.md),
> cuyas preguntas §2 siguen vigentes.
>
> **Refs:** pivot §4.1–§4.3 · N.1, N.2, N.8 · C.5 (spec estructurada como PI) ·
> #23 (Figura 1 consume esta definición para computar b_i).

---

## 1. La fuente: el goal dict de WebShop

Cada episodio de WebShop instancia un goal con esta estructura
(`get_human_goals`, `goal.py:48-58`):

```python
{
    'asin': 'B078GWRC1J',            # product ID — EXCLUIDO de g (ver §3)
    'category': 'beauty',
    'query': '...',                   # query interna del engine — excluida
    'name': 'Bright Citrus Deodorant ...',   # título del producto target
    'product_category': 'Beauty › Deodorant', # jerarquía › separada
    'instruction_text': '...',        # lo que el STUDENT ya ve — no es PI
    'attributes': ['3 ounce', 'bright citrus', 'sensitive skin'],
    'price_upper': 50.0,              # randomizado por el env al instanciar
    'goal_options': {'size': '3 ounce', 'scent': 'bright citrus'},
}
```

El reward del env (`get_reward`, `goal.py:228-269`) puntúa exactamente sobre
estos campos: tipo (name/category), attributes (fuzzy match), options, precio.
**Eso garantiza que cada g_i es causalmente relevante para el outcome** — no
estamos eligiendo hechos decorativos.

## 2. Los componentes g_i

`K = 2 + |attributes| + |goal_options|` componentes por episodio (mediana
esperada K ≈ 4–6; verificar con el extractor):

| i | Componente | Valor canónico (verbatim del dataset) | Por qué es un componente separado |
|---|---|---|---|
| `g_prod` | Producto target | `goal['name']` | Es la incógnita principal de la fase de búsqueda. |
| `g_attr_k` (uno por atributo) | Atributo requerido k | `goal['attributes'][k]`, orden del dataset | El env los puntúa individualmente; el student los descubre de a uno (curriculum natural del gating). |
| `g_opt_k` (uno por opción) | Opción requerida k | `goal['goal_options'][key]`, keys en orden alfabético | Ídem; además el key (`size`, `color`) va al wrapper, no al target — sin ambigüedad de a qué opción se refiere el belief. |
| `g_price` | Precio máximo | `f"{goal['price_upper']:.2f} dollars"` | Hecho puntual; se omite el componente si `price_upper ≥ 1000000` (sentinel de "sin restricción", `goal.py:43`). |

Restricción de invariante 10 propuesto (N.2): todos los g_i son **hechos del
dataset**, ninguno opina sobre acciones. Nada de "search for X first".

## 3. Exclusiones explícitas

- **`asin`**: es el leak trivial (identifica el producto sin entender nada).
  Queda fuera de g. Solo aparece en la **ablation B** de C.5 (upper bound de
  leakage) y como key de logging.
- **`query`**: artefacto interno del engine de búsqueda, no un hecho de la
  tarea que el usuario "sabría".
- **`instruction_text`**: lo ve el student en su prompt normal — no es
  privilegiado. (Es el lower bound de info de C.5.)
- **`category` / `product_category`**: redundantes con `g_prod` para el
  scoring y con riesgo de sesgar la búsqueda por categoría. Fuera del default;
  candidatos a ablation de "g extendida" si la Figura 1 muestra que `g_prod`
  solo no mueve el belief.

## 4. Wrappers w_i y cómputo del belief

Wrappers **en inglés** (el env y el modelo operan en inglés), fijos por tipo
de componente, con el valor canónico como continuación a puntuar:

```text
g_prod:   "Based on the interaction so far, the exact product the user is looking for is:"
          → " {name}"
g_attr_k: "Based on the interaction so far, required attribute #{k} of the target product is:"
          → " {attribute_k}"
g_opt_k:  "Based on the interaction so far, the required {option_key} option of the target product is:"
          → " {option_value}"
g_price:  "Based on the interaction so far, the maximum acceptable price for the target product is:"
          → " {price_upper:.2f} dollars"
```

Belief del componente i en el turno t (pivot §4.2):

    b_i(t) = (1/|g_i|) · Σ_{tok ∈ g_i} log π_old(tok | h_t, w_i)

con h_t = historial hasta la observación o_t inclusive, |g_i| = longitud en
tokens del valor canónico (normalización 1/L estilo IGPO Eq. 3 — ver
[`paper-igpo.md`](../notes/paper-igpo.md)). Los tokens del wrapper NO entran
a la suma; solo el valor.

**Detalle de implementación**: los K beliefs del turno t comparten el prefijo
h_t — con prefix caching son K continuaciones cortas sobre un solo prefill.
Costo marginal ≈ cero frente al rollout.

### 4.1 Caveat conocido: mass-splitting entre atributos intercambiables

Para `g_attr_k` el wrapper indexado (`#1`, `#2`) es determinístico pero el
modelo no tiene forma de saber qué atributo es "el #1" — la masa de
probabilidad se reparte entre los atributos válidos y los b_attr quedan
deprimidos de forma pareja. Esto **no rompe el método** (el gating compara
b_i contra τ y el backward usa Δb_i, ambos relativos), pero sí puede requerir
**τ por tipo de componente** en vez de un τ global. Decisión data-driven en
la Figura 1: loggear distribuciones de b_i por tipo y elegir.

Variante ablation (si el indexado resulta ruidoso): wrapper conjunto
"the required attributes of the target product are:" con los atributos en
orden canónico como un solo componente `g_attrs` (K más chico, curriculum más
grueso).

## 5. Serialización del residual R(t) al prompt del scorer

El residual (pivot §4.3) entra como bloque fijo al inicio del prompt del
scorer del forward, **solo con los componentes no sabidos**
(`b_i(t−1) < τ_tipo`), cada uno en una línea con label fijo:

```text
Privileged information (for evaluation only — the shopper cannot see this):
- Target product: {name}
- Required attribute: {attribute_k}        # una línea por g_attr_k ∈ R(t)
- Required {option_key}: {option_value}    # una línea por g_opt_k ∈ R(t)
- Maximum price: {price_upper:.2f} dollars
```

Reglas:
- Orden fijo (producto → atributos → opciones → precio) independiente de qué
  componentes estén presentes — el formato no debe filtrar información sobre
  *qué sabe* el student (el scorer no debe poder inferir el turno por el
  layout).
- Si `R(t) = ∅`, el bloque se omite entero y los dos prompts del forward son
  idénticos → `r_fwd(t) = 0` exacto (auto-annealing, pivot §4.3).
- El bloque reemplaza al template estilo OPSD de C.3 ("Here is a reference
  solution...") — aquel asumía golden monolítica; el formato por líneas es la
  versión por componentes. C.3 queda para la ablation full-golden (brazo A1).

## 6. Logging (invariante 5 + N.9)

Por episodio, en `experiments/ENNN/`:
- El goal dict completo (incluido `price_upper` instanciado — es aleatorio
  por episodio, `goal.py:39`).
- La serialización exacta de cada g_i y su |g_i| en tokens.
- Por turno: los K beliefs b_i(t), el R(t) resultante, y los dos scores.
- Por acción: el tipo (`search` / `click-nav` / `click-opción-buy`) — lo
  exige el riesgo de loitering (pivot §10.2) y la Figura 1.

## 7. Qué queda pendiente para cerrar #17

1. **Descarga del dataset + correr `tools/extract_webshop_specs.py`**
   (bloqueado: requiere Python local o la VM — no disponible en esta máquina
   al 2026-06-12). Responde las preguntas de cobertura/riqueza/token-budget
   del [plan de análisis](../notes/webshop-specs-analysis-plan.md) §2.
2. **Adaptar `serialize_spec_for_teacher_prompt`** del extractor al formato
   por componentes de §5 (hoy serializa la spec monolítica de C.5 pre-pivot).
3. **Verificación de cuasi-ortogonalidad de los g_i** (caveat KnowRL "pruning
   interaction paradox", [`paper-knowrl.md`](../notes/paper-knowrl.md)) — se
   hace con datos de la Figura 1, no antes.
4. **τ global vs τ por tipo** (§4.1) — Figura 1.
