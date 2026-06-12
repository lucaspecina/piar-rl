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

**Costo de implementación (corregido 2026-06-12)**: la versión original de
esta sección decía "con prefix caching son K continuaciones cortas sobre un
solo prefill; costo marginal ≈ cero". **Eso es falso con el código tal cual**:
el path de scoring (`compute_log_prob`) va por el actor FSDP, sin prefix
caching — cada query repaga el prefijo completo, y el belief pass cuesta
≈ K× el pass de old_log_probs (~+30-80% del wall-clock del step; con el K
efectivo ≈ 4 medido, ~7-11M token-forwards por iteración). Viable pero no
gratis; la palanca si duele es scoring vía vLLM con APC. Detalle completo y
verificado: [`piar-implementation-points.md`](piar-implementation-points.md)
§7.4.

**Inyección del wrapper — dos variantes a arbitrar en la Figura 1** (riesgo
§7.5.4 del doc de implementación): el prompt de cada fila termina en
`<|im_start|>assistant\n` tras un template que instruye a producir
`<think>...<action>...`. (a) **Assistant-prefill** (barato): concatenar el
wrapper ahí — pero puntúa g_i en un contexto que pide otra cosa y puede
deprimir el nivel absoluto de b_i (otro golpe al gate; el backward, que usa
deltas, es más robusto). (b) **User-message** (más LOC): reemplazar el bloque
de instrucciones ReAct por la pregunta del wrapper como mensaje de usuario.
P5 se corre para ambas; la que separe mejor queda fijada para training
(regla de selección pre-registrada en `figura1-prereg.md`).

### 4.1 Caveat conocido: mass-splitting entre formas intercambiables

Dos fuentes de mass-splitting, con efectos distintos sobre los dos usos del
belief (distinción del review externo 2026-06-12):

- **El backward es robusto**: usa deltas — si la masa se reparte de forma
  estable entre paráfrasis/órdenes válidos durante el episodio, el offset se
  cancela en b_i(t) − b_i(t−1).
- **El gate es el que sufre**: compara el *nivel absoluto* contra τ. Un
  componente que el student ya sabe pero puede expresar de cinco formas queda
  ~log(5) por debajo de donde debería; el umbral no se cruza nunca y el
  scorer repite lo que el student ya sabe — muere el auto-annealing.

Mitigaciones, en orden de preferencia:

1. **Canonicalización agresiva vía wrapper**: restringir el formato de la
   respuesta en el propio wrapper para colapsar las paráfrasis — ej.
   `"Answer with the exact attribute phrase, lowercase: ..."` /
   `"Answer with the exact product title: ..."`. Los valores de WebShop ya
   son frases cortas del dataset; el wrapper restrictivo reduce el espacio de
   formas válidas hacia la canónica.
2. **τ por percentil empírico por tipo de componente** (no umbral universal):
   calibrado con las distribuciones de b_i de la Figura 1.
3. **Check pre-registrado P5** (ver [`figura1-prereg.md`](figura1-prereg.md)
   §4.1): AUC > 0.8 separando b_i de componentes ya aparecidos verbatim en
   observaciones vs no aparecidos. Si P5 falla, el gating por belief se
   rediseña antes de gastar GPU → §4.2.

Para `g_attr_k` en particular, el wrapper indexado (`#1`, `#2`) tiene
mass-splitting adicional entre atributos (el modelo no sabe cuál es "el #1").
Variante ablation: wrapper conjunto "the required attributes of the target
product are:" con orden canónico como un solo componente `g_attrs` (K más
chico, curriculum más grueso).

### 4.2 Fallback determinístico: gating observacional

Si P5 falla, el residual se gatea **sin belief**: el componente g_i sale de
R(t) cuando su **forma canónica apareció en alguna observación del episodio**
(matching textual determinístico, mismo criterio que la sub-clasificación
"informativa" del pre-registro). Menos elegante — no usa el belief del
student, mide exposición y no comprensión — pero cumple el espíritu del
profesor-que-no-repite, es 100% reproducible, y es **legítimo como ablation
del gating aunque P5 pase** (¿el belief agrega algo sobre la exposición
observacional?). El backward no se toca en este fallback (es robusto por
§4.1); lo único que cambia es la regla de R(t). Cumple el invariante 10
propuesto: la regla depende solo de (dataset, estado del simulador), sin
juicios de utilidad.

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

## 7. Pendientes (actualizado 2026-06-12, post-extracción)

1. ~~Descarga del dataset + correr el extractor~~ ✅ Hecho (mirror HF, §8).
2. ~~Adaptar el extractor al formato por componentes~~ ✅ Hecho
   (`decompose_goal_components` + `serialize_residual_block`, smoke-tested y
   corrido contra el dataset real).
3. **Verificación de cuasi-ortogonalidad de los g_i** (caveat KnowRL) — con
   datos de la Figura 1, no antes.
4. **τ global vs τ por tipo** (§4.1) — Figura 1.
5. **Longitud de g_prod (name) offline** — requiere merge con
   `items_shuffle.json` (5.5GB; diferido a la VM). **No bloquea nada**: en
   runtime el env construye el goal con `name` incluido (`goal.py:48-58`),
   así que la Figura 1 tiene g_prod disponible por episodio.
6. **Subset de ~1.6K tareas con trayectorias humanas** (criterio original
   pre-pivot de #17) — es otro artefacto (demos completas de Yao et al.), no
   está en `items_human_ins.json`. Post-pivot pertenece a la **ablation A de
   C.5** (fase 6), no al método. Se difiere a esa fase.

## 8. Datos reales (2026-06-12 — 12,087 goals, dataset completo)

Fuente: `items_human_ins.json` vía mirror HF verificado
(`YWZBrandon/webshop-data`, sha256 `cf786675...`, blob idéntico en segundo
mirror independiente; el Drive oficial está quota-exceeded). Corrido:
`tools/extract_webshop_specs.py` → `experiments/E000/webshop-specs.json`.
**12,087 goals** extraídos (número exacto del paper de WebShop), 164
skipeados sin atributos, 10,014 ASINs únicos.

| Métrica | Valor | Implicación |
|---|---|---|
| Attributes/goal | mediana **1** (dist: 1→8358, 2→3034, 3→539, ≥4→156) | La "spec rica" asumida por C.5 pre-pivot NO existe: 69.1% de goals tienen <2 attrs |
| Options/goal | mediana 1 (0→2482, 1→6369, 2→2936, ≥3→300) | 20.5% sin opciones |
| Keys de opciones recuperados | 70.8% como dict `{key: value}`; 29.2% keyless (fallback wrapper indexado) | El campo `options` del crudo permite recuperar el key posicionalmente; el resto usa el fallback de §4 |
| K por goal (sin g_prod ni g_price) | mediana **2** | Con g_prod (siempre disponible en runtime) y g_price (casi siempre instanciado por el env): **K efectivo ≈ 4** |
| Token budget del bloque full-golden | mediana 45, p99 60 tokens | Trivial — `max_prompt_length=4096` sobra; pregunta §2.3 del plan cerrada |
| Calidad de los values | Mayormente frases cortas limpias; casos sucios reales (IDs numéricos como "color", caracteres fullwidth `（width）`) | Casos de prueba concretos para la canonicalización N.8 y el matching textual de P2'/P5 |

### Lectura post-pivot (la que importa)

El criterio pre-pivot de C.5 ("se sostiene si mediana ≥ 2 attrs") **falla** —
pero ese criterio era para el forward monolítico, donde la spec necesitaba
ser "rica" para dar señal. El diseño dual+residual cambia la pregunta:

1. **Los attrs/options NO son información oculta**: salen de la instrucción
   humana, que el student VE en su prompt. Su "privilegio" es estructural
   (cuáles partes de la instrucción son criterios duros del reward del env),
   no informacional. Predicción derivada para la Figura 1: **b_attr y b_opt
   arrancan altos en t=0** (están en la instrucción) → el gating los saca
   del residual temprano; el auto-annealing sobre ellos es casi inmediato.
2. **La información genuinamente oculta de WebShop es g_prod** (qué producto
   del catálogo satisface la instrucción — el student no lo sabe hasta
   navegar) y, parcialmente, la combinación exacta de opciones de ese
   producto. Predicción derivada: **el backward en WebShop gana información
   principalmente vía Δb_prod**, y r_bwd debería concentrarse en las
   acciones que revelan el producto (search con buenos resultados,
   click-producto).
3. Esto **cuantifica el mapa dónde/por qué** antes de correr nada: WebShop
   es el environment de "PI mayormente visible en la instrucción" (poca
   info oculta, K efectivo chico ≈ 4, un solo componente verdaderamente
   oculto), ALFWorld el de estado oculto genuino. Si el dual gana en
   ALFWorld y empata en WebShop, esa asimetría ES el resultado del mapa —
   no un fracaso del método.
4. **C.5 se sostiene con esta reinterpretación** (la spec estructurada sigue
   siendo la PI correcta — es la que el reward del env puntúa), pero su
   fila en `design-decisions.md` queda actualizada con el dato real y la
   lectura post-pivot. El fallback pre-pivot del plan ("pivot a
   spec+trayectoria humana si mediana < 2") no aplica mecánicamente: el
   dual no necesita spec rica, necesita componentes verificables con al
   menos uno genuinamente oculto — y lo hay.
5. **Bonus para el mass-splitting** (§4.1): con mediana 1 atributo, el
   wrapper indexado `#k` no tiene ambigüedad en el 69% de los goals — el
   problema queda acotado a la cola de goals multi-atributo.
