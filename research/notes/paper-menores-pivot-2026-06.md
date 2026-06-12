# Vecinos menores del pivot 2026-06 — nota conjunta

> **Nota conjunta de vecinos menores del pivot 2026-06.** Cubre HCAPO, IG
> semántico, QuestA y OPCD/OEL (vía survey de on-policy distillation) en ~1
> página cada uno. Para los vecinos mayores ver las notas dedicadas (IGPO,
> TAMTRL, survey CA, GiGPO, KnowRL — pendientes al momento de escribir esta).
> **Issue:** epic Pivot 2026-06 (ver Project v2) · **Fuente de verdad del pivot:**
> [`research/synthesis/pivot-2026-06.md`](../synthesis/pivot-2026-06.md)
> **Rol en PIAR:** ninguno de estos 4 mata componentes del diseño post-pivot;
> los cuatro acotan el related work y validan la distinción estructural
> reward-channel vs prompt-channel (invariante 11 propuesto, N.3).

---

## 1. HCAPO — Hindsight Credit Assignment Policy Optimization

> **arxiv:** [2603.08754](https://arxiv.org/abs/2603.08754) · **Autores:** Tan, Yang, Chen, Shao, Wen, Shen, Luo, Du, Guo, Li. Marzo 2026 — uno de los **3 papers de hindsight/counterfactual de la misma semana** (con C3 y CCPO) que el survey CA (2604.09459) usa como evidencia de la velocidad del nicho.

### Mecanismo

> ⚠️ **Enmendado 2026-06-12 tras verificación independiente** (segunda lectura
> a pedido de review externo). La primera versión de esta sección describía a
> HCAPO como "importance ratio same-model dos-contextos" — eso es la **forma
> teórica (Eq. 5, derivada de HCA, Harutyunyan et al. 2019)**, no lo que el
> paper implementa.

El propio LLM del agente, prompteado (sin entrenamiento de un crítico
separado; ⚠️ el paper usa la notación `π_θ` y **no afirma explícitamente que
esté congelado** — podría ser la política viva re-scoreando), actúa como
crítico post-hoc. Inyectan el **outcome realizado** `s_final` (auto-generado
por el propio rollout) al contexto y leen **logprobs token-level** de la
acción:

    π_hind(a_t) = exp( 1/(T·|a_t|) · Σ_j log π_θ(y_j | y_<j, s_t, s_final) )

**Lo que implementan (Eq. 7) tiene UN solo contexto**: el denominador no es
π(a_t | s_t) sino la **media empírica intra-trayectoria** de los propios
scores de hindsight —

    ρ_t = clip( π_hind(a_t) / π̄_hind , 0.8, 1.2 ),   π̄_hind = (1/T) Σ_k π_hind(a_k)

("Since this marginalization is intractable, we approximate it using the
empirical mean of hindsight scores within a trajectory"). La señal es
**necesidad relativa de cada acción dentro de la trayectoria bajo contexto de
hindsight**, no un delta de información entre contexto-con-PI y
contexto-sin-PI. Refina el return: `Q^H_t = ρ_t · G_t`; advantage multi-escala
(macro GRPO de 8 rollouts + `ω·` término micro normalizado), PPO clipped con
KL. Aplican una máscara **"do-no-harm"** que anula señales de hindsight
negativas en trayectorias exitosas. ⚠️ No verificados en la segunda lectura:
T = 5.0, ω = 1.0 final, β_KL = 0.01, α = 0.5 del smoothing, overhead +8.3%,
Search-QA 48.3 (afirmados en la primera; no contradichos).

### Números

Qwen2.5-7B-Instruct. ALFWorld **91.4±2.3** — pero el headline +13.8 es **solo
vs GRPO vanilla** (77.6±5.2); **vs GiGPO (90.8±1.3 en la propia tabla de
HCAPO) la ganancia es +0.6 con barras de error solapadas**. WebShop SR
73.8±2.8 vs GRPO 66.1±3.7 (vs GiGPO 72.8±3.2 — ídem, margen chico). 96.9% con
smoothing temporal. ⚠️ El paper **no desambigua el split seen/unseen de
ALFWorld** (delega el protocolo a GiGPO, que tampoco lo explicita en el HTML)
y la config está **cerca de saturación** (>90%) — alerta para nuestro uso de
ALFWorld como benchmark del backward (¿hace falta un tercer environment con
info oculta de verdad? → N.13). Status: **preprint v1 (marzo 2026, template
ICML), sin statement de aceptación y sin código público encontrado**.

### Contraste con PIAR

⚠️ **Doble corrección acumulada**: (1) contra la tabla §2 del pivot — HCAPO
**no es juez generativo**: no emite scores opinados por generación, el canal
es logprob mecánico. (2) Contra la primera versión de esta nota — **tampoco
es un ratio dos-contextos**: solo computa logprobs bajo el contexto de
hindsight y se auto-normaliza intra-trayectoria (Eq. 7). La caracterización
justa para related work: *"scoring de necesidad bajo contexto de hindsight,
normalizado intra-trayectoria, motivado teóricamente por el ratio HCA"*.
Sigue siendo el vecino mecánicamente más cercano de estos 4, pero el delta
de PIAR es mayor de lo que la primera lectura sugería.

Diferencias reales que lo separan del forward (y del método post-pivot):

1. **Qué es la PI**: el outcome realizado *auto-generado por el propio rollout*
   (hindsight puro), no información privilegiada fáctica del dataset/simulador
   (golden estructurada en componentes g_i). HCAPO no usa PI externa.
2. **Qué se computa**: score de un solo contexto normalizado contra la media
   de la trayectoria, vs **contraste explícito de dos contextos** (con-PI /
   sin-PI) — el objeto que define al forward de PIAR. HCAPO nunca computa el
   delta de información.
3. **Canal**: multiplicativo (corrección del return, clipped [0.8, 1.2]) vs
   aditivo (step reward dentro del advantage).
4. **Hindsight bias por diseño**: premia acciones correlacionadas con el éxito
   ya realizado — exactamente el sesgo que el pivot §4.3 identifica como lo que
   el forward NO arregla y el backward compensa. HCAPO no tiene componente
   epistémico ni dosificación: ni backward, ni residual, ni mapa por
   environment. La máscara do-no-harm restringe además el refinamiento a
   trayectorias exitosas (⚠️ tratamiento de fallidas no claro) — pariente del
   gating por outcome de TAMTRL: redistribuye crédito dentro del éxito.
5. **Solapamiento de benchmarks**: usa WebShop y ALFWorld con Qwen2.5-7B —
   nuestros mismos targets. Referencia de magnitudes para A1/A3, con el caveat
   de saturación de arriba.

---

## 2. IG semántico — Synthetic Semantic Information Gain Reward

> **arxiv:** [2602.00845](https://arxiv.org/abs/2602.00845) — "Optimizing Agentic Reasoning with Retrieval via Synthetic Semantic Information Gain Reward" · **Autores:** Hu, Dai, Zhao, Tao, Guo, Fang, Kwong, Fang. Febrero 2026 — **sucesor de IGPO a ~4 meses** de su publicación.

### Mecanismo (verificado a nivel abstract; HTML no disponible)

Redefine el information gain como **reducción de incertidumbre sobre los
belief states del modelo**, medida vía **clustering semántico por bidirectional
textual entailment** (la maquinaria de semantic entropy) — explícitamente *en
lugar de* logprobs de la golden answer. Propiedades teóricas declaradas:
**no-negatividad, aditividad telescópica y monotonicidad de canal**.
Optimización con GRPO. Dominio: razonamiento agentic con retrieval; 7
benchmarks de QA; hasta **+5.4% accuracy promedio** sobre baselines
retrieval-augmented.

⚠️ **No verificado** (la versión HTML del paper devuelve 404 y solo accedí al
abstract): si cita a IGPO explícitamente, el significado exacto de "synthetic"
(presumiblemente muestrear respuestas del modelo para estimar la distribución
de belief — sin confirmar), la mecánica por-turno del reward y la normalización
(separada vs conjunta con outcome).

### Qué difiere del IG de IGPO y qué confirma

- **Semántico vs logprob**: IGPO mide Δ log π(golden | historial) — necesita la
  golden y es sensible a parafraseo de superficie. Este paper mide colapso de
  la *distribución semántica de respuestas muestreadas* — agrega invarianza a
  parafraseo vía entailment. Es exactamente la debilidad que nuestra N.8
  (canonicalización de g_i) mitiga por la vía estructurada; el clustering por
  entailment queda como mitigación alternativa citable (y posible ablation
  futura del belief b_i).
- **Dominio**: retrieval/QA, igual que IGPO. La línea information-theoretic
  **sigue confinada a search/QA** — agentes de decisión/ejecución multi-turn
  (ALFWorld, WebShop) siguen libres para el backward trasplantado (brazo A2).
- **Telescoping como selling point estándar**: declaran la misma propiedad que
  nuestro §4.2. Confirma que la garantía teórica del backward es *table
  stakes* del nicho, no diferenciador — el diferenciador es la combinación
  dual + residual + mapa.
- **Velocidad del nicho del backward, confirmada**: sucesor refinando a IGPO en
  ~4 meses. Refuerza el riesgo de scoop §10.5 del pivot y la decisión de
  re-check mensual de literatura.

---

## 3. QuestA — Question Augmentation (hints parciales lado student)

> **arxiv:** [2507.13266](https://arxiv.org/abs/2507.13266) — "QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation" · **Autores:** Li, Lin, Lu, Wen, Yang, Gao, Wu, Zhang. Julio 2025. Ejemplar canónico de la **familia hints** de la tabla §2 del pivot.

### Mecanismo de scaffolding parcial

Para problema `x` con solución de n pasos `y`, construye prompts aumentados
`x̃^(p)` prependeando los **primeros p% de tokens de la solución** como
"Hint: Partial Solution Analysis" en el prompt **del student** durante el
rollout de RL. Usan p = 50% y p = 25%. Doble filtrado de dificultad: 220K
problemas → 26K (filtro con R1-Distill-1.5B) → solo problemas donde el base
acierta 0–1 de 8 intentos (~10K finales). GRPO + técnicas DAPO,
Nemotron-1.5B, 16 respuestas/prompt, math.

### Schedule de retirada

**Curriculum en dos etapas, programado a mano**: 100 steps con p = 50% →
1900 steps con p = 25%. El punto de transición se elige monitoreando entropía
("entropy... begins to decline beyond 100 steps, so transitioning at this
point prevents overconfidence"). Justificación explícita: "the most
appropriate inference distribution for the model should be the original
(no-hint) distribution. Hence, during training we should gradually reduce
reliance on hints". Un tercer stage a p = 0 **no dio ganancia**. En
evaluación, **cero hints**.

### Números

AIME24 72.50% (+10.73), AIME25 62.29% (+12.79), HMMT25 41.67% (+10.11) con
1.5B — compite con distills de 32B. pass@k se preserva (argumentan capacidad
real, no overfitting al hint).

### Contraste con N.3 / invariante 11 propuesto

QuestA es la materialización exacta de lo que el pivot §2 llama "familia
estructuralmente opuesta al residual":

| | QuestA (hints lado student) | PIAR (PI lado scorer) |
|---|---|---|
| Dónde entra la PI | Prompt del student → contamina la política | Solo prompts del scorer → student rollea a ciegas |
| Train/test mismatch | Sí (input distinto en eval) → obliga retirada | No hay nada que retirar del input |
| Retirada | Schedule manual (100/1900 steps, p discreto y global, calibrado por entropía) | Auto-annealing emergente, por componente, gateado por belief medido (R(t) = ∅ ⇒ r_fwd = 0) |
| Espacio de la PI | Prefijo de la *solución* (espacio de acciones/razonamiento) | Hechos verificables g_i (invariante 10: nunca espacio de acciones) |
| Qué se apaga al final | El input | Solo la señal |

Nota honesta para related work: QuestA demuestra que el lado-student
**funciona** en math single-turn con problemas casi imposibles — el argumento
de PIAR no es "los hints no funcionan", sino que el canal reward elimina el
mismatch estructural y reemplaza el schedule manual por dosificación medida.
Eso es una hipótesis a demostrar (brazo A4), no un hecho. El dato de que
p = 0 no aportó nada en QuestA es interesante: sugiere que la retirada total
programada es delicada — el residual la hace continua y por componente.

---

## 4. OPCD / OEL — privileged context como distillation loss (vía survey arXiv 2604.00626)

> **arxiv del survey:** [2604.00626](https://arxiv.org/abs/2604.00626) — "A Survey of On-Policy Distillation for Large Language Models" · **Autores:** Song & Zheng. Nota: OPCD y OEL no tienen nota propia; lo que sigue es lo extraído del survey (no de los papers originales).

### Qué son (según el survey, Table 2)

- **OPCD — On-Policy Context Distillation** (Ye et al., 2026b): el mismo
  modelo con **contexto aumentado** actúa de teacher y destila vía **reverse-KL**
  al mismo modelo sin ese contexto, sobre rollouts on-policy del student.
- **OEL — Online Experiential Learning** (Ye et al., 2026a): combina KL +
  context distillation con aprendizaje experiencial, también same-model.

⚠️ Matiz de clasificación: en el survey, la subcategoría explícita
"Self-Distillation → Privileged Information" lista **OPSD** (Zhao et al.,
2026 — teacher condicionado a la respuesta correcta, F-KL/JSD sobre rollouts
del student), **GATES** (Stein et al., 2026 — PI = documento relevante
solo-training, con *consensus gating* que activa la señal solo cuando hay
acuerdo entre múltiples muestras del teacher) y **Privileged Information
Distillation** (Penaloza et al., 2026 — framework general). OPCD/OEL aparecen
como self-distillation por contexto aumentado; la agrupación funcional que
hace el pivot doc (misma asimetría same-model-dos-contextos) es correcta,
pero la etiqueta "privileged" formal del survey no los incluye.

⚠️ También: el survey lista un "π-Play" (Zhang et al., 2026h, "Multi-Agent
Self-Play via Privileged Self-Distillation") — no pude verificar si es el
mismo trabajo que nuestro π-Distill ([`paper-pi-distill.md`](paper-pi-distill.md))
u otro de la misma línea.

### Relación con OPSD / π-Distill (que ya tenemos en notas)

Toda la familia comparte la primitiva de PIAR — un solo modelo, la asimetría
vive en el contexto — y difiere en el **canal**: la asimetría se consume como
**loss de distillation** (KL token-level que empuja la política del student
hacia la del teacher privilegiado), no como **reward dentro del advantage**.
OPCD/OEL extienden la familia de OPSD/π-Distill con variantes de divergencia
(R-KL) y de fuente de contexto, confirmando que la rama *loss* está poblada y
activa en 2026.

### Por qué rodean al forward sin matarlo

1. **Verificado en el survey**: NO registra **ningún** método que use la PI
   como *reward* — todos los privileged/context-distillation usan KL como
   loss. El cruce "PI → canal reward" está vacío en su mapa, igual que el
   cruce dual estaba vacío en el survey de CA. Eso es lo que deja vivo al
   forward *como componente* del método post-pivot.
2. **Diferencia funcional del canal**: la loss KL es imitación densa
   obligatoria (matching token a token hacia el teacher); el reward en el
   advantage se mezcla con el outcome, tolera desacuerdo con el teacher, y
   admite exactamente las operaciones que definen al pivot — apagarse por
   componente (residual), combinarse con el backward (dual), y entrar al
   mismo hook que iStar.
3. **Caveat para related work**: el survey cita la equivalencia formal
   OPD ↔ KL-RL (MiniLLM: minimizar reverse-KL secuencial = policy gradient
   con reward per-step; G-OPD formaliza vía constrained MDP). En review nos
   van a preguntar en qué se diferencia el reward de PIAR del "reward
   implícito" de esa equivalencia: respuesta — el de PIAR no es el gradiente
   de una KL hacia el teacher; es log-ratio sumado por span de acción,
   combinado con un término epistémico que la distillation no puede expresar
   (para el teacher privilegiado, informarse vale cero), y dosificado por
   belief.
4. **GATES como pariente del residual**: su consensus gating también dosifica
   la señal privilegiada — pero gatea **una loss** por **acuerdo del teacher**,
   no **la PI misma** por **belief medido del student**. Buen contraste para
   el related work del residual: la idea de "no toda la señal privilegiada,
   todo el tiempo" está en el aire; nuestra instanciación (gating epistémico
   lado scorer) no aparece.

---

## Lo que estos vecinos significan PARA PIAR (post-pivot)

Releyendo la tabla §2 y el veredicto §3 del pivot doc contra lo verificado:

1. **El veredicto §3 se sostiene**: ninguno de los 4 toca la descomposición
   dual, la PI residual gateada por belief, ni el mapa por environment. El
   cuadrante "PI fáctica → canal reward → dual → dosificada" sigue vacío
   tanto en el survey de CA como en el survey de OPD.
2. **Una corrección a la tabla §2** (enmendada 2026-06-12 tras verificación
   independiente): HCAPO no es "juez generativo/opinado" — es scoring
   mecánico de logprobs same-model bajo contexto de hindsight, **normalizado
   intra-trayectoria (un solo contexto, Eq. 7)**, no un ratio dos-contextos
   (eso es solo su forma teórica Eq. 5). Sigue siendo el vecino mecánicamente
   más cercano de estos 4 y el primero a diferenciar en related work:
   (i) PI fáctica del dataset vs outcome auto-generado en hindsight,
   (ii) contraste explícito con-PI/sin-PI vs score de un contexto normalizado
   contra la media de la trayectoria, (iii) aditivo en advantage vs
   multiplicativo en return, (iv) dual + residual vs forward-hindsight puro.
3. **El backward corre contra reloj**: IG semántico refina a IGPO en ~4 meses
   y con garantías teóricas equivalentes a las nuestras (telescoping incluido)
   — pero sigue en retrieval/QA. La ventana en ejecución multi-turn está
   abierta; el re-check mensual (§10.5) no es opcional.
4. **El invariante 11 tiene su contraste empírico canónico**: QuestA documenta
   con precisión el costo estructural del lado-student (curriculum manual en
   dos etapas, calibración por entropía, retirada total que no aporta). Es la
   cita exacta para "lo que se apaga en PIAR es la señal, no el input".
5. **La rama loss valida la primitiva y cede el canal**: OPSD, π-Distill,
   OPCD, OEL, GATES y PID pueblan densamente "same-model + contexto
   privilegiado → KL loss" y dejan explícitamente despoblado "→ reward". La
   equivalencia OPD↔KL-RL obliga a una diferenciación técnica fina en el
   paper (punto 3 de la sección 4), pero no invalida nada.

## Lo más importante para retener

1. **HCAPO, caracterización final post-verificación**: scoring mecánico de
   logprobs same-model bajo contexto de hindsight, normalizado
   intra-trayectoria — ni juez generativo (error del pivot doc) ni ratio
   dos-contextos (error de la primera versión de esta nota). Es el vecino
   más peligroso de los 4 en review y el primero a diferenciar: hindsight
   auto-generado + un-solo-contexto + multiplicativo + sin backward/residual.
   Comparte nuestros benchmarks exactos (WebShop/ALFWorld, Qwen2.5-7B), pero
   su ganancia real vs GiGPO es +0.6 con barras solapadas, ALFWorld está
   cerca de saturación en esa config, y es preprint sin código — verificar
   números propios antes de citarlos como referencia dura.
2. **El nicho del backward se mueve en ciclos de ~4 meses** (IGPO → IG
   semántico), pero sigue encerrado en retrieval/QA. ALFWorld/ejecución sigue
   libre. Velocidad > perfección.
3. **QuestA = el contraste estructural del invariante 11**: hints al prompt
   del student funcionan pero exigen schedule manual y retirada delicada
   (p=0 no aporta). El residual reemplaza eso por auto-annealing medido, por
   componente, sin tocar el input. Es hipótesis del brazo A4, no hecho.
4. **El survey de OPD confirma el hueco**: cero métodos con PI como reward;
   toda la familia privilegiada va por KL loss. Pero la equivalencia
   OPD↔KL-RL (MiniLLM/G-OPD) exige explicar en related work por qué el reward
   de PIAR no es esa misma KL con otro nombre (respuesta: backward + residual
   + mezcla con outcome no son expresables como distillation hacia un teacher
   privilegiado).
5. **Lagunas marcadas ⚠️**: del IG semántico solo verifiqué el abstract (sin
   HTML); de OPCD/OEL solo lo que dice el survey (no los papers originales);
   "π-Play" del survey vs nuestro π-Distill sin confirmar si son el mismo
   trabajo.
