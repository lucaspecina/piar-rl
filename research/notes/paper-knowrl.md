# KnowRL — Boosting LLM Reasoning via RL with Minimal-Sufficient Knowledge Guidance

> **Issue:** epic Pivot 2026-06 (ver Project v2) · **arxiv:** [2604.12627](https://arxiv.org/abs/2604.12627) · **Código:** [`Hasuer/KnowRL`](https://github.com/Hasuer/KnowRL) · **Autores:** Yu, Yang, Ding, Jin, Gu, Hao, Nie, Xiong, Yin, Sun, Wu (11 autores; afiliaciones ⚠️ no verificado — la página de abs no las listaba en el fetch). Submitted 14 abril 2026.
> **Rol en PIAR:** **vecino conceptual más cercano DEL LADO DEL STUDENT** (familia hints: QuestA, BREAD, Prefix-RFT, Scaf-GRPO...). Valida dos ideas centrales del residual de PIAR — descomposición atómica de la guía (≈ componentes $g_i$, decisión N.1) y dosificación mínima-suficiente (≈ gating por belief, N.4) — pero las implementa del lado estructuralmente opuesto: la información entra al **prompt del student** durante el rollout, no al canal del reward. Es el paper contra el cual se define el invariante nuevo 11 (N.3).

## 1. Setup y claim central

Single-turn math reasoning (verificado — sin componente multi-turn ni agéntico).
El problema que atacan: sparse reward en RL verificable hace que los problemas
duros (pass rate ≈ 0 del student) no aporten gradiente. La familia hints lo
resuelve inyectando ayuda al prompt del student durante el rollout; la crítica
de KnowRL a sus predecesores (QuestA usa un prefijo fijo del p% de la solución)
es que "largely overlook the issue of guidance redundancy": demasiada ayuda
contamina, muy poca no destraba.

Su reformulación: el diseño de hints es un problema de **guía
mínima-suficiente** — descomponer el conocimiento necesario en unidades
atómicas y buscar el subconjunto más chico que maximiza el pass rate del
student.

- Modelo: OpenMath-Nemotron-1.5B. Algoritmo: GRPO, reward puramente
  rule-based outcome (mathverify + CompassVerifier-3B como fallback en la
  eval offline). Sin reward denso, sin reward por paso.
- Data: dataset de QuestA, ~8.8k instancias dedup. 2,960 steps en 8×H100
  (~13 días).

## 2. Knowledge points — la descomposición atómica

Pipeline de tres etapas para construir $\mathcal{K} = \{k_1, \dots, k_n\}$
por problema:

1. **Extracción:** samplean DeepSeek-R1 hasta obtener ≥1 solución verificada;
   el mismo R1 extrae de ahí "solo los principios matemáticos indispensables
   requeridos para resolver el problema".
2. **Anti-leakage:** verifican que los KPs no contengan valores numéricos ni
   estructura específica de la instancia. "Failed cases are manually revised
   to ensure all retained KPs are generalizable and not instance-bound."
3. **Formato** (Apéndice C.2): nombre/descripción del concepto + "Key
   Considerations" (caveats de aplicación). Ejemplo: "Unit conversion between
   metric length units follows powers of 10... 10 mm = 1 cm...".

**Punto fino para PIAR:** los KPs de KnowRL son deliberadamente
*no-instance-bound* (principios generales, tipo "lemas reutilizables") porque
si el KP contiene la respuesta, el reward verificable se hackea por copia.
Los $g_i$ de PIAR son lo contrario en ese eje: **hechos de la instancia**
(producto target, atributos, ubicaciones de objetos). PIAR puede permitirse
$g_i$ instance-bound precisamente porque la PI nunca toca el prompt del
student — no hay nada que copiar. El anti-leakage de KnowRL es el síntoma
de estar del lado student (ver §6).

Lo que sí comparten: la restricción de que las unidades viven en el espacio
de conocimiento/hechos, no en el espacio de acciones ("hacé X"). Es el mismo
espíritu del invariante nuevo 10 (N.2), llegado por otra ruta: a KnowRL un
hint procedural le rompería la generalización; a PIAR le degeneraría el
reward en imitación del opinador.

## 3. Constrained Subset Search (CSS) — la dosificación

La búsqueda del subset mínimo-suficiente $S^* \subseteq \mathcal{K}$, hecha
**offline, antes del RL**, con el mismo student sin entrenar
(OpenMath-Nemotron-1.5B, 8 configuraciones × 32 samples por problema).
Criterio de suficiencia = **accuracy (pass rate) del student** con ese subset
en el prompt: $S^* = \arg\max_S A(S)$. No hay umbral fijo de "suficiente";
es maximización empírica.

### 3.1 La pruning interaction paradox

Hallazgo previo que motiva el algoritmo: con leave-one-out, remover cada
$k_i$ individualmente puede mejorar ($A_{-i} \geq \max(A_\mathcal{K}, A_\emptyset)$),
pero remover esos mismos KPs **en conjunto** degrada. Formalmente, para
$S \subseteq \mathcal{K}^+$, comparan $A_{\text{joint}}(S) = A(\mathcal{K} \setminus S)$
contra $\bar{A}_{\text{single}}(S)$ (promedio de los LOO individuales): la
paradoja ($A_{\text{joint}} < \bar{A}_{\text{single}}$) ocurre con
probabilidad ~40–60% según configuración. Causa diagnosticada:
"cross-hint inconsistency" — KPs interdependientes que se desambiguan
mutuamente. Moraleja: **la utilidad de los componentes de la guía no es
aditiva**; greedy/LOO ingenuo falla.

### 3.2 El algoritmo

Dos fases:
1. **Poda:** $\mathcal{H} = \{k_i \mid A_{-i} \geq \max(A_\mathcal{K}, A_\emptyset)\}$
   (no-degradantes), $\mathcal{N} = \{k_i \in \mathcal{H} \mid A_{-i} \geq A_{\max}\}$
   (remociones casi-óptimas, se descartan), $\mathcal{C} = \mathcal{H} \setminus \mathcal{N}$
   (candidatos).
2. **Búsqueda exhaustiva** sobre $2^{|\mathcal{C}|}$ (más $\emptyset$ y
   $\mathcal{K}$ completo), tractable porque la poda achica $|\mathcal{C}|$.

Resultado: promedio **2.57 KPs por problema** vs 5.86 iniciales. CSS gana en
la ablation de estrategias de selección (vs random, max-score, S-LOO, T-LOO,
CBRS): 63.90% accuracy offline. Los problemas fáciles reciben $S^* = \emptyset$
(cero hints); los duros, su subset mínimo — la dosificación es por problema,
no por umbral global de dificultad.

Costo: no cuantifican el overhead computacional del search (⚠️ no hay
sección de limitations en el paper).

## 4. Dónde entra la información — verificado: al prompt del student

Confirmado con texto del paper: "We added pre-curated KPs to prompts under
the `## Hint` header". El hint se antepone al problema **durante los rollouts
del RL**. La información entra a la política. (Si los tokens del hint se
enmascaran del loss: ⚠️ no verificado — el paper no lo especifica; usa
"token-mean loss" sin mencionar mask.)

### 4.1 Train/test mismatch y la (no-)retirada

El punto más interesante — y donde el paper real **difiere de la
caracterización genérica de la familia** en nuestro pivot doc (§2: "retirada
por schedule"):

- **KnowRL NO tiene annealing ni retirada programada.** Los subsets $S^*$ se
  fijan offline antes del RL y quedan constantes todo el training ("All
  construction and selection procedures are performed via offline evaluation
  before RL training"). No hay decay, ni curriculum temporal, ni schedule.
- En test, el modelo se evalúa **sin hints** por default. El mismatch
  train-con-hints / test-sin-hints existe y es total — KnowRL simplemente lo
  asume y muestra empíricamente que el policy mejorado sobrevive la retirada:
  "KnowRL improves the underlying policy itself, rather than relying only on
  test-time hint injection".
- No hay discusión explícita del distribution mismatch como problema en
  ninguna sección (⚠️ verificado por ausencia en el HTML).

Es decir: dentro de la familia hints, QuestA-style maneja el mismatch con
schedules de retirada; KnowRL lo maneja con **dosificación estática mínima**
(menos hint ⇒ menos mismatch que retirar) y cruza los dedos con el resto. La
mitigación es la minimalidad, no la retirada. El problema estructural
(la política entrenada vio un input que en deployment no existe) queda sin
resolver de raíz — solo amortiguado.

## 5. Resultados

8 benchmarks de math competition (AIME24/25, BRUMO25, HMMT-Feb-25, AMC23,
CMIMC25, MATH-500, OlympiadBench):

| Config | Avg |
|---|---|
| Nemotron-1.5B base | 60.45% |
| QuestA, JustRL | baselines intermedios |
| **KnowRL, eval sin hints** | **70.08%** (+9.63pp) |
| KnowRL, eval con hints CSS | 74.16% (+13.71pp) |

La fila clave para su claim es la de **70.08% sin hints**: la guía durante
training mejoró la política en sí, no solo la performance condicionada al
hint. El gap de ~4pp entre con/sin hints en test es la medida honesta del
residuo de dependencia que les queda.

## 6. El contraste estructural: lado-student vs lado-scorer

Esta es la sección que justifica el invariante nuevo 11 (N.3). Mismo par de
ideas (descomposición atómica + dosificación mínima-suficiente), dos
arquitecturas de inyección incompatibles:

| | KnowRL (familia hints, lado student) | PIAR residual (lado scorer) |
|---|---|---|
| Dónde entra la info | Prompt del **student** durante el rollout | Solo prompts del **scorer** ($\pi_{\text{old}}$ con dos contextos); el student rollea a ciegas |
| Qué contamina | La **política**: $\pi(a \mid \text{problema}, \text{hint})$ | Nada: la política nunca condiciona en PI |
| Train/test mismatch | Sí, del **input** (hint en train, no en test) | No existe: el input del student es idéntico en train y test |
| Qué hay que retirar | El hint del prompt (KnowRL: no lo retira, lo minimiza estático) | Nada que retirar del input; lo que se apaga (solo) es la **señal**: $R(t) = \emptyset \Rightarrow r_{\text{fwd}} = 0$ |
| Dosificación | **Estática y offline**: CSS una vez antes del RL, por problema; ciega al progreso durante el training | **Dinámica y online**: gating por $b_i(t-1) < \tau$, por componente, por turno; sigue al student durante el training |
| Granularidad temporal | Por problema (single-turn: no hay turnos) | Por turno dentro del episodio + a lo largo del training |
| Unidades | KPs: principios generales, anti instance-bound (si no, copia) | $g_i$: hechos de la instancia (pueden serlo: no hay canal de copia) |
| Anti-leakage | Necesario sobre el **contenido** del hint (revisión manual) | Necesario sobre el **score** (controles D.1/D.9), no sobre el contenido |
| Costo de la dosificación | 8×32 rollouts offline por problema + búsqueda $2^{|\mathcal{C}|}$ | Forward passes de belief con prefix caching, cero rollouts extra |
| Dominio validado | Math single-turn, 1.5B | (target) agentes multi-turn WebShop/ALFWorld |

Dos lecturas:

1. **La paradoja de pruning es evidencia a favor de la dosificación adaptativa.**
   KnowRL muestra que el subset óptimo de guía no es trivial (no-aditividad,
   interdependencias) — y su solución es una búsqueda combinatoria offline
   carísima por problema. El gating por belief de PIAR ataca el mismo
   problema ("¿qué componentes hacen falta?") con un criterio distinto: no
   "qué subset maximiza el pass rate" sino "qué componentes el student
   todavía no sabe", medido online y gratis. Caveat honesto: PIAR no resuelve
   la interdependencia entre $g_i$ (el gating es por componente,
   independiente); si los $g_i$ de WebShop/ALFWorld resultan tan
   interdependientes como los KPs de math, la versión naïve del gating puede
   tener su propia paradoja. Anotado como riesgo menor — los $g_i$ fácticos
   de PIAR (atributos del producto, ubicaciones) son más ortogonales por
   construcción que principios matemáticos encadenados.

2. **El auto-annealing emergente de PIAR es exactamente lo que a la familia
   hints le falta.** KnowRL evita el schedule de retirada congelando una
   dosis mínima — pero la dosis correcta *cambia* a medida que el student
   aprende, y CSS no puede verlo (se computó contra el student pre-RL). En
   PIAR la retirada es por construcción: cuando $b_i \geq \tau$ para todo
   $i$, los dos prompts del scorer son idénticos y la señal forward es
   exactamente cero, componente a componente, en el orden en que el student
   aprende. Sin búsqueda, sin schedule, sin mismatch.

## 7. Lo que esto significa PARA PIAR (post-pivot)

1. **Valida N.1 (descomposición atómica de la PI).** Un paper publicado e
   independiente muestra que descomponer la guía en unidades atómicas
   verificables y dosificarla supera tanto a guía completa como a guía nula
   (+13.7pp vs base, y la dosificada gana a $\mathcal{K}$ completo en la eval
   offline). El paralelismo KPs ↔ $g_i$ es directo y citable en related work.
2. **Valida la dosificación mínima-suficiente como idea** — pero la
   implementa del lado equivocado para RL de agentes: estática, offline,
   combinatoriamente cara, y con la información contaminando la política. El
   residual de PIAR toma las dos ideas (átomos + dosis mínima) sin tocar el
   prompt del student: la PI vive solo en el canal del reward, el student
   rollea a ciegas, y el annealing es emergente vía gating por belief en
   lugar de búsqueda offline congelada. Ese contraste ES el related work del
   residual.
3. **Refuerza el invariante 11 (N.3) con evidencia concreta del costo del
   lado student:** anti-leakage manual sobre el contenido, train/test
   mismatch asumido sin resolver, gap residual de ~4pp con/sin hints en
   test, y dosis que no puede adaptarse al progreso del training.
4. **Corrección al pivot doc (§2):** la familia hints no es monolítica en
   "retirada por schedule". KnowRL específicamente NO programa retirada —
   congela una dosis mínima estática. El argumento estructural de PIAR no
   cambia (la info sigue entrando a la política y el mismatch sigue
   existiendo), pero la frase de related work debe ser precisa: la familia
   maneja el mismatch *por schedule (QuestA-style) o por minimalidad
   estática (KnowRL)*; ninguna lo elimina.
5. **Riesgo nuevo menor para anotar:** interdependencia entre $g_i$
   (pruning paradox trasplantada al gating). Mitigación natural: los $g_i$
   fácticos de WebShop son cuasi-ortogonales; verificable en la Figura 1
   (día-1, sin training) mirando correlaciones entre los $b_i$.
6. **No es baseline ni amenaza de scoop:** dominio disjunto (math
   single-turn, 1.5B, sin agentes, sin multi-turn, sin step reward — el
   reward de KnowRL es outcome-only rule-based). Es related work que valida
   componentes, igual que TAMTRL/IGPO para las direcciones del reward.

## 8. Lo más importante para retener

1. **KnowRL = átomos + subset mínimo-suficiente, del lado del student.**
   Descompone la guía en knowledge points atómicos (extraídos por DeepSeek-R1
   de soluciones verificadas, anti instance-bound), busca offline el subset
   que maximiza el pass rate del student (CSS, ~2.6 de ~5.9 KPs), y lo
   inyecta fijo al prompt durante GRPO. Math single-turn, Nemotron-1.5B,
   60.45% → 70.08% (test sin hints).
2. **La pruning paradox:** la utilidad de los componentes de guía no es
   aditiva (remociones individuales ayudan, conjuntas degradan, ~40–60% de
   los casos). Motiva búsqueda global sobre candidatos podados — y para PIAR,
   anticipa que el gating por componente independiente podría necesitar
   revisión si los $g_i$ resultan interdependientes.
3. **No hay annealing en KnowRL** (a diferencia de lo que el pivot doc
   generaliza sobre la familia): dosis estática offline + test sin hints +
   mismatch asumido. La dosis no se adapta al progreso del student durante
   el RL — exactamente el hueco que el gating dinámico por belief de PIAR
   llena, del otro lado de la arquitectura.
4. **El contraste de §6 es la tabla de related work del residual:** misma
   pareja de ideas, lados opuestos del pipeline; en KnowRL la información
   moldea la política y deja mismatch + gap residual; en PIAR moldea solo la
   señal y la retirada es un teorema trivial ($R = \emptyset \Rightarrow
   r_{\text{fwd}} = 0$), no un hiperparámetro.
