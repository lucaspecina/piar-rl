# Survey CA — From Reasoning to Agentic: Credit Assignment in RL for LLMs

> **Issue:** epic Pivot 2026-06 (ver Project v2) · **arxiv:** [2604.09459](https://arxiv.org/abs/2604.09459) (v1 10-abr-2026, v2 13-abr-2026) · **Código/repo:** [`xxzcc/Awesome-Credit-Assignment-in-LLM-RL`](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL) (lista curada viva, último update visible 2026-05) · **Autor:** Chenchen Zhang (autor único).
> **Rol en PIAR:** **documento de due diligence del pivot 2026-06.** Es el mapa de 47 métodos contra el que verificamos que (a) la descomposición dual pragmático/epistémico y (b) la PI residual gateada por belief NO existen en la literatura, y que (c) el "mapa dónde/por qué" es una frontera abierta declarada por el propio survey. Además define el protocolo de re-check mensual (riesgo de scoop, pivot §10.5).

## 1. Qué es y qué cubre

Survey dedicado a credit assignment (CA) en RL para LLMs, enero 2024 → abril 2026.
**47 métodos = 41 core CA + 6 "CA-adjacent enablers"** (infraestructura, reward
shaping, frameworks). Dos regímenes:

- **Reasoning RL**: 1 turno, 500–30K+ tokens, transiciones determinísticas, steps verificables.
- **Agentic RL**: 10–100+ turnos, 100K–1M tokens, POMDP, transiciones estocásticas, estados intermedios NO verificables.

Tesis central (abstract, verificado textual):

> "agentic CA is driving genuinely new approaches—hindsight counterfactual
> analysis, privileged asymmetric critics, and turn-level MDP
> reformulations—that have no direct precedent in reasoning RL."

Protocolo de búsqueda (§1.1): keyword search arXiv/Semantic Scholar/Google
Scholar + citation chasing desde VinePPO/ArCHer/GRPO/R1 + monitoreo de venues +
HF Daily Papers. Snapshot a abril 2026. **Caveat de peso de evidencia (§9.4):
autor único** — "All screening, classification, and evidence-level coding was
performed by the single author"; preprint volatility declarada. El survey es un
mapa útil, no un censo perfecto (ver §10 abajo: lo que NO cubre).

## 2. Taxonomía bidimensional (§2.4)

Dos ejes ortogonales (Figura 2 del paper):

1. **Granularidad** (vertical): token / segment / step-turn / multi-agent.
2. **Metodología** (horizontal): Monte Carlo / Temporal Difference / Model-based–LLM-as-Critic / Game-theoretic / **Information-theoretic**.

Estructura de secciones (verificada contra el HTML completo):

- §3 Reasoning RL: token-level (§3.1), segment-level (§3.2), step-level (§3.3).
- §4 Por qué agentic reshapea el CA (6 challenges, ver §8 abajo).
- §5 Agentic RL — **8 subsecciones** (acá vive todo lo que toca a PIAR):

| § | Familia | Métodos |
|---|---|---|
| 5.1 | Turn-Level PRMs | AgentPRM, SWEET-RL, Turn-Level Reward Design, Turn-PPO, SORL, TARL, ITPO |
| 5.2 | Hindsight & Counterfactual | HCAPO, C3, CCPO, CriticSearch |
| 5.3 | Critic-Free Step-Level | GiGPO, POAD |
| 5.4 | Hierarchical | ArCHer, PilotRL, CARL |
| **5.5** | **Information-Theoretic** | **IGPO (solo — único método de la categoría)** |
| **5.6** | **Implicit & DPO-Based** | **iStar**, StepAgent |
| 5.7 | Infrastructure & Practical | Agent Lightning, RAGEN/StarPO, SPA-RL, SCRIBE, LaRe, PRS+VSPO, AdaptSeg |
| 5.8 | Discussion: emerging patterns | — |

- §6 Multi-agent (M-GRPO, LLM-MCA, QLLM, SHARP, MAPPA, Dr. MAS).
- §7 Systematic Comparison · §8 Pipeline · §9 Open Problems.

**Verificación del claim del pivot doc:** ✅ exacto. IGPO está **solo** en §5.5
(information-theoretic) e iStar está en §5.6 (implicit/DPO-based) junto a
StepAgent. Son categorías disjuntas que el survey nunca cruza. La combinación
forward-pragmático × information-gain es literalmente una celda vacía de la
taxonomía.

## 3. Los métodos con información privilegiada (la competencia directa)

Búsqueda exhaustiva de "privileged" / "golden" / "gold answer" en el texto
completo. Solo DOS métodos usan PI en el scorer:

### 3.1 SWEET-RL (§5.1) — privileged critic ENTRENADO

> "SWEET-RL trains a critic that conditions on this privileged
> information—specifically, the ground truth answer, the complete future
> trajectory, and possibly environment state variables—to provide high-quality
> turn-level reward signals, which are then used for DPO-style optimization."

Privilegio en **pesos de un critic entrenado** (= la familia de iStar en
cuanto a "dónde vive la asimetría"). No es log-ratio del mismo modelo.

### 3.2 CriticSearch (§5.2) — privileged critic PROMPTEADO

> "A frozen, asymmetric critique LLM retrospectively evaluates each search
> turn using privileged information (the full trajectory and gold answers),
> converting these assessments into dense, turn-level rewards."

El vecino más cercano del forward viejo de PIAR dentro del survey: PI (gold
answer) en el prompt de un crítico congelado. Diferencias: el score es
**generativo/opinado** (el crítico razona y emite un juicio), no un log-ratio
mecánico entre dos contextos del mismo modelo; y está especializado a search
agents. Consistente con la tabla §2 del pivot doc.

**Ningún otro método del survey mete PI en el canal del reward.** Y ninguno —
ni SWEET-RL ni CriticSearch — descompone la PI en componentes ni dosifica qué
ve el scorer según el estado de conocimiento del student. El gating por belief
no aparece en ninguna celda.

## 4. IGPO según el survey (§5.5)

Formulación que da el survey (Eq. 4 del paper):

$$c_t = \log P(\text{success} \mid h_{1:t}) - \log P(\text{success} \mid h_{1:t-1})$$

> "The probability $P(\text{success}|h)$ is estimated by a learned verifier or
> the LLM itself. IGPO's main limitation is its requirement for a reliable
> success probability estimator at each turn."

⚠️ **Discrepancia de rendering, no de fondo:** el survey abstrae IGPO como
"probabilidad de éxito", mientras que el paper original de IGPO (2510.14967) y
nuestro backward usan $\Delta \log \pi(\text{golden} \mid h)$ — la probabilidad
de la *respuesta ground-truth*, no de "success" genérico. Para la equivalencia
exacta con nuestro $r_{\text{bwd}}$ hay que citar el paper de IGPO
(`paper-igpo.md`), no el survey. En Table 5 el survey le asigna a IGPO
`Aux. Model = Verifier` — bajo nuestra instanciación (belief del propio
$\pi_{\text{old}}$, sin verifier entrenado) esa celda cambiaría a `No`.

Las referencias del survey confirman el renombre: el título citado es
*"Information gain-based policy optimization: a simple and effective approach
for **multi-turn search agents**"* — ejecución/embodied quedó fuera de su claim,
como dice el pivot doc.

## 5. iStar según el survey (§5.6)

> "iStar (Implicit Step Rewards) … leverages trajectory-level DPO: given pairs
> of trajectories (one successful, one not), iStar extracts implicit step-level
> rewards by comparing the log-probability ratios at each turn. Building on the
> 'From r to Q*' insight … iStar further introduces multi-level advantage
> fusion, combining turn-level and token-level implicit signals."

Lo que el survey destaca: "requires no explicit reward model, critic, or
environment re-execution". En Table 5: `Aux. Model = No`, `Compute = Low` —
la misma celda de presupuesto donde queremos vivir nosotros. El survey NO
menciona que el juez de iStar tiene privilegio implícito vía los pares
success/failure (nuestro punto del par mínimo C.2 / outcome-label-in-prompt
sigue siendo análisis nuestro, no del survey).

## 6. La evidencia de velocidad del nicho: 3 papers en 1 semana

Verificado textual, aparece TRES veces en el paper:

- Intro (§1): *"Notably, three independent papers on counterfactual/hindsight
  credit appeared within a single week in March 2026, suggesting growing
  community interest in this problem."*
- §5.2 (cierre de CCPO): *"The appearance of three independent
  hindsight/counterfactual papers (HCAPO, C3, CCPO) within a single week in
  March 2026 is a striking signal of community convergence."*
- Figura 5 (distribución temporal de papers): anotación explícita
  *"3 counterfactual papers in 1 week (Mar 2026)"*, sobre un histograma que
  muestra el shift de reasoning (2024) → agentic (2025–2026).

Los tres: HCAPO (LLM critic genera continuaciones contrafactuales en
"imaginación", sin re-ejecutar el env), C3 (leave-one-out contrafactual,
$c_t = R(\tau) - R(\tau_{\setminus t})$ aproximado por LLM), CCPO (trayectoria
como SCM, crédito = average treatment effect vía do-calculus). Esto calibra la
ventana de scoop: **cualquier idea "obvia en retrospectiva" en este nicho se
publica en meses, y a veces en triplicado.**

## 7. Tablas comparativas (§7)

⚠️ Precisión vs el pivot doc: no existe una "Tabla 7.1" — la tabla comparativa
central es la **Table 5 ("Unified Comparison Table")**, que vive en la
*sección* 7.1.

### 7.1 Table 5 — dimensiones que compara

| Columna | Valores |
|---|---|
| Granularity | Token / Segment / Step / Turn / Multi-Agent |
| Methodology | MC, TD, LLM-as-Critic, game-theoretic, info-theoretic, implicit, hindsight, … |
| Setting | R (reasoning) / A (agentic) / M (multi-agent) |
| Type | C (core CA) / E (adjacent enabler) |
| Aux. Model? | No / RM / PRM / Critic / LLM / Verifier / MLP / Library |
| Compute | Low / Med / High |
| Venue / Year | — |

Filas relevantes (textuales): GiGPO = Step · MC(group) · A · **No aux · Low
compute** · NeurIPS'25. iStar = Step · Implicit DPO · A · **No aux · Low**.
IGPO = Turn · Info-theoretic · A · **Verifier · Med**. SWEET-RL = Turn ·
Privileged Critic · A · Critic · Med. CriticSearch = Turn · Retrospective
Critic · A · LLM · Med. HCAPO = Turn · Hindsight · A · LLM · Med.

**Dónde caerían los métodos de PIAR:** el dual residual sería fila nueva
Step/Turn · *info-theoretic × privileged-context implicit* (cruce de columnas
de metodología que hoy no existe) · A · **Aux Model = No** (mismo modelo,
forward passes) · Compute = Low-Med. Es decir: el cuadrante barato de
GiGPO/iStar pero con la señal informada de SWEET-RL/CriticSearch — esa
combinación de celdas está vacía en Table 5.

### 7.2 Tables 6–7 — números (con el caveat del survey: no comparables entre papers)

Agentic (Table 7), los que nos importan:

| Método | Modelo | Benchmark | Score | Baseline | Δ |
|---|---|---|---|---|---|
| GiGPO | Qwen2.5-7B-Instruct | ALFWorld (succ.) | 90.2% | GRPO 77.6% | **+12.6** |
| GiGPO | Qwen2.5-7B-Instruct | WebShop (succ.) | 75.2% | GRPO 66.1% | **+9.1** |
| GiGPO | Qwen2.5-1.5B-Instruct | WebShop (succ.) | 67.4% | GRPO 56.8% | +10.6 |

Confirma N.10: GiGPO es el SOTA critic-free publicado en *nuestros dos
benchmarks target con nuestro modelo base exacto*. Esos son los números a
batir/igualar en los brazos B2.

### 7.3 Table 8 — guía práctica del survey (dónde nos posiciona)

- **Tool-use agents (WebShop, ALFWorld)**: recomienda GiGPO, AgentPRM,
  Turn-PPO — "critic-free preferred for efficiency".
- **Web navigation (WebArena)**: recomienda SWEET-RL, HCAPO, IGPO —
  *"privileged critic exploits training info"*.
- **Compute-constrained**: GRPO, CARL, iStar, GiGPO.

Lectura para PIAR: el survey ya reconoce que "privileged training info" es la
jugada correcta en POMDPs, pero solo la ofrece vía critics entrenados o jueces
LLM. Nadie la ofrece critic-free. PIAR es exactamente ese merge de filas.

### 7.4 Los 4 trade-offs (§7.4, con evidence levels)

1. Granularidad vs costo computacional [SE].
2. **Forward estimation vs hindsight analysis [AS]** — *"Hindsight has a strict
   informational advantage but introduces latency and may suffer from
   hindsight bias."* (El survey nombra el hindsight bias como riesgo abierto —
   nuestro backward + residual es una respuesta directa.)
3. Auxiliary model requirements [SE] — sin aux: CARL, iStar, GiGPO.
4. Reasoning-specific vs agent-general [LS].

## 8. El gap §4.2 — nuestra descomposición, declarada como problema abierto

El hallazgo más importante del due diligence no está en §9 sino en **§4.2
(Challenge 2: Partial Observability)**. Textual:

> "A correct credit assignment system must distinguish between:
> 1. **Decision errors**: the agent had sufficient information but chose poorly.
> 2. **Information gaps**: the agent lacked critical information and no
>    available action could have bridged the gap.
> 3. **Exploratory actions**: the agent correctly chose to gather information,
>    even if the immediate outcome was negative.
>
> Most current CA methods do not explicitly address this distinction, assigning
> credit based on outcomes rather than decision quality relative to available
> information. Addressing this gap is an important open problem (see Section 9)."

Esto ES la descomposición de PIAR con otro vocabulario: "decision quality
relative to available information" = forward pragmático condicionado al belief
(residual); "exploratory actions… even if the immediate outcome was negative" =
exactamente lo que el backward premia y el forward solo castiga (pivot §4.3,
"lo que NO arregla"). El survey lo marca como gap **que ningún método de los 47
ataca** y lo deriva a §9.

## 9. Fronteras abiertas (§9) — verificación

Estructura real de §9 (el pivot doc decía "§9 — verificar": ✅ correcto):

- **9.1 The Agentic Frontier**: Ultra-Long Horizon · Open-World sin rewards verificables · Multi-Agent at Scale.
- **9.2 Theoretical Frontiers**: **Credit Assignment Meets Exploration** · Formal Guarantees · Computation-Signal Trade-off.
- **9.3 Practical Frontiers**: Unified Benchmarks · CA and Memory · Reasoning→Agentic Transfer.
- (9.4 Threats to validity · 9.5 release de materiales.)

### 9.1 "Credit Assignment Meets Exploration" (§9.2) — textual completo

> "Better credit assignment should enable more targeted exploration, yet
> current methods treat CA and exploration as independent problems. The
> connection is natural: states where credit assignment is most uncertain are
> precisely the states where the agent should explore… **IGPO (Wang et al.,
> 2025a) provides a starting point by framing credit in information-theoretic
> terms, but no current method explicitly uses credit uncertainty to drive
> exploration.** We identify this as one of the most promising research
> directions, as it could simultaneously improve both sample efficiency and
> credit quality."

Ídem en §8.1 ("CA × Exploration"): *"no current method explicitly uses CA
uncertainty to drive exploration. This is a significant missed opportunity."*

⚠️ **Matiz vs el pivot doc**: el pivot (§3) dice que "CA meets exploration" es
"nuestra descomposición con otro nombre". Es *casi* eso pero no exacto: §9.2
habla específicamente de usar **incertidumbre del crédito para dirigir
exploración** (más cercano a curiosity/active learning). El match exacto de
nuestra descomposición dual es el **gap de §4.2** (decision errors / info gaps
/ exploratory actions, §8 arriba), que §9.2 viene a operacionalizar a su
manera. Para el related work del paper conviene citar §4.2 como el gap que el
dual cierra, y §9.2 como la dirección que el survey declara prometedora y que
nadie tomó. En ambas lecturas: **celda vacía confirmada, y además señalizada
como valiosa por el propio survey** — arma de doble filo (valida la dirección Y
le avisa al resto del campo; refuerza §10.5 del pivot).

### 9.2 Otras fronteras que nos tocan

- **Computation-Signal Trade-off** (§9.2): ¿más rollouts con crédito crudo o
  menos rollouts con crédito fino? "No paper provides a systematic answer."
  PIAR vive en el extremo bueno: señal fina a costo de forward passes (con
  prefix caching), cero rollouts extra — argumento de venta directo.
- **Unified Benchmarks** (§9.3): no hay benchmark estándar de CA; pide tareas
  con "controlled bifurcation points". Nuestro mapa dónde/por qué (info en la
  instrucción vs estado oculto) es una instancia de eso.
- **Key gaps del reporting checklist** (Apéndice C, textual): **"of the 41 core
  CA methods, 0/41 report total GPU-hours, 2/41 report variance or confidence
  intervals, and 0/41 include a compute-controlled baseline."** Bar bajísimo:
  pre-registro + presupuesto honesto (pivot §9) ya nos diferencia
  metodológicamente de todo el campo.

## 10. Qué NO cubre el survey (límites del due diligence)

Verificado por búsqueda en el texto completo y en el README del repo Awesome:

- **TAMTRL (2603.21663) NO aparece** — ni en el survey ni en el repo. El
  asesino del forward standalone quedó fuera del scope (compresión
  long-context, no agentic CA). ⇒ **El survey solo NO alcanza como due
  diligence del forward**; la amenaza TAMTRL se encontró por búsqueda directa
  (pivot §2). Corolario: el re-check mensual no puede ser solo "mirar el repo
  Awesome".
- **La familia hints (QuestA, KnowRL, BREAD, Scaf-GRPO…) NO aparece** — el
  survey clasifica CA en el canal del reward, no intervenciones sobre el prompt
  del student. Consistente con nuestra distinción estructural (invariante nuevo
  11): son literaturas distintas hasta para el surveyor.
- **OPSD / π-Distill / OPCD NO aparecen** — privileged context como loss de
  distillation tampoco es CA para el survey.
- Es un survey de **autor único**, snapshot abril 2026, mayoría preprints
  (§9.4 threats to validity). Mapa orientativo, no garantía de exhaustividad.

## 11. Lo que esto significa PARA PIAR (post-pivot)

El rol de este paper en el proyecto es **due diligence**: la evidencia citable
de que las tres contribuciones del pivot están libres a abril 2026.

### (a) Dual pragmático/epistémico — VERIFICADO LIBRE

- IGPO está **solo** en §5.5; iStar/StepAgent en §5.6; las categorías nunca se
  cruzan. Ningún método de los 47 combina una señal forward de acción bajo PI
  con una señal de information gain.
- Más fuerte: el survey **declara el gap explícitamente** (§4.2: ningún método
  distingue decision errors / information gaps / exploratory actions; §9.2 +
  §8.1: "no current method…", "significant missed opportunity"). No solo no
  existe — el surveyor lo pide.

### (b) PI residual gateada por belief — VERIFICADO LIBRE

- Solo dos métodos usan PI en el scorer: SWEET-RL (critic entrenado) y
  CriticSearch (juez generativo congelado). Ambos usan la PI **completa,
  siempre, sin descomponer ni dosificar**. "Belief-gated" / dosificación de PI
  no aparece en ninguna parte del survey.
- La familia que dosifica información (hints con annealing programado) ni
  siquiera entra al scope del survey porque opera del lado del student —
  refuerza que el residual lado-scorer es estructuralmente otra cosa.

### (c) Mapa dónde/por qué — VERIFICADO LIBRE y demandado

- §9.3 pide benchmarks unificados con bifurcation points controlados; §7.2
  constata que la fragmentación de benchmarks impide saber "which CA methods
  are genuinely better". Nuestro mapa por tipo de environment (info en la
  instrucción vs estado oculto) responde a esa demanda con dos benchmarks que
  el survey ya considera canon (WebShop, ALFWorld).

### Posicionamiento y números

- **GiGPO confirmado como baseline obligatorio (N.10)**: +12.6 ALFWorld /
  +9.1 WebShop sobre GRPO con Qwen2.5-7B-Instruct — nuestro modelo, nuestros
  benchmarks, celda `No aux · Low compute`.
- El claim de venta sale solo de Table 5 + Table 8: el survey recomienda
  privileged critics para POMDPs pero todos cuestan un critic entrenado o un
  juez LLM; los critic-free baratos (GiGPO/iStar) no usan PI. PIAR ofrece la
  señal privilegiada al costo critic-free.
- El reporting checklist (0/41 GPU-hours, 0/41 compute-controlled baselines)
  hace que nuestro pre-registro + presupuesto honesto sea diferenciador
  metodológico citable.

## 12. Lo más importante para retener

1. **Las tres celdas del pivot están vacías a abril 2026, con cita:** dual
   (IGPO solo en §5.5, iStar en §5.6, sin cruce), residual (SWEET-RL y
   CriticSearch usan PI completa sin gating), mapa (frontera abierta §9.3).
   Y el survey *pide* lo que el dual da (§4.2 + §9.2). Este paper es la
   referencia [survey] del related work del futuro paper de PIAR.
2. **El nicho se mueve a velocidad de semanas**: HCAPO + C3 + CCPO en una sola
   semana de marzo 2026 (verificado textual, 3 menciones + Figura 5). La
   ventana de las síntesis "obvias en retrospectiva" se mide en meses.
3. **Protocolo de re-check mensual (pivot §10.5) — operativo:**
   (i) repo `Awesome-Credit-Assignment-in-LLM-RL` (se actualiza, último update
   2026-05) — mirar las categorías Information-Theoretic, Implicit/DPO-Based y
   Hindsight/Counterfactual; (ii) citas nuevas a IGPO y a TAMTRL (alertas);
   (iii) búsqueda directa de términos tipo "belief-gated", "privileged
   residual", "pragmatic epistemic reward decomposition" — **el repo Awesome
   solo NO alcanza**: TAMTRL no está ni en el survey ni en el repo (§10).
   Trigger de alarma: cualquier paper que combine info-gain con un score de
   acción bajo PI, o que dosifique PI del scorer por estado del student.
4. **GiGPO es el número a batir**: 90.2% ALFWorld / 75.2% WebShop con
   Qwen2.5-7B-Instruct. Cualquier resultado de PIAR se lee contra esa fila de
   Table 7.
5. ⚠️ Precisiones vs el pivot doc: la tabla central es **Table 5 en §7.1** (no
   "Tabla 7.1"); el survey rinde IGPO como $\Delta \log P(\text{success}|h)$
   con verifier (la equivalencia exacta con nuestro backward se cita del paper
   de IGPO, no del survey); y "CA meets exploration" (§9.2) es *adyacente* a
   nuestra descomposición — el match exacto es el gap de §4.2.
