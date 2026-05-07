# SWEET-RL — Asymmetric Critic con Privileged Training-Time Info

> **Issue:** [#7](https://github.com/lucaspecina/piar-rl/issues/7) · **arxiv:** [2503.15478](https://arxiv.org/abs/2503.15478) · **Código:** [`facebookresearch/sweet_rl`](https://github.com/facebookresearch/sweet_rl) · **Datos:** [`facebook/collaborative_agent_bench`](https://huggingface.co/datasets/facebook/collaborative_agent_bench) · **Autores:** Zhou, Jiang, Tian, Weston, Levine, Sukhbaatar, Li (Meta). Submitted Mar 2025.
> **Rol en PIAR:** **la línea de la que nos diferenciamos.** Misma intuición de "privileged training-time info para credit assignment en agentic multi-turn", pero con critic LLM separado entrenado con Bradley-Terry. Es exactamente la línea que el invariante 4 de PIAR descarta.

## 1. Setup MDP — POMDP colaborativo

Modelan colaboración human-agent como POMDP finito:
$$\mathcal{M} = \{\mathcal{O}, \mathcal{C}, \mathcal{A}, \mathcal{T}, \mu_1, \mathcal{R}, N\}$$

- $\mathcal{O}$: observable history $o_t$ (lo que ve el actor).
- $\mathcal{C}$: **hidden training-time information** $c$ (lo que ve el critic, NO el actor).
- $\mathcal{A}$: tokens $a_t^{1:L}$ (acción multi-token).
- $\mathcal{R}$: scalar reward $r(o_t, a_t, c)$ **solo al final** del episodio (sparse).
- $N \leq 10$: máximo 10 turnos.

Objetivo: $\sum_t r(o_t, a_t, c)$.

## 2. Información privilegiada $c$ — qué es exactamente

| Tarea | $c$ |
|---|---|
| Backend Programming | **Código Python correcto** + 10 unit tests |
| Frontend Design | **HTML de referencia** + screenshot renderizado |

$c$ son **soluciones completas**, no hints. Generadas con Llama-3.1-70B + filtros de validez (code que pasa unit tests; HTML del dataset WebSight).

**Importante:** $c$ se usa en training time pero no en inference. Mismo invariante que PIAR.

## 3. Arquitectura — critic separado

**Critic = modelo LLM separado** del actor:
- **Backend Programming:** mismo backbone size que el actor (Llama-3.1-8B), parametrización log-ratio Eq. 3 (sin head de regresión).
- **Frontend Design:** VLM Qwen2-VL-7B-Instruct con head de regresión (necesario por la naturaleza multimodal del input). El paper aclara explícitamente que para Frontend usa regression head, no log-ratio puro.
- **Para el resto del análisis técnico siguiente uso la variante Backend (log-ratio Eq. 3) que es la que más se conecta con PIAR.** Frontend con regression head es el caso multimodal que valida que el patrón general (privileged info en el critic) funciona pero no informa la decisión de parametrización para PIAR.

**Función de ventaja parametrizada (Eq. 3):**
$$A_\theta(o_t, a_t, h) = \frac{1}{L} \sum_{l=1}^{L} \log \frac{\pi_\theta(a_t^l \mid o_t, a_t^{1:l-1}, c)}{\pi_{\text{ref}}(a_t^l \mid o_t, a_t^{1:l-1}, c)}$$

Donde:
- $\pi_\theta$: el critic LLM (entrenado con Bradley-Terry, ver §4).
- $\pi_{\text{ref}}$: snapshot frozen del modelo inicial.
- $1/L$: **normalización por longitud** (crítica, ver §6 ablation).
- **AMBOS modelos ven $c$.**

**Esta fórmula es un log-ratio.** Pero es un log-ratio fundamentalmente distinto al de PIAR:

| | $\pi$ del numerador | $\pi$ del denominador | Asimetría vive en |
|---|---|---|---|
| **PIAR** | mismos pesos del student + golden en contexto | mismos pesos del student sin golden | **el contexto** |
| **SWEET-RL** | critic entrenado con BT + $c$ en contexto | snapshot frozen + $c$ en contexto | **el entrenamiento** ($\pi_\theta$ aprendió, $\pi_{\text{ref}}$ no) |

Ambos usan log-ratio; ambos usan información privilegiada. Pero:
- En PIAR la asimetría es solo de contexto, no de pesos (invariante 4).
- En SWEET-RL la asimetría es el entrenamiento; el contexto privilegiado lo tienen **los dos** (es input a ambos $\pi_\theta$ y $\pi_{\text{ref}}$).

## 4. Loss del critic — Bradley-Terry sobre trayectorias

Pares $(\tau^+, \tau^-)$ ordenados por reward acumulado (positivo = más reward).

$$\mathcal{J}_A(\theta) = -\log \sigma\left(\sum_t \beta A_\theta(o_t^+, a_t^+, c) - \sum_t \beta A_\theta(o_t^-, a_t^-, c)\right)$$

Interpretación literal del paper: "increase the advantage for each action in the chosen trajectory and lower the advantage for each action in the rejected trajectory."

Dato importante: el critic se entrena offline con trayectorias pre-generadas (Llama-3.1-8B agent + Llama-3.1-70B simulator).

## 5. Pipeline two-stage

**Stage 1 (offline):** entrenar $A_\theta$ con BT sobre pares de trayectorias pre-generadas.

**Stage 2 (offline DPO policy improvement):** sobre estados/historias del dataset offline (no environment interaction durante stage 2):
1. Por cada step, samplear 16 candidatos de $\pi_\phi$ (actor).
2. **El critic emite scores escalares $A_\theta$** (log-ratio sumado y normalizado). El pipeline después discretiza esos scores: top 50% = chosen, bottom 50% = rejected — es decir, el critic emite valores numéricos pero el actor solo consume pares pre-construidos.
3. Optimizar $\pi_\phi$ con DPO multi-turn (Eq. 4):

$$\mathcal{J}_\pi(\phi) = -\log \sigma\left(\beta' \log \frac{\pi_\phi(a^+ \mid o_t)}{\pi_{\text{ref}}(a^+ \mid o_t)} - \beta' \log \frac{\pi_\phi(a^- \mid o_t)}{\pi_{\text{ref}}(a^- \mid o_t)}\right)$$

Detalles que importan a PIAR:
- **No GRPO/PPO**, sino DPO multi-turn aplicado por step.
- **No advantage normalization de batch** — el critic emite scores que se discretizan en pares; DPO consume los pares.
- **No es co-training online** (estilo iStar). Es **offline → offline**: stage 1 entrena el critic con trayectorias pre-generadas; stage 2 hace DPO del actor sobre candidatos sampleados desde el dataset offline. **No hay environment interaction durante stage 2.**

## 6. Ablations clave (Tabla 3)

| Variante | Backend Success | Δ vs full |
|---|---|---|
| **SWEET-RL full** | **40.4** | — |
| w/ Regression Head (en vez de log-ratio) | 36.2 | −4.2 |
| **w/o train-time info** ($c$ removida) | **31.2** | **−9.2** |
| w/o normalization $1/L$ | **3.6** | **−36.8** (collapse) |

**Tres conclusiones que afectan PIAR:**

1. **Privileged info aporta +9.2 puntos.** Ablation directa que valida la intuición compartida con PIAR: tener $c$ en training mejora credit assignment.

2. **Log-ratio supera regression head por +4.2** (en este setup). Refuerza que el log-ratio como parametrización de advantage es buena elección — converge con Yuan/iStar/OPSD.

3. **Length normalization es CRÍTICA.** Sin $1/L$ el método colapsa a 3.6. Para PIAR esto es alarma: si sumamos log-ratio sobre el span de la acción ReAct sin normalizar por longitud, podemos tener el mismo problema. iStar también suma sobre acción (no normaliza por longitud explícita), pero las acciones de iStar son más uniformes en longitud que las de SWEET-RL. **Loggear longitud de spans en PIAR y considerar normalización si hay alta varianza.**

**Figure 3a** (best-of-N scaling): SWEET-RL escala exponencial con N de candidatos; sin privileged info, plateau. Otra evidencia de que la info privilegiada es lo que driving la ganancia.

## 7. Setup experimental y resultados

**Modelo base:** Llama-3.1-8B-Instruct (actor + critic).

**Benchmark:** **ColBench** (introducido en este paper). Dos tareas:
- **Backend Programming:** 10K tareas train, 1K test. Python functions ≤50 líneas, 10 unit tests por function.
- **Frontend Design:** 10K tareas train, 500 test. HTML snippets ~100 líneas. Métrica: CLIP cosine similarity entre render final y referencia.

**Datos para entrenar el critic:** 15K trayectorias offline (Backend), 6K (Frontend).

**Tabla 2 — comparativos:**

| Método | Backend % Tests | Backend Success | Frontend Win |
|---|---|---|---|
| Zero-Shot | 34.2 | 22.4 | 33.8 |
| Rejection FT | 40.9 | 28.2 | 38.6 |
| Multi-Turn DPO | 48.0 | 34.4 | 42.8 |
| **SWEET-RL** | **56.8** | **40.4** | **48.2** |

**Ganancia:** +6.0% Backend Success, +5.4% Frontend Win Rate vs Multi-Turn DPO.

**Empate con GPT-4o:** Llama-3.1-8B + SWEET-RL = 40.4% = GPT-4o collaborative. GPT-4o single-turn = 16.2%.

## 8. ¿Discuten log-ratio puro o same-model with privileged context?

**Comparan vs value function (Section 5.4):** "while being standard practice in the deep RL literature, the use of a value function fails to achieve comparable scaling performance compared to SWEET-RL." (Esto refuerza que log-ratio > regression head, ya en §6 ablation.)

**No comparan vs same-model-different-context** (la línea de OPSD/PIAR). El paper menciona arquitecturas asimétricas solo en robotics (cita Pinto et al. 2017; Wilson & Hermans 2020 sobre sim-to-real con asymmetric actor-critic). **No conectan con OPSD ni con la idea de "un modelo, dos contextos."**

**Implicación para PIAR:** SWEET-RL valida la idea de "privileged training-time info para credit assignment" pero **no exploró** la simplificación de PIAR (mismo modelo, asimetría solo en contexto). Tres líneas convergen en log-ratio (Yuan, iStar, SWEET-RL) cada una con un régimen distinto, y la intersección "same-model + log-ratio + multi-turn agentic + privileged context" es genuinamente PIAR.

## 9. Limitaciones declaradas

1. **Necesita ~6K samples para superar Multi-Turn DPO** (Fig 3b). Con 3K, DPO gana. Curva de aprendizaje del critic es relevante.
2. **Asume $c$ disponible en training.** Para benchmarks donde $c$ no se puede automatizar (verifiable answer ausente), no aplica.
3. **Two-stage offline**: el critic se entrena con trayectorias pre-generadas, no co-entrenado online. Esto puede ser limitación si la distribución del actor cambia mucho.

## 10. Lo que esto significa PARA PIAR

### 10.1 Refuerzos de decisiones existentes

- **D.5 NUEVA: Length normalization a considerar para el span de acción.** SWEET-RL ablation: sin $1/L$ collapse a 3.6%. Si los spans de acción ReAct en PIAR tienen varianza alta de longitud, normalizar antes de sumar el log-ratio. **Loggear distribución de longitudes y decidir data-driven.** (Esta es la única decisión nueva inclinada que sale del paper.)

- **B.1 reforzada (log-ratio como parametrización):** SWEET-RL ablation log-ratio vs regression head: +4.2% a favor del log-ratio. Otra línea más que converge en la elección.

- **D.1 reforzada (medir efecto del privileged info):** SWEET-RL aísla la contribución de $c$ con ablation directa (40.4 → 31.2 sin $c$). PIAR debería tener una ablation análoga: golden vs no-golden en el contexto del teacher. Es además una sanity check obvia.

### 10.2 El delta de PIAR vs SWEET-RL — explícito

Ambos usan la misma intuición central: **privileged training-time info ayuda al credit assignment en multi-turn**. Pero por caminos distintos:

| | SWEET-RL | PIAR |
|---|---|---|
| Asimetría vive en | **El entrenamiento del critic** | **El contexto del teacher** |
| Critic | Modelo LLM separado entrenado con Bradley-Terry | El mismo modelo del student, frozen al checkpoint inicial |
| Critic ve golden? | Sí | Sí (en el prompt) |
| Critic se entrena? | Sí, con BT offline | No |
| Reward por step | Bradley-Terry $A_\theta$ + DPO multi-turn ranking | Log-ratio $r_{\text{PIAR}}^{[a]}$ + GRPO advantage |
| Combinación con outcome | Implícita en el ranking | Explícita: $A^E + \alpha A^S$ (heredado de iStar) |
| Optimizer del actor | DPO multi-turn | GRPO (heredado de iStar) |
| Pipeline | Two-stage offline → online | Online co-training |
| Líneas de código adicionales | Mucho más (entrenar y mantener un critic completo) | Pocas (50–250 estimado) |

**Tradeoff direccional:** SWEET-RL tiene más expresividad (un critic puede aprender features no triviales del privileged info), PIAR tiene más simplicidad (sin training de critic, sin pipeline two-stage). El test científico para PIAR es: ¿la simplicidad pierde mucho? Si el log-ratio context-induced captura suficiente señal, PIAR gana en simplicidad sin perder en credit assignment.

### 10.3 Riesgos identificados al cruzar SWEET-RL con PIAR

1. **Si las acciones ReAct tienen longitud muy variable, length normalization estilo SWEET-RL puede ser necesaria.** Inclinada a D.5.

2. **PIAR no tiene la ablation directa de "privileged info contribuye X%" todavía planificada.** SWEET-RL la tiene clara (40.4 vs 31.2). PIAR debe tener su análogo.

3. **El éxito de SWEET-RL en ColBench valida la utilidad del privileged-info para credit assignment en multi-turn.** Es buena evidencia direccional para PIAR. Pero Cohen-style: SWEET-RL gana solo +6% sobre Multi-Turn DPO con ~6K samples. Si PIAR (más simple) extrae una fracción de eso, ya es éxito.

### 10.4 El argumento de defensa de PIAR vs SWEET-RL

Cuando un reviewer pregunte "¿por qué PIAR si ya está SWEET-RL?", la respuesta fundamentada es:

1. **SWEET-RL requiere entrenar y mantener un critic separado.** Operativamente: dos modelos en GPU, pipeline two-stage offline → online, BT loss tuning, costo de cómputo extra para el critic.

2. **PIAR usa el mismo modelo.** Costo operativo: dos forward passes (con context vs sin) por trayectoria. Sin entrenamiento extra.

3. **PIAR generaliza la simplificación.** Si el log-ratio context-induced ya captura señal de progreso — **condicionado al resultado positivo del ablation D.1 (leakage vs progreso causal)** — no hace falta el critic entrenado. Si el ablation D.1 falla, esta línea de defensa cae y PIAR pivota o se convierte en variante minimalist con caveat.

4. **Los dos comparten el spirit, no el método.** PIAR es la versión minimalist de SWEET-RL — válida si la apuesta del invariante 4 funciona.

## 11. Conexiones con otros papers

- **Yuan 2024 (#3):** SWEET-RL hereda el log-ratio como parametrización del advantage, igual que Yuan/iStar/OPSD. Distingue: BT en lugar de outcome cross-entropy o KL.
- **iStar (#4):** ambos son agentic multi-turn con log-ratio. iStar entrena un PRM con DPO trayectorial sin privileged context; SWEET-RL entrena un critic con BT con privileged context. PIAR está más cerca de SWEET-RL en intuición pero más cerca de iStar en formulación operativa (action-level + GRPO).
- **OPSD (#5):** SWEET-RL no menciona OPSD ni same-model-with-context. Son líneas paralelas.
- **Pinto et al. 2017 / Wilson & Hermans 2020:** asymmetric actor-critic en robotics (sim-to-real). SWEET-RL los cita como antecedente conceptual, no técnico.

## 12. Decisiones / preguntas que esto dispara

- [ ] **D.5 (nueva en design-decisions):** loggear distribución de longitud de spans de acción ReAct en PIAR; aplicar length normalization si hay varianza alta. Inclinada por SWEET-RL ablation.
- [ ] **D.1 reforzada:** PIAR debe tener ablation análoga a "privileged info on/off". Es la sanity check más básica del método; SWEET-RL la hizo y gana 9.2%; PIAR debe medir cuánto gana.
- [ ] **Argumento de defensa de PIAR vs SWEET-RL** preparado para el paper writeup. Lo de §10.4.

## 13. Lo más importante para retener

1. **SWEET-RL valida la intuición de PIAR pero no su método.** Mismo problema, distintas soluciones. PIAR se diferencia explícitamente por el invariante 4.
2. **Log-ratio como parametrización gana otra vez** (vs regression head, +4.2 en SWEET-RL). Cuarto paper que converge en eso.
3. **Length normalization es crítica en SWEET-RL** (collapse sin ella). Riesgo análogo identificado para PIAR.
4. **Privileged info aporta +9.2 puntos en SWEET-RL** (ablation directa). Buena evidencia direccional para PIAR.
5. **Código y datos liberados** (`facebookresearch/sweet_rl` + `collaborative_agent_bench`). Si en el futuro queremos comparar PIAR vs SWEET-RL en ColBench, el infrastructure existe.
6. **PIAR es el minimalist de SWEET-RL.** Si la apuesta del invariante 4 funciona, PIAR es genuinamente más simple sin perder señal. Si no, hay que pivot.
