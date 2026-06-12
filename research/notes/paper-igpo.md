# IGPO — Information Gain-based Policy Optimization

> **Issue:** epic Pivot 2026-06 (ver Project v2) · **arxiv:** [2510.14967](https://arxiv.org/abs/2510.14967) (usar **v2**, 2026-03-24) · **Código:** [`GuoqingWang1/IGPO`](https://github.com/GuoqingWang1/IGPO) (veRL-based, sobre Search-r1 + DeepResearcher) · **Autores:** Wang, Dai, Ye, Gan, Yao, Deng, Wu, Ying (afiliaciones no listadas en la página de arXiv ⚠️ no verificado). **ICLR 2026.**
> **Rol en PIAR:** **el vecino que tomó el backward.** IGPO define el reward por turno como el incremento marginal de $\log \pi(\text{golden} \mid \text{historial})$ — exactamente la dirección epistémica de PIAR (§4.2 del pivot doc), validada solo en search/QA. Pasa de "amenaza" a related work que valida el componente: el brazo A2 de PIAR es ≈ IGPO trasplantado a ejecución. Además resuelve empíricamente la decisión N.7 (normalización separada > conjunta) y su repo expone ambas variantes como flag.

## 1. Setup y problema

Agentes de búsqueda multi-turn (ReAct-style: think → tool call → tool response, con golden answer $a$ disponible en training). El problema que atacan: con outcome-only reward en multi-turn aparecen (i) **advantage collapse** — todos los rollouts del grupo reciben el mismo reward y el advantage GRPO se anula — y (ii) **falta de credit assignment fino** entre turnos en horizontes largos. Mismo diagnóstico que motiva a iStar y a PIAR, pero la solución va por la dirección opuesta al forward: en vez de preguntar "¿actuaste como quien sabe?", pregunta "¿ahora sabés más?".

Posicionamiento explícito del paper: a diferencia de PRMs externos o estimación Monte Carlo costosa (Math-Shepherd-style), IGPO deriva rewards **intrínsecos** de los belief updates del propio modelo. Cero modelos extra, cero rollouts extra.

## 2. El reward de information gain (definición exacta)

### 2.1 Belief sobre la golden (Eq. 3)

Con $a = (a_1, \dots, a_L)$ los tokens de la golden answer y $o_{i,\le t}$ el prefijo del rollout $i$ hasta el turno $t$:

$$\log \pi_\theta(a \mid q, o_{i,\le t}) = \frac{1}{L} \sum_{j=1}^{L} \log \pi_\theta(a_j \mid q, o_{i,\le t}, a_{<j})$$

Detalles que importan:
- **Length-norm explícita**: promedio sobre los $L$ tokens de la golden ($1/L$). Idéntico al $b_i(t) = \frac{1}{|g_i|} \log \pi_{\text{old}}(g_i \mid h_t, w)$ de PIAR.
- **Wrapper**: la golden se envuelve en el mismo schema XML que una predicción del agente para mantener consistencia de formato con el rollout: `<think>Now there's enough information...</think><answer>Ground Truth a</answer>`. Es decir, el "wrapper textual fijo $w$" de PIAR existe en IGPO y es el template de respuesta del propio agente.
- **Condicionamiento**: sobre $q$ + historial completo del rollout hasta el turno $t$ — incluye los thoughts del agente, no solo acciones+observaciones. IGPO no problematiza esto (relevante para N.6, ver §10.4).
- Se computa con la política del rollout y se aplica **stop-gradient** al reward (el IG es señal, no objetivo diferenciable).

### 2.2 Reward por turno (Eq. 4)

$$r^{IG}_{i,t} = \log \pi_\theta(a \mid q, o_{i,\le t}) - \log \pi_\theta(a \mid q, o_{i,\le t-1}), \qquad 1 \le t < T$$

- Diferencia de log-probs entre turnos consecutivos = exactamente el $r_{\text{bwd}}(t) = \sum_i [b_i(t) - b_i(t-1)]$ de PIAR con $K=1$ (golden monolítica, sin descomposición en componentes).
- **El turno final $T$ NO recibe IG reward**: recibe solo el outcome reward. IG vive en $1 \le t < T$.
- El reward del turno se asigna a **todos los tokens generados en ese turno** (reasoning + tool calls); los tokens de tool response se enmascaran en el backward pass.

### 2.3 Outcome reward (Eq. 2)

$$r^O = \begin{cases} F1(\hat{a}, a) \in [0,1] & \text{formato válido} \\ \lambda_{fmt} < 0 & \text{formato inválido} \end{cases}$$

F1 word-level entre predicción y golden, con penalidad de formato.

## 3. Normalización y combinación (la decisión N.7)

**Eq. 5 — z-normalización SEPARADA por grupo de $G$ rollouts:**

$$\tilde{r}_{i,t} = \begin{cases} (r^{IG}_{i,t} - \mu_{IG})/\sigma_{IG} & 1 \le t < T \\ (r^O_i - \mu_O)/\sigma_O & t = T \end{cases}$$

donde $\mu_{IG}, \sigma_{IG}$ son mean/std de **todos los IG rewards del grupo** y $\mu_O, \sigma_O$ de los outcome rewards del grupo. Razón declarada: evitar "disparidades de escala significativas" entre IG y outcome (Apéndice D.1).

**Eq. 6 — return descontado por turno:**

$$\tilde{R}_{i,t} = \sum_{k=t}^{T} \gamma^{k-t} \, \tilde{r}_{i,k}, \qquad \gamma = 1.0 \text{ en todos los experimentos}$$

Con $\gamma = 1$, el "advantage" de cada turno es la suma de todos los rewards normalizados futuros (IG futuros + outcome) — el outcome llega a todos los turnos, el IG de turnos tardíos no llega a turnos tempranos en reversa pero sí al revés.

**Eq. 7 — objetivo:** GRPO-style con ratio clipping ($\epsilon$ estándar) sobre $\tilde{R}_{i,t}$ a nivel token (turn-level reward broadcast a los tokens del turno), más KL penalty $\beta\, D_{KL}(\pi_\theta \| \pi_{ref})$ con $\beta = 0.001$. Nota: a diferencia de la línea Yuan/iStar/PRIME (KL = 0), IGPO **sí usa KL penalty**, aunque chico.

**El repo expone ambas variantes como flags** (verificado en README):
- `+algorithm.info_gain_type`: `"log_prob_diff"` vs `"prob_diff"`.
- `+algorithm.info_gain_norm_mode`: `"separate"` vs `"joint"`.
- Bonus no mencionado en el paper note del pivot: `+algorithm.use_curriculum` — decay programado del peso del IG reward (`curriculum_ig_init/final`, `curriculum_f1_init/final`). O sea, IGPO contempla annealing del IG, pero **por schedule** (ver §10.3).
- `+algorithm.use_vectorized_gt_logprob`: cómputo vectorizado de los log-probs de la golden.

## 4. Justificación teórica

Apéndice A: conexión con **"snowball errors"** — Lemma A.2 acota el error final por abajo con $\Omega(\text{entropía acumulada}/(T-1))$; Theorem A.4 muestra que maximizar el process reward acota por arriba la entropía acumulada de la golden condicionada al historial. La intuición: cada turno debe reducir la incertidumbre del modelo sobre la respuesta correcta.

**No invocan potential-based reward shaping ni telescoping formal.** El telescoping ($\sum_t r^{IG}_t = \log\pi(a|o_{\le T-1}) - \log\pi(a|o_0)$) está implícito en la construcción pero no lo nombran ni derivan la garantía de invarianza de política óptima (Ng et al. 1999). PIAR puede reclamar esa formalización como propia (§4.2 del pivot doc) — con el caveat honesto de que la z-normalización por grupo (Eq. 5) **rompe técnicamente la forma potential-based** ($F = \gamma\Phi(s') - \Phi(s)$ exacta), igual que nos pasaría a nosotros con $\hat{z}(r_{\text{bwd}})$. Ver §10.2.

## 5. Experimentos

**Modelo base:** Qwen2.5-7B-Instruct (principal), Qwen2.5-3B-Instruct (ablations). Mismo modelo base que PIAR.

**Benchmarks:** in-domain NQ, TriviaQA, HotpotQA, 2Wiki; out-of-domain MuSiQue, Bamboogle, PopQA. Métrica: F1 word-level. Tool: Google search API. Todo search/QA — **cero environments de ejecución/decisión**.

**Baselines:** prompt-based (CoT, CoT+RAG, Search-o1), outcome-reward RL (Search-R1 base/instruct, R1-Searcher, DeepResearcher), step-reward RL (StepSearch, ReasoningRAG, **GiGPO**), RL estándar (PPO, RLOO, GRPO, Reinforce++, GSPO).

**Números principales (Tabla 1, 7B, avg sobre los 7 benchmarks):**

| Método | NQ | TQ | HotpotQA | 2Wiki | MuSiQue | Bamboogle | PopQA | **Avg** |
|---|---|---|---|---|---|---|---|---|
| **IGPO** | 46.4 | **80.6** | **59.0** | **72.1** | **32.7** | **77.0** | **53.8** | **60.2** |
| DeepResearcher | 39.6 | 78.4 | 52.8 | 59.7 | 27.1 | 71.0 | 48.5 | 53.9 |
| GiGPO | 46.4 | 64.7 | 41.6 | 43.6 | 18.9 | 68.9 | 46.1 | 47.2 |

+6.3 pts avg sobre el mejor baseline (DeepResearcher). **+13 pts sobre GiGPO** en este dominio — dato relevante para calibrar expectativas del baseline B2: GiGPO sufre donde los estados no se repiten entre rollouts (search abierto), que es precisamente su limitación estructural.

**Hyperparams clave:** batch 32 prompts, $G = 16$ rollouts/prompt, lr actor 1e-6, mini-batch 512, max 10 turnos, $\gamma = 1.0$, $\beta_{KL} = 0.001$, temperature 1.0, max prompt 30,767 tokens / response 2,000. Framework veRL, 8×A100-80G, sequence parallel 4.

**Overhead del IG:** <0.4% del cómputo del reward, <0.02% end-to-end. Confirma que el costo de los forward passes extra del belief es despreciable (lo mismo aplica al $b_i(t)$ de PIAR, que además amortiza con prefix caching).

## 6. Ablations

**Tabla 3 — componentes del reward (7B, avg):**

| Variante | Avg |
|---|---|
| Solo outcome (F1 only) | 51.9 |
| **Solo IG (sin outcome)** | **53.5** |
| F1 + IG (IGPO completo) | **60.2** |

El dato más fuerte para PIAR: **IG solo supera a outcome solo** en su dominio, y la combinación supera ampliamente a ambos. La señal epistémica no es un regularizador cosmético: lleva información de crédito real. (Ojo: "IG only" mantiene igual el F1 implícito en el telescoping hacia la respuesta — pero el paper lo reporta como brazo separado.)

**Tabla 5 — base del IG y normalización:** LogProb + Separate gana: 60.2 (7B) / 48.9 (3B), vs 59.4 con Prob + Separate en 7B. ⚠️ no verificado: los números exactos de las celdas Prob+Joint / LogProb+Joint no pude extraerlos del HTML; el paper afirma que separate evita disparidades de escala y logprob da estabilidad numérica.

**Escala:** la mejora es mayor en el modelo chico — +16.6 pts sobre GRPO estándar en 3B.

**Análisis adicionales:** (i) convergencia más rápida que GRPO a igual presupuesto de tokens; (ii) Figura 5: IGPO reduce más la entropía de la golden condicionada al contexto turno a turno (validación directa de que el mecanismo opera); (iii) trazas de razonamiento de mayor calidad según juez Gemini 2.5-Pro.

## 7. v1 → v2: el renombre a "Search Agents"

Verificado contra arXiv:

- **v1 (2025-10-16):** "...A Simple and Effective Approach for Multi-Turn **LLM Agents**".
- **v2 (2026-03-24, la de ICLR):** "...A Simple and Effective Approach for Multi-Turn **Search Agents**".

El claim se **achicó** de agentes LLM en general a agentes de búsqueda. Consistente con eso, toda la evaluación es search/QA, y las limitaciones declaradas dicen: el método "still relies on the availability of ground-truth answers, which limits its applicability in open-ended settings", con "broader agentic scenarios beyond search" relegado a future work — sin experimentos ni discusión en embodied/web/code/decisión.

**Lectura para PIAR:** la validación del IG reward en **ejecución/decisión** (WebShop, ALFWorld — donde la PI no es "la respuesta" sino hechos del environment, y el episodio termina en una acción, no en una respuesta textual) quedó explícitamente fuera del claim de IGPO. El brazo A2 (IGPO trasplantado) no es una réplica: es la extensión que ellos mismos dejaron como future work, y la predicción de PIAR es que **ahí el epistémico solo NO alcanza** (ejecutar no informa — §4.5 del pivot doc), que es justamente lo que motiva la descomposición dual.

## 8. Failure modes y limitaciones declaradas

1. **Golden mismatch / ambigüedad (Apéndice E):** en preguntas ambiguas el agente encuentra respuestas correctas pero no-gold → el IG las penaliza (caso reportado: IG = −0.81 sobre razonamiento válido). Cuantifican: el IG "amplifies the impact of noise within the ground truth" en ~3.6% de las muestras.
2. **Requiere golden answer** → no aplica a open-ended.
3. **No analizan reward hacking del IG** (loitering: acumular contenido parecido a la golden sin responder). Tampoco hindsight bias — no les aplica porque no tienen componente forward.
4. KL penalty activa ($\beta=0.001$) sin ablation reportada de KL=0 ⚠️ no verificado si la probaron.

## 9. Conexiones con otros papers

- **GiGPO (#paper-gigpo):** baseline directo en IGPO y B2 en PIAR. IGPO lo supera por +13 en search QA — pero GiGPO fue diseñado para ALFWorld/Sokoban (estados repetidos); el orden puede invertirse en ejecución. Los dos números (IGPO>GiGPO en search, GiGPO SOTA en ALFWorld) son exactamente el "mapa dónde/por qué" que PIAR quiere trazar.
- **iStar:** misma motivación (sparse reward multi-turn), mecanismo ortogonal (juez DPO en pesos vs belief intrínseco). El survey de CA los pone en categorías que no se citan entre sí — el cruce es el hueco de PIAR.
- **Math-Shepherd:** el IG de IGPO es la amortización del value Monte Carlo (qué probabilidad de llegar a la respuesta correcta desde acá) sin rollouts extra — misma observación que hicimos para $r_{\text{bwd}}$.
- **TAMTRL (#paper-tamtrl):** la otra mitad — TAMTRL toma el forward (prob del teacher con contexto privilegiado), IGPO toma el backward. Ninguno combina.

## 10. Lo que esto significa PARA PIAR (post-pivot)

### 10.1 El backward de PIAR ≈ IGPO generalizado por componentes

El $r_{\text{bwd}}(t) = \sum_i [b_i(t) - b_i(t-1)]$ del pivot doc §4.2 **es** la Eq. 4 de IGPO con dos generalizaciones: (a) $K$ componentes fácticos $g_i$ en vez de golden monolítica (N.1) — necesario en environments donde la PI no es "una respuesta" (ALFWorld: hechos del estado oculto; WebShop: atributos del producto); (b) el wrapper $w$ por componente en vez del template `<answer>` único. Todo lo demás coincide: length-norm $1/L$ (verificada en Eq. 3), diferencia entre turnos consecutivos, stop-gradient, belief medido con la política del rollout (su $\pi_\theta$ del rollout ≈ nuestro $\pi_{\text{old}}$). **La descomposición por componentes además mitiga su failure mode #1**: con golden monolítica una respuesta correcta-pero-parafraseada tanquea todo el belief; con componentes + canonicalización (N.8) el daño se acota por componente.

### 10.2 N.7 queda resuelta empíricamente: normalización separada

Tabla 5 + Apéndice D.1 + flags del repo (`info_gain_norm_mode = separate/joint`): separada gana, logprob > prob. Confirma el default del pivot doc ("arrancar separada") con evidencia publicada — N.7 puede pasar de abierta a **inclinada (separada)**, dejando conjunta como ablation barata vía el flag. Caveat teórico para el paper: la z-norm por grupo rompe la forma potential-based exacta del shaping; si reclamamos la garantía PBRS (que IGPO no reclama — no mencionan a Ng et al.), hay que o (a) presentarla sobre el reward sin normalizar y tratar la norm como aproximación de varianza, o (b) verificar empíricamente que no induce las policies degeneradas que PBRS previene.

### 10.3 El curriculum flag valida (por contraste) la PI residual

El repo expone `use_curriculum`: decay **programado** del peso del IG. Es la solución-por-schedule al mismo problema que el residual de PIAR resuelve **emergentemente** (auto-annealing por belief, §4.3 del pivot doc, N.4). Que IGPO haya necesitado implementar el knob refuerza que la no-estacionariedad de la señal es un problema real reconocido — y nos da el contraste exacto para related work: schedule manual (IGPO curriculum, familia QuestA/KnowRL) vs gating por estado epistémico medido (PIAR). ⚠️ no verificado si el paper reporta resultados con curriculum ON o si es solo del repo.

### 10.4 Toca la decisión N.6 (¿thoughts en el historial del belief?)

IGPO condiciona el belief sobre el rollout completo **incluyendo los thoughts** y no reporta problemas de autosugestión — pero tampoco lo ablaciona, y en search el riesgo es menor (la evidencia entra por tool responses verificables). Default de N.6 (historial completo) queda consistente con IGPO; la ablation solo-acciones+observaciones sigue siendo nuestra.

### 10.5 Detalles de implementación importables

(i) Asignación turn-level → broadcast a tokens del turno con masking de tool responses (igual que iStar/veRL); (ii) el turno final solo recibe outcome — decisión a copiar o ablacionar en PIAR (¿$r_{\text{bwd}}(T)$ aporta algo si la acción final es ejecutar?); (iii) `use_vectorized_gt_logprob` como referencia de cómo abaratar el cómputo del belief; (iv) $\gamma = 1$, $G = 16$, lr 1e-6 como punto de partida; (v) overhead <0.02% end-to-end desactiva cualquier objeción de costo contra el belief tracking; (vi) ojo con su $\beta_{KL} = 0.001 \ne 0$, que rompe el patrón Yuan/iStar/PRIME — anotar para la replicación.

### 10.6 Qué NO valida IGPO (el espacio de PIAR)

- **Ejecución/decisión**: claim renombrado a search, future work declarado. Brazo A2 = el experimento que falta en su paper.
- **La descomposición dual**: IGPO no tiene forward; su "IG only" (53.5) vs "F1+IG" (60.2) muestra que combinar señales ayuda, pero la otra señal es el outcome, no el pragmático.
- **PI residual**: su annealing es schedule opcional, no gating por belief.
- **Componentes fácticos $g_i$**: golden monolítica de QA.
- **Controles de leakage** estilo D.1/D.9 (shuffled-golden): no los corre.

## 11. Lo más importante para retener

1. **El backward de PIAR está publicado y validado — en search/QA.** Eq. 4 de IGPO = $r_{\text{bwd}}$ con $K=1$. Length-norm $1/L$, diferencia entre turnos, stop-gradient, wrapper XML: todos los detalles coinciden con §4.2 del pivot doc. IGPO es citación obligada y el brazo A2 es su extensión a ejecución, no su réplica.
2. **N.7 → inclinada a separada**: z-norm separada de IG y outcome dentro del grupo gana en su Tabla 5; el repo expone ambas (`info_gain_norm_mode`). Logprob > prob cruda.
3. **"IG only" supera a "outcome only"** (53.5 vs 51.9 avg) y la combinación da +8.3 más (60.2): la señal epistémica sola ya entrena. Predicción de PIAR a testear: en ejecución (WebShop buy, ALFWorld) ese orden se rompe porque ejecutar no informa.
4. **IGPO no reclama PBRS ni telescoping formal** (su teoría es vía snowball errors / entropía acumulada): la formalización potential-based del pivot doc sigue disponible como contribución — cuidando que la z-norm rompe la forma exacta.
5. **El renombre v1→v2 ("Multi-Turn LLM Agents" → "Multi-Turn Search Agents") está verificado** y delimita exactamente el espacio libre: agentes de decisión/ejecución, descomposición dual, residual, y el mapa dónde/por qué.
