# PRIME — Process Reinforcement through Implicit Rewards

> **Issue:** [#6](https://github.com/lucaspecina/piar-rl/issues/6) · **arxiv:** [2502.01456](https://arxiv.org/abs/2502.01456) · **Código:** [`PRIME-RL/PRIME`](https://github.com/PRIME-RL/PRIME) (veRL-based) · **Autores:** Cui et al. (25 autores) — Tsinghua + colaboradores. Submitted Feb 2025, v2 Sep 2025.
> **Rol en PIAR:** **framework reference** — toma el implicit PRM de Yuan y lo envuelve en un loop completo de RL con online update del PRM, combinación con outcome rewards, y aplicación a múltiples algoritmos. Útil para diseño del pipeline de PIAR. Confirma muchos defaults de PIAR + reabre una pregunta sobre frozen vs online (C.2).

## 1. Setup MDP

Single-turn reasoning de math y code (no multi-turn agentic). Notación:
- $\pi_\theta$: policy.
- $\pi_{\text{ref}}$: reference frozen (típicamente SFT inicial).
- $\pi_\phi$: implicit PRM (se entrena online).
- $r_o$: outcome verifier (ground truth).

Trayectorias $\mathbf{y} = (y_0, \dots, y_T)$, autoregresivas.

## 2. Implicit step reward (igual que Yuan)

$$r_\phi(y_t) := \beta \log \frac{\pi_\phi(y_t \mid y_{<t})}{\pi_{\text{ref}}(y_t \mid y_{<t})}$$

con $\beta = 0.05$ — consistente con Yuan, iStar, SWEET-RL. Default recurrente en esta línea de literatura.

## 3. Combinación con outcome rewards (Eq. 5)

$$A_t^i = \underbrace{\sum_{s=t}^{|y^i|} \gamma^{s-t} \left( r_\phi(y_s^i) - \frac{1}{K-1} \sum_{j \neq i} r_\phi(y^j) \right)}_{\text{step component, LOO baseline}} + \underbrace{r_o(y^i) - \frac{1}{K-1} \sum_{j \neq i} r_o(y^j)}_{\text{outcome component, LOO baseline}}$$

**Observaciones críticas:**
- **Suma directa**, no interpolación. Sin coeficiente $\alpha$.
- **LOO baseline por componente** antes de combinar (para reducir varianza). Para process rewards, el baseline LOO usa "averaged implicit process rewards" del rollout — es decir, el $r_\phi(y^j)$ que aparece en la fórmula es el promedio de step rewards de la trayectoria $j$, no un token puntual. Aclarar al implementar.
- $\gamma = 1$ (sin discount) en las variantes analizadas en la comparación value/reward (§5.5 del paper). No es claro si es default universal en todos los runs reportados.
- **Sin curriculum**: ambos signals activos desde iteración 1.
- **Razón explícita** de calcular returns separados antes de combinar:
  > "directly mixing their values may lead to numerical instability"

**Para PIAR:** PRIME confirma B.4 (advantage-level combination con normalización). Diferencia con iStar: iStar normaliza advantages con stats de batch (mean/std), PRIME usa LOO. Operativamente similares; ambos resuelven el mismo problema de escala.

## 4. Online update del PRM (la novedad de PRIME)

**Loss del PRM (CE, Algorithm 1 línea 8):**
$$\mathcal{L}_{CE}(\phi) = -\mathbb{E}\left[r_o(\mathbf{y}) \log \sigma(r_\phi(\mathbf{y})) + (1 - r_o(\mathbf{y})) \log(1 - \sigma(r_\phi(\mathbf{y})))\right]$$

donde $r_\phi(\mathbf{y}) = \beta \log \pi_\phi(\mathbf{y}) / \pi_{\text{ref}}(\mathbf{y})$ es el reward implícito a nivel **trayectoria completa**.

**Target $r_o(\mathbf{y})$:** binario en math (0 = incorrecto, 1 = correcto vs ground truth); en code es soft label en $[0, 1]$ (fracción de unit tests que pasan). El loss CE acepta ambos.

**Update flow por iteración:**
1. Samplear $K$ rollouts del policy.
2. Filtrar por accuracy (mantener prompts con varianza no-cero, ver §6).
3. Update $\pi_\phi$ con CE loss usando outcome labels.
4. Forward $\pi_\phi$ y $\pi_{\text{ref}}$ sobre rollouts → extraer $r_\phi(y_t)$.
5. Update $\pi_\theta$ con PPO usando advantage $A_t^i$ de Eq. 5.

**Inicialización crítica:** $\pi_\phi \leftarrow \pi_{\text{SFT}}$ (mismo modelo que el policy inicial).
> "Initializing policy model and PRM from the same model largely alleviates the distribution shift issue"

**LR PRM = 1e-6, LR policy = 5e-7** (PRM se actualiza más rápido).

## 5. Frozen vs online PRM — el ablation que toca PIAR

**Figure 5 (PRIME):** PRM frozen empieza alto pero **degrada** classification accuracy de ~70% a ~55% en 240 steps. Online PRM **mejora** de ~55% a ~75% en mismos steps.

**¿Esto contradice nuestra C.2 ("teacher frozen") inclinada por OPSD?** Hay que pensar bien la diferencia de roles:

| | PRIME PRM | OPSD teacher | PIAR teacher (propuesto) |
|---|---|---|---|
| Rol | PRM aprendido sobre outcomes | Privileged-context distiller (no entrenado) | Privileged-context evaluator (no entrenado) |
| Se entrena? | **Sí, con CE sobre outcomes** | No | No |
| Frozen? | Update online | Frozen al checkpoint inicial | Frozen al checkpoint inicial |
| Razón del frozen/online | Online porque la distribución del policy cambia y el PRM debe seguirla | Frozen para regularizar (que el student no se aleje) | Frozen para regularizar + porque no hay nada que entrenar |

**Conclusión refinada:** PRIME y OPSD/PIAR no son comparables directamente porque sus "teachers" cumplen roles distintos. PRIME entrena el PRM con outcome labels (loss específico); OPSD/PIAR usan el modelo inicial como referencia "neutral" (sin loss propio). En PIAR el teacher no puede degradar por la misma razón que el de PRIME — porque PIAR no entrena al teacher para nada.

**Sin embargo, hay una pregunta abierta nueva** que vale anotar:

> Si el policy de PIAR aprende a generar outputs que el teacher frozen ya no rationaliza bien (porque el policy actual está lejos del SFT inicial), ¿el log-ratio se vuelve menos útil? PRIME sugiere que sí. **¿Hace falta re-snapshot periódico del teacher en PIAR?**

Esa pregunta no la había planteado antes. Es candidata para E.7 (nueva pregunta abierta).

## 6. Accuracy filtering — engineering insight

PRIME filtra prompts donde no hay todos correctos o todos incorrectos:
> "Online prompt filtering preserves prompts within a certain median-level difficulty range...and balances data distribution for the Implicit PRM online training."

**Efecto reportado (Fig 2):** ~35% mejor estabilidad del training.

**Conexión con OPSD:** mismo problema que OPSD identificó — cuando todos los rollouts del batch comparten outcome reward, el PRM update no aporta señal y el policy update tiene varianza cero. PRIME y OPSD lo resuelven con técnicas distintas (filtering vs token-level dense reward), pero ambos apuntan al mismo bug del régimen "outcome-only sparse".

**Para PIAR:** decisión nueva candidata D.6 (accuracy filtering en rollout selection). Menos crítica que D.5 (length norm) porque PIAR ya tiene reward denso (no depende de varianza de outcome), pero útil para estabilidad del PRM si en algún momento metemos updates online del teacher.

## 7. Otras decisiones de PRIME que tocan PIAR

### 7.1 Implicit PRM como reward, NO como value (Tabla 3 / Figura 11 v2)

PRIME comparó cuatro variantes:
| Variante | Avg Acc |
|---|---|
| REINFORCE outcome-only | 36.0 |
| PPO + value head | 35.8 |
| PPO + implicit PRM como value | 36.6 |
| **PPO + PRIME (implicit PRM como reward)** | **39.4** |

(Nota: REINFORCE + PRIME es 37.8, no la fila ganadora; el 39.4 corresponde a **PPO** + PRIME).

**Conclusión PRIME:** usar el implicit PRM como **reward** que se suma al advantage (Eq. 5) supera a usarlo como value head para baseline (39.4 vs 36.6).

**Implicación PIAR:** confirma que la formulación PIAR (log-ratio como step reward dentro del advantage GRPO) es la elección correcta, no como crítico de baseline. Refuerza B.4.

### 7.2 KL coefficient = 0

PRIME explícitamente:
> "We set KL coefficient to 0 in all experiments."

**Quinto paper que omite KL penalty** (Yuan, iStar, OPSD, SWEET-RL, PRIME). Refuerza B.6.

### 7.3 SFT ref vs policy_old ref

Comparable. PRIME elige SFT ref por leve mayor estabilidad. iStar elige policy_old (snapshot móvil) porque está entrenando el PRM. Para PIAR (sin entrenamiento del teacher) **SFT ref es la elección natural** — además es lo que C.2 ya dice ("frozen al checkpoint inicial").

### 7.4 CE vs DPO para entrenar el implicit PRM

PRIME prueba ambos: similares en performance; CE elegido por memoria.
> "DPO and CE achieve similar results, and we chose CE for memory efficiency."

**No aplicable directamente a PIAR** (no entrenamos al teacher), pero útil si en algún momento exploramos updates del teacher: CE es una opción más eficiente que DPO.

## 8. Setup experimental y resultados

**Modelo base:** Qwen2.5-Math-7B-Base (principal), Qwen2.5-Math-32B-Base (escala), RhoMath-1.1B (small).

**Datasets:**
- SFT math: 230K (MathInstruct + OpenMathInstruct-2 + NuminaMath + Reasoning-001).
- RL: 150K queries × 4 samples math; 66K queries × 32 samples DeepScaleR.

**Benchmarks:** AIME 2024, AMC, MATH-500, Minerva Math, OlympiadBench (math), LeetCode, LiveCodeBench (code).

**Hyperparams:**
- $\beta = 0.05$, $\gamma = 1$, $K = 4$ rollouts/prompt.
- LR policy $5 \times 10^{-7}$, LR PRM $10^{-6}$.
- Batch 256, micro-batch 8.
- PPO ε = 0.2.
- KL coef = 0.

**Eurus-2-7B-PRIME — Tabla 1 (los números 26.7/57.8/79.2/38.6/42.1 son la fila final del paper, correspondiente a step 592, no step 240; en step 240 el average es 41.0):**

| Benchmark | SFT base | RLOO+OV | **PRIME (step 592)** |
|---|---|---|---|
| AIME 2024 | 3.3 | 20.0 | **26.7** |
| AMC | 30.1 | 47.0 | **57.8** |
| MATH-500 | 66.2 | 73.2 | **79.2** |
| MinervaMath | 32.7 | 36.4 | **38.6** |
| OlympiadBench | 29.8 | 35.4 | **42.1** |
| **Average math (5 benchmarks)** | — | — | **48.9** |
| **Average con code (7 benchmarks)** | 28.8 | 36.9 | **43.9** |

(Los números 28.8 y 36.9 son sobre 7 benchmarks combinando math + code. Mantener clara la distinción math-only vs combined al comparar.)

**Δ vs RLOO outcome-only:** ~+7 puntos average sobre 7 benchmarks. Supera Qwen2.5-Math-7B-Instruct en 5/7 benchmarks.

**Eficiencia:** 2.5× más sample efficient que RLOO outcome-only (alcanza misma reward en ~40% steps).

## 9. Compatibility ablation (Tabla 4)

PRIME funciona aplicado a múltiples algoritmos:
| Base algo | Mejora con PRIME |
|---|---|
| REINFORCE | +1.8% |
| GRPO | +1.7% |
| RLOO | +4.1% |
| PPO | +3.6% |

**Implicación PIAR:** la primitiva (log-ratio implicit + advantage combination) es compatible con la mayoría de algoritmos RL. Confirma que la decisión de framework (Plan A prime-rl vs Plan B veRL) no afecta la formulación matemática.

## 10. Lo que esto significa PARA PIAR

### 10.1 Confirmaciones (refuerzos sin nuevas decisiones)

- **B.2** (β = 0.05): otro paper más que lo usa. Default recurrente en la literatura; no tunear primero salvo evidencia clara.
- **B.4** (advantage-level combination): PRIME confirma con LOO baseline. Variante alternativa a la stats-de-batch de iStar; misma idea direccional.
- **B.6** (sin KL penalty): otro paper más que lo omite. Default recurrente; reabrir solo si hay inestabilidad de training.
- **B.7** (action-level / aggregation a nivel acción): PRIME usa token-level porque es single-turn math. No directamente aplicable, pero su Eq. 8 ablation refuerza que log-ratio como reward (no value) es la elección.
- **C.2** (teacher frozen al checkpoint inicial): PRIME muestra que **PRM trained-frozen degrada** — pero como vimos en §5, el rol del PRM de PRIME y el del teacher de PIAR son distintos. La conclusión de PRIME no se transfiere mecánicamente. **C.2 sigue inclinada hacia frozen** pero ahora con un caveat extra (E.7 abajo).

### 10.2 Decisiones nuevas candidatas

- **D.6 (candidata):** **accuracy filtering en rollout selection** estilo PRIME. Filtrar prompts donde todos los rollouts comparten el mismo outcome (todos correctos o todos incorrectos). PRIME reporta 35% mejor estabilidad. **Menos crítico para PIAR** que para PRIME (porque PIAR ya tiene reward denso, no depende de varianza de outcome para tener gradiente), pero útil si exploramos updates del teacher o si la varianza de PIAR resulta baja en práctica. **Inclinada como observación, no como decisión cerrada.**

### 10.3 Pregunta abierta nueva

- **E.7 (candidata):** Si el policy aprende a generar outputs que el teacher frozen ya no rationaliza bien (porque la distribución del policy se aleja del SFT inicial), ¿el log-ratio de PIAR se vuelve menos útil? **¿Hace falta re-snapshot periódico del teacher cada N steps?** Importante: **esto NO es una inferencia de PRIME** — PRIME demuestra drift de un PRM con loss CE outcome (el reward "se desfasa" porque está optimizado para una distribución específica que el policy abandona). El teacher de PIAR mide un objeto distinto (delta context-induced de logprobs), que en principio es estable independiente de la política. Es un **riesgo empírico a testear**, no un problema demostrado. Si el ablation muestra que el delta context-induced sigue capturando progreso aunque el policy se aleje del SFT, E.7 se cierra como "no aplica".

### 10.4 Reference de pipeline

PRIME es el **mapa más completo** que tenemos de cómo se ensambla un loop RL alrededor de un implicit PRM. Algorithm 1 sirve como template directo para el pipeline de PIAR (cambiando "update PRM" por "use teacher with golden context"). Cuando arranque la fase de implementación, conviene seguir la estructura de PRIME y solo modificar los puntos específicos del invariante 4.

## 11. Conexiones con otros papers

- **Yuan 2024 (#3):** PRIME es la aplicación RL completa de Yuan. La formulación del implicit PRM es idéntica; PRIME agrega online update + combinación con outcome.
- **iStar (#4):** ambos son "loop completo con implicit step reward". iStar usa DPO trayectorial para el PRM; PRIME usa CE sobre outcomes. iStar es multi-turn agentic, PRIME es single-turn math. **Para PIAR, iStar es más cercano operativamente** (multi-turn) pero **PRIME es más cercano filosóficamente** (PRM init = SFT model, no separado).
- **OPSD (#5):** PRIME y OPSD son complementarios. OPSD usa privileged-context teacher; PRIME no usa teacher con golden context — usa solo policy + outcome verifier. PIAR combina las dos primitivas.
- **SWEET-RL (#7):** PRIME es lo opuesto operativo de SWEET-RL. SWEET-RL = critic LLM separado entrenado offline con BT + privileged info. PRIME = PRM init from SFT, online update con CE, sin privileged info. **PIAR es híbrido**: same-init como PRIME, privileged-context como SWEET-RL, frozen como OPSD, action-level + advantage-combine como iStar.

## 12. Limitaciones declaradas

1. Modelos hasta 32B (no probaron más grande).
2. Saturación en RL "Zero" (base model RL) ~50 steps; suspechan diversity loss.
3. No exploran multi-stage training (DeepScaleR-style context lengthening) en profundidad.
4. Single-turn math/code; no multi-turn agentic.

## 13. Lo más importante para retener

1. **PRIME es el template del pipeline para PIAR.** Algorithm 1 + Eq. 5 + filtering + LOO baselines son ingredientes reusables. Modificar solo el punto de "update PRM" (en PIAR no se updatea, se usa golden context).
2. **β = 0.05 y sin KL penalty son defaults recurrentes en esta línea** (Yuan, iStar, OPSD, SWEET-RL, PRIME). No tunear desde cero; arrancar con esos valores y reabrir solo con evidencia.
3. **D.6 candidata:** accuracy filtering como engineering insight — no decisión cerrada, pero incluir en el pipeline si la varianza inicial de PIAR resulta problemática.
4. **E.7 nueva pregunta:** ¿re-snapshot periódico del teacher en PIAR? PRIME sugiere que un PRM trained degrada si frozen; el teacher no-trained de PIAR puede ser distinto pero merece verificación empírica.
5. **PRIME no usa privileged context.** PIAR sigue siendo único en combinar same-model + golden context + log-ratio + multi-turn agentic.
