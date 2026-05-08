# π-Distill — Privileged Information Distillation for Language Models

> **Issue:** [#10](https://github.com/lucaspecina/piar-rl/issues/10) · **arxiv:** [2602.04942](https://arxiv.org/abs/2602.04942) · **Autores:** Penaloza, Vattikonda, Gontier, Lacoste, Charlin, Caccia (ServiceNow + Mila). Submitted Feb 2026.
> **Rol en PIAR:** **el vecino conceptual más cercano matemáticamente y operativamente.** Mismo setup (PI teacher + agentic multi-turn) pero con KL distillation en lugar de log-ratio reward. Aporta tres cosas a PIAR: (1) taxonomía de tipos de PI, (2) evidencia empírica directa de leakage token-level (refuerza D.1), (3) decisión clave de diseño confirmada: si entrenamos al teacher (no es nuestro caso) hay que tener cuidado con shared-params + dual gradients.

## 1. Setup y notación

Multi-turn agentic. Misma red parametrizada por θ que actúa como **teacher** o **student** según el prompt:
- $\pi_\theta^T(o \mid s, I)$: teacher condicionado en PI (tools, hints, etc.).
- $\pi_\theta^S(o \mid s)$: student sin PI.

Outputs $o = (z, a)$: reasoning tokens + action tokens. Estado $s$ = contexto agregado (prompt + outputs previos + respuestas del environment). Reward $R \in [-1, 1]$ ambiental.

**Parámetros compartidos** entre teacher y student. La asimetría vive en el prompt — igual que PIAR. Pero el régimen de actualización es distinto (ver §4).

## 2. Tipos de PI explorados (taxonomía relevante para PIAR)

El paper extrae trayectorias de DeepSeek-chat-v3.1 (modelo frontera con reasoning tokens accesibles) y deriva tres niveles de PI (Figura 3):

| Tipo de PI | Densidad | Ejemplo |
|---|---|---|
| **Tool Calls & Arguments** | máxima | `GetUserDetails(Name:"Kevin Lau")` |
| **Tool Calls Only** | media | `GetUserDetails(...)` (modelo infiere args) |
| **Self-Generated Hints** | baja | resumen libre de la trayectoria exitosa |

PI se inserta como prefijo en el system prompt del teacher.

**Impacto directo en PIAR — C.5 informada:** π-Distill provee una taxonomía concreta de privileged context con resultados comparables. Para PIAR el equivalente directo sería:
- **PI denso ↔ golden answer con CoT completo** (estilo OPSD).
- **PI medio ↔ scratchpad estructurado / SCM**.
- **PI bajo ↔ hint resumido** (más abstracto que el answer).

## 3. π-Distill — el método central

### 3.1 Loss del teacher (Eq. 2)

$$\mathcal{J}_{\text{Teacher}}(\theta) = \mathbb{E}\left[R(o, s)\right] - \beta D_{KL}\left( \pi_\theta^T(o \mid s, I) \,\|\, \text{sg}(\pi_\theta^S(o \mid s)) \right)$$

Esperanza sobre trayectorias muestreadas del **teacher**. Stop-gradient sobre el student dentro del KL — solo el teacher recibe gradiente de este término.

**Intuición:** el teacher maximiza reward pero se regulariza para no alejarse del student.

### 3.2 Loss del student (Eq. 3)

$$\mathcal{J}_{\text{Student}}(\theta) = \mathbb{E}\left[ \frac{\pi_\theta^S(o \mid s)}{\text{sg}(\pi_\theta^T(o \mid s, I))} R(o, s) \right] - \beta D_{KL}\left( \text{sg}(\pi_\theta^T) \,\|\, \pi_\theta^S \right)$$

**Off-policy** respecto al student: aprende de trayectorias del teacher via importance weighting. **Stop-gradient sobre el teacher.**

### 3.3 Loss combinado (Eq. 4)

$$\mathcal{J}_{\pi\text{-Distill}}(\theta) = \alpha \cdot \mathcal{J}_{\text{Teacher}}(\theta) + (1-\alpha) \cdot \mathcal{J}_{\text{Student}}(\theta)$$

con $\alpha \in [0, 1]$ regulando si el énfasis está en mejorar el teacher (α=1) o entrenar al student directamente (α=0).

**Detalles importantes de las orientaciones de KL:**
- Eq. 2 (teacher-side): $D_{KL}(\pi^T \| \pi^S)$ — reverse KL desde la perspectiva de "quién aprende" (el teacher matchea al student frozen).
- Eq. 3 (student-side): $D_{KL}(\pi^T \| \pi^S)$ con stop-grad sobre teacher — el student aprende a aproximar la distribución del teacher.
- Eq. 5 (OPSD): $D_{KL}(\pi^S \| \pi^T)$ con stop-grad sobre teacher — esto es "reverse KL clásico" en el sentido de distillation literature.

**KL se estima Monte Carlo token-level**. El paper también usa importance weighting en Eq. 3 pero **a nivel trayectoria**, no per-token: ratio $\pi^S(o \mid s) / \text{sg}(\pi^T(o \mid s, I))$ aplicado al reward del trace completo.

### 3.4 Conexión variacional

> "can be viewed as form of Variational Expectation-Maximization."
- E-step: mejorar $\pi^T$ (aproximación posterior con PI).
- M-step: destilar en $\pi^S$.

## 4. La diferencia operativa con PIAR — esto es lo que importa

| | π-Distill | PIAR |
|---|---|---|
| Mismo modelo? | Sí | Sí |
| Asimetría vive en | Contexto | Contexto |
| **Teacher se entrena?** | **Sí, con $\mathcal{J}_{\text{Teacher}}$** | **No, frozen al checkpoint inicial** |
| **Student se entrena?** | **Sí, con $\mathcal{J}_{\text{Student}}$ off-policy** | **Sí, con GRPO usando log-ratio como step reward** |
| Loss | KL distillation + RL | Log-ratio como reward |
| Sampling de trayectorias | Teacher samplea → student aprende off-policy | Student samplea → teacher solo prefilling para logits |
| Hyperparams clave | $\alpha$ (peso teacher/student), $\beta$ (KL) | $\beta$ (escala log-ratio) |

**Interpretación:** π-Distill y PIAR comparten el spirit (PI teacher + agentic) pero divergen en la mecánica:
- π-Distill **entrena al teacher** activamente con $\mathcal{J}_{\text{Teacher}}$ para que aprenda a usar bien la PI, después destila al student via KL.
- PIAR **no entrena al teacher** — usa el modelo inicial como evaluador con golden context para extraer una señal por step.

**Caveat importante (codex review):** como π-Distill comparte parámetros entre teacher y student, incluso con $\alpha = 1$ (solo teacher entrena en el loss combinado), el student **se mueve indirectamente** porque actualiza θ — y ese θ es el de ambos. La distinción "teacher entrena, student no" es operativa, no de parámetros. En PIAR no hay esta dualidad porque solo hay un objetivo y un set de gradientes; el teacher es snapshot frozen separado.

**Consecuencia operativa:** π-Distill es **off-policy** desde la perspectiva del student (rollouts vienen del teacher). PIAR es **on-policy** estricto (invariante 6 de PROJECT.md): el student genera sus propias trayectorias; el teacher solo emite logprobs sobre lo generado.

## 5. OPSD (de este paper) vs OPSD (de Zhao et al. #5)

El paper introduce también una versión "OPSD" (Eq. 5):

$$\mathcal{J}_{\text{OPSD}}(\theta) = \mathbb{E}_{\pi^S}\left[R(o, s)\right] - \beta D_{KL}\left( \pi^S(o \mid s) \,\|\, \text{sg}(\pi^T(o \mid s, I)) \right)$$

Esto es **on-policy** (samplea del student), con KL del student hacia el teacher (estilo PIAR — student trata de imitar al teacher).

**¿Es el mismo OPSD de Zhao et al. (issue #5, arxiv 2601.18734)?** Comparten el **núcleo objetivo** (student samplea + KL hacia teacher con PI). Penaloza cita a Zhao como "concurrent work". Pero hay diferencias de **contexto y aplicación**:
- **Zhao apunta a supervised reasoning con ground-truth answers** (math single-turn, distillation con CoT como PI).
- **Penaloza estudia agentic PI transfer** (multi-turn, transferencia de capabilities desde modelos frontera con razonamiento oculto).

Mismo objetivo formal, distinto régimen empírico. **Confluencia simultánea pero no idéntica.**

**Para PIAR:** valida nuestro framing — la idea "student samplea, teacher con PI da target denso" tiene independent rediscovery en dos venues. PIAR cambia el target denso de "KL hacia teacher" a "log-ratio teacher/student", manteniendo el spirit de Zhao+Penaloza.

## 6. ¿Discuten log-ratio como reward (estilo Yuan / iStar / PIAR)?

**No.** El paper no menciona Yuan, iStar, PRIME, ni la línea de implicit PRM. La elección entre KL y log-ratio no se evalúa explícitamente — simplemente eligen KL como continuation natural de la línea distillation.

Para PIAR esto refuerza el posicionamiento: la **intersección "same-model + privileged context + log-ratio + multi-turn agentic"** sigue sin ser explorada por la literatura. π-Distill es el vecino más cercano operativamente y conscientemente eligen otro camino (KL distillation), no porque hayan descartado log-ratio sino porque vienen de otra tradición.

## 7. Setup experimental

**Benchmarks:**
- **τ-Bench** (customer service, retail + airline): 500 train, 115 retail hold-out, 50 airline OOD.
- **TravelPlanner**: 45 train, 180 hold-out (con reward modificado para evitar reward hacking).
- **GEM** (out-of-domain): 7 datasets multi-turn QA.

**Modelos base:** Qwen3-4B, Qwen3-8B (reasoning), R1-Distill-Llama-8B.

**Baselines:** GRPO puro, SFT con/sin CoT, SFT+RL, OPSD.

**Detalles:**
- RL algorithm: **GRPO**. Mismo que PIAR.
- Context limit: 25K tokens (16K con length penalty).
- KL estimator: Monte Carlo per-token.
- $\beta$ swept en $\{0, 0.1, 0.25, 0.5\}$.

**PI source:** trayectorias de DeepSeek-chat-v3.1 (~16K en τ-Bench retail, ~2K en TravelPlanner).

## 8. Resultados — Qwen3-8B (Tabla 1)

| Método | TravelPlanner | τ-Bench Retail | τ-Bench Airline |
|---|---|---|---|
| Base | 23.6 | 3.4 | 6.4 |
| SFT w/ CoT | 26.0 | 16.5 | 5.3 |
| SFT w/o CoT + RL | 31.3 | 23.5 | 6.0 |
| **SFT w/ CoT + RL** (industry std) | 32.3 | 29.1 | 8.0 |
| **OPSD** | 37.5 | 27.3 | 14.0 |
| **π-Distill (α=0)** | 40.7 | 31.1 | 12.0 |
| **π-Distill (α=0.5)** | 41.1 | 30.6 | 7.3 |
| **π-Distill (α=1)** | **44.1** | 29.7 | 9.3 |

**Δ vs SFT+CoT+RL:** π-Distill α=1 supera por +11.8% en TravelPlanner. En τ-Bench retail mejora moderada; en τ-Bench airline pierde un poco con α=0.5 pero gana con OPSD (+6.0).

**OOD generalization (GEM):** Qwen3-8B muestra que π-Distill y OPSD superan SFT+CoT+RL OOD; en Qwen3-4B SFT+CoT+RL sigue siendo mejor OOD. **Modelos más grandes parecen extraer más beneficio del régimen distillation con PI.**

## 9. Ablations — los hallazgos GORDOS para PIAR

### 9.1 Token-level leakage OBSERVADO empíricamente — refuerzo crítico de D.1

> "Few tokens referencing PI... exhibited very high KL (e.g., token 'hint')... incorporate a penalty on frequency."

Esto es **exactamente** el riesgo de leakage que PIAR tiene identificado en D.1. π-Distill lo OBSERVA en producción y lo mitiga con un frequency penalty. Confirma que:
1. **El leakage token-level es real, medible y problemático en métodos PI.**
2. La mitigación posible incluye: detección por frecuencia anómala, penalty sobre tokens que referencian PI por marca textual, monitoreo activo de la distribución de KL/log-ratio por token.

**Caveat importante:** π-Distill reporta que la mitigación **casi no cambia el performance final**. Es decir, el leakage existe y es detectable, pero su impacto en métricas downstream fue moderado en su setup. Para PIAR esto significa: el leakage es un **diagnóstico útil** (entender qué está midiendo el reward) pero no necesariamente un killer del método. PIAR tiene que medir leakage **principalmente como sanity check semántico**, no porque vaya a destruir performance.

**Para PIAR:** esto refuerza D.2 (loggear contribuciones por tipo de token) y D.1 (medir leakage como diagnóstico semántico). π-Distill es evidencia externa de que la preocupación es genuina y operativizable. Su mitigación (frequency penalty) queda como candidata D.7 condicional, no como default.

### 9.2 Tipo de PI — Tool Calls & Arguments es el ganador

Figura 6 / 8: PI con máxima densidad (Tool Calls & Arguments) supera consistente a PI menos densa. **Excepción**: π-Distill α=1 puede aprender a usar PI menos denso vía $\mathcal{J}_{\text{Teacher}}$, mientras que OPSD (sin teacher training) es más sensible al tipo de PI.

**Para PIAR (donde teacher no se entrena, igual que OPSD):** esperar que el tipo de privileged context importe más que en métodos con teacher training. Default: empezar con PI denso (golden answer + CoT estilo OPSD/Zhao), explorar variantes menos densas como ablation.

### 9.3 α y β

- **α = 0.5** es el más robusto (gana en 7/16 escenarios, pierde en 1/16). Default recomendado por el paper.
- **β > 0** crítico para training estable cuando α > 0 (entrenamos al teacher).
- $\beta \in \{0.1, 0.25, 0.5\}$ es el rango efectivo.

**No directamente aplicable a PIAR** porque no entrenamos al teacher (α=0 efectivo en nuestro régimen). Pero sí informa: si en algún momento exploramos variantes co-evolutivas, $\alpha = 0.5$ y $\beta \in [0.1, 0.5]$ son arranque razonable.

### 9.4 Inicial KL como predictor

Inicial $D_{KL}(\pi^T_{\text{base}} \| \pi^S_{\text{base}})$ alta es predictor negativo de performance. Si el modelo base no rationaliza bien el PI desde el arranque, el método tiene dificultades.

**Para PIAR:** sugiere validar que el modelo base de PIAR rationaliza razonablemente el golden context antes de empezar el RL training. Una sanity check rápida: comparar logprobs de tokens correctos del student con vs sin golden — si la diferencia es muy chica desde el arranque, el modelo base no captura bien la asimetría.

## 10. Limitaciones declaradas

1. **Warm-start con SFT+CoT necesario para R1-Distill-Llama-8B**: del cero el método no puede empezar.
2. **OOD frágil en modelos pequeños**: los Qwen3-4B se benefician más de SFT+CoT+RL OOD, π-Distill solo gana en 8B.
3. **Token-level leakage** (ver §9.1).
4. **No comparan vs DPO/IPO multi-turn** (ej. SWEET-RL — interesante que se hayan saltado eso).
5. **Reward modification en TravelPlanner**: tuvieron que rediseñar el reward para evitar reward hacking.

## 11. Código

**Repo oficial placeholder existe** en [`Emilianopp/Privileged-Information-Distillation-and-Self-Distillation`](https://github.com/Emilianopp/Privileged-Information-Distillation-and-Self-Distillation), pero al momento de la lectura el README indica que **el código se publicará cuando tengan aprobación legal**. Sin código liberado todavía. Watch para cuando aparezca — sería referencia operativa muy útil.

## 12. Lo que esto significa PARA PIAR

### 12.1 Refuerzos sin nuevas decisiones

- **D.1 (medir leakage vs progreso causal):** **MUY reforzado.** π-Distill OBSERVA leakage token-level empíricamente. Es evidencia externa fuerte de que la preocupación de PIAR es genuina. La mitigación de π-Distill (frequency penalty) es candidata para PIAR.
- **D.2 (loggear contribuciones por tipo de token):** reforzado. π-Distill identifica tokens específicos ("hint" entre otros) con KL anómala.
- **C.5 (forma del privileged context):** taxonomía concreta de π-Distill (Tool Calls & Arguments / Tool Calls Only / Self-Generated Hints) es directamente reusable para diseñar las variantes de ablation en PIAR.
- **B.7 (action-level / aggregation):** π-Distill usa GRPO + KL token-level. PIAR usa GRPO + log-ratio aggregado a action-level. Ambos son compatibles con GRPO; la elección de granularidad es ortogonal a si entrenamos al teacher.

### 12.2 Decisiones nuevas candidatas

- **D.7 (candidata):** **frequency penalty sobre tokens con log-ratio anómalamente alto** (estilo π-Distill mitigación token-level leakage). Activar solo si D.2 detecta el problema. Default: no aplicar; agregar como mitigación si la distribución de log-ratios muestra outliers.
- **D.8 (candidata):** **sanity check al inicio del training**: verificar que el modelo base de PIAR distinga razonablemente "con golden" vs "sin golden" — diferencia de logprobs no trivial sobre tokens correctos. Si no, el método tiene poco margen para mejorar (warning de π-Distill §9.4).

### 12.3 Argumento de defensa de PIAR refinado

π-Distill demuestra que la idea "PI teacher + agentic multi-turn" funciona (+11.8% TravelPlanner). PIAR difiere en:
1. **No entrena al teacher** → menor costo de cómputo, sin riesgo de colapso por shared-params + dual gradients.
2. **Log-ratio como reward** vs KL distillation → integración natural con GRPO advantage (estilo iStar), no requiere importance weighting off-policy.
3. **On-policy estricto** (invariante 6 PROJECT.md): student genera, teacher evalúa. π-Distill es off-policy del student.

Si un reviewer dice "ya hay π-Distill", la respuesta es: comparten spirit, divergen en mecánica. PIAR es la versión on-policy + log-ratio + teacher-frozen del concept.

### 12.4 Lo que PIAR NO debe ignorar

- **Warm-start puede ser necesario** (R1-Llama no arrancó del cero en π-Distill). Para PIAR significa: verificar que el modelo base haya pasado SFT en el domain antes del RL — y considerar SFT inicial como parte del pipeline si el base no es suficiente.
- **OOD generalization no es gratis**: π-Distill solo gana OOD en modelos grandes (8B). PIAR con el modelo correcto puede ser mejor o peor; no asumir que el método transfiere automáticamente.

## 13. Conexiones con otros papers

- **OPSD-Zhao (#5):** mismo concept aplicado en single-turn math. π-Distill cita explícitamente a Zhao como concurrent work. Confirmación de convergencia de la idea.
- **Yuan (#3) / iStar (#4) / PRIME (#6):** **NO se mencionan**. La línea log-ratio + implicit PRM no se cruza con la línea distillation con PI en este paper. PIAR es la unión.
- **SWEET-RL (#7):** **NO se menciona**. La línea critic asimétrico + BT no aparece. SWEET-RL y π-Distill son soluciones paralelas independientes al mismo problema (PI para credit assignment multi-turn).
- **Variational Reasoners (Zhou et al. 2025):** "most similar to π-Distill". Línea variacional general, no directamente relevante a PIAR.
- **STaR (Li et al. 2025):** versión SFT-only de la idea student=teacher. π-Distill argumenta que su KL-regularized RL es mejor que SFT puro.

## 14. Lo más importante para retener

1. **π-Distill OBSERVA leakage token-level empíricamente.** D.1 de PIAR está reforzado por evidencia externa. La mitigación (frequency penalty) es candidata D.7.
2. **Taxonomía de PI types** (Tool Calls & Args / Tool Calls Only / Self-Hints) es reusable para C.5 — define cómo variar el privileged context en PIAR ablations.
3. **OPSD-Zhao y OPSD-Penaloza son concurrent rediscovery** del mismo concept. Validación de la dirección general de PIAR.
4. **π-Distill entrena al teacher; PIAR no.** Diferencia clara y defendible. Ventaja PIAR: simplicidad operativa, on-policy estricto, sin riesgo de colapso shared-params.
5. **Sanity check inicial sugerido** (D.8 candidata): verificar que el modelo base distingue golden vs no-golden antes de RL training.
6. **No usan log-ratio como reward.** PIAR sigue siendo único en la combinación log-ratio + privileged context + multi-turn agentic.
