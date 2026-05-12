# PIAR — Delta deep-dive vs π-Distill α=0 ("OPSD-Penaloza")

> **Qué es esto:** doc focalizado que responde honestamente a la pregunta "¿esto no es lo mismo que X?" contra el **vecino más cercano de PIAR en la literatura**: π-Distill en el régimen α=0 (también conocido como "OPSD-Penaloza", Eq. 5 del paper π-Distill).
>
> **Cuándo NO usar este doc:** para la vista multi-paper "qué hereda PIAR de cada vecino" → ir a [`papers-cross-mapping.md`](papers-cross-mapping.md). Este doc es deep-dive sobre un solo vecino — el más peligroso para defender en review.
>
> **Cuándo SÍ usar este doc:** cuando estás escribiendo el related work del paper, preparando una defense para un reviewer escéptico, o evaluando si una decisión de diseño cambia la posición de PIAR vs OPSD-Penaloza α=0.
>
> **Status:** vivo. Cada delta marcado como **🔒 estructural** (claro y defendible sin experimentos) o **🟡 empírico** (depende de mediciones que todavía no hicimos).

---

## 1. Qué es PIAR, en una frase precisa

> **PIAR = un agente RL multi-turn entrenado con GRPO, donde el step reward por acción ReAct se obtiene del log-ratio entre dos forward passes del MISMO modelo: uno con golden answer / SCM en el contexto (teacher), otro sin (student). El teacher usa la snapshot `π_old` reciente (igual al denominador del log-ratio en iStar); el student es quien genera las trayectorias y se actualiza.**

Componentes sin redondeo:

1. **Mismo modelo, asimetría solo en contexto.** Teacher y student comparten arquitectura y pesos; el teacher tiene info privilegiada en su prompt. (Invariante 4.)
2. **Teacher = `π_old` snapshot reciente.** No se entrena nada. Frozen θ₀ (estilo OPSD-Zhao) queda como ablation de estabilidad, no como default — porque rompe "mismos pesos del student" después del primer update y mezcla efecto-contexto con weight-drift. Ver [`design-decisions.md`](design-decisions.md) C.2 (actualizada 2026-05-11).
3. **Student on-policy estricto.** El student genera; el teacher solo puntúa lo que el student hizo. (Invariante 6.)
4. **Señal por acción, no por token.** Log-ratio agregado sobre el span de la acción ReAct completa (CoT + tool call). (Decisión heredada de iStar.)
5. **Reward, no loss.** El log-ratio escalado por β entra como step reward al advantage de GRPO. Se combina con outcome reward via advantage normalization estilo iStar (no via β como regularizador). (Decisiones B.1, B.3, B.4 en `design-decisions.md`.)
6. **Privileged context loggeable, reproducible, determinístico.** (Invariante 5.)

---

## 2. El vecino más cercano y la fuente principal de confusión: π-Distill α=0

π-Distill (Penaloza et al. 2026, [`paper-pi-distill.md`](../notes/paper-pi-distill.md)) **no es un solo método**, son **tres regímenes según α**:

```
J_πDistill = α · J_Teacher + (1-α) · J_Student
```

| Caso | Quién samplea | Teacher se entrena? | Loss principal |
|---|---|---|---|
| **α=1** | Teacher (con PI) | Sí | $E[R] - \beta KL(\pi^T \| sg(\pi^S))$ |
| **α=0.5** | Mix | Parcial | combinado |
| **α=0** ("OPSD-Penaloza") | **Student** | **No (stop-grad sobre params compartidos)** | $E_{\pi^S}[R] - \beta KL(\pi^S \| sg(\pi^T))$ |

**El vecino más cercano de PIAR es el caso α=0**, NO el caso α=1. En el caso α=0:
- ✅ Mismo modelo (igual que PIAR).
- ✅ Teacher no se entrena, via stop-grad — funcionalmente equivalente a usar la snapshot actual del student como teacher (igual que PIAR con `π_old`).
- ✅ Student samplea (igual que PIAR).
- ✅ Outcome reward presente (igual que PIAR).

**La distinción NO es "ellos entrenan al teacher, nosotros no".** Eso aplica al caso α=1, no al α=0. Y NO es "ellos usan frozen θ₀, nosotros no" — α=0 con stop-grad usa la snapshot current de los params, no θ₀ inicial. La sub-decisión `π_old` vs frozen θ₀ vive en C.2 / E.3 y es ortogonal a la comparación PIAR vs α=0.

### 2.1 Las TRES diferencias técnicas con π-Distill α=0

| Dimensión | π-Distill α=0 | PIAR | Status |
|---|---|---|---|
| **Cómo entra la señal del teacher** | KL en el loss como regularizador: $-\beta KL(\pi^S \| \pi^T)$ | Log-ratio como reward por acción que entra al advantage de GRPO | 🟡 Empírico |
| **Granularidad** | Token-level KL (estimador Monte Carlo per-token) | Action-level (span ReAct completo: CoT + tool call) | 🟡 Empírico |
| **Combinación con outcome** | β fijo multiplicando KL crudo, sumado a $E[R]$ | Step-advantage normalizado por batch stats antes de combinarse con outcome-advantage (estilo iStar): $A = A^E + \alpha A^S$ | 🟡 Empírico |

**Los tres deltas son técnicamente verificables pero no están validados empíricamente.** Su importancia depende de los experimentos.

### 2.2 La conexión matemática que hay que ver clara

Hay una identidad útil: para una distribución $\pi^S$ y $\pi^T$ fijas, $E_{a \sim \pi^S}[\log \pi^T(a) - \log \pi^S(a)] = -KL(\pi^S \| \pi^T)$.

Es decir, **maximizar el log-ratio como reward (PIAR) es, en expectativa sobre el sampling del student, equivalente a minimizar el reverse-KL (π-Distill α=0)**. La diferencia operativa real vive en:

- **Granularidad de agregación** (action-level vs token-level).
- **Normalización del estimador** (advantage normalization vs β crudo).
- **Cómo el optimizer trata la señal** (clipping y trust region de PPO/GRPO sobre el step advantage vs gradient flow del KL como término del loss).

**Esto es honesto pero incómodo para PIAR**: no es una idea conceptualmente nueva. **Es un refinamiento técnico de un concept que π-Distill α=0 ya implementó**. La validez como paper independiente depende de si los tres deltas técnicos producen ganancias empíricas medibles **o** si el análisis de leakage (§5) resulta ser contribución suficiente por sí solo.

---

## 3. Vecinos secundarios y deltas

> **Vista multi-paper completa en [`papers-cross-mapping.md`](papers-cross-mapping.md).** Esta sección comprime los deltas más relevantes para no perder de vista que π-Distill α=0 NO es el único vecino.

### 3.1 OPSD-Zhao (single-turn math, paper #5)

[`paper-opsd.md`](../notes/paper-opsd.md). Concurrent rediscovery de π-Distill α=0 pero en **single-turn matemática**, con forward KL token-level entre teacher y student.

- ✅ Comparte: same model, KL del student hacia teacher con stop-grad, outcome reward.
- ❌ Difiere: single-turn math vs multi-turn agentic. Sin tool calls, sin estado del environment, sin créditos a asignar entre pasos.
- ❌ Detalle clave: OPSD-Zhao usa frozen θ₀ explícitamente (no stop-grad sobre params current). Es lo que disparó la sub-decisión C.2 original "frozen como default" — revisada 2026-05-11 a `π_old` porque frozen θ₀ rompe invariante 4 en multi-turn.
- **Delta de PIAR vs OPSD-Zhao:** los mismos tres de §2.1, **más** la transición a multi-turn agentic (que ningún paper validó todavía con este mecanismo). 🔒 Estructural en lo de single→multi-turn; 🟡 empírico en los tres técnicos.

### 3.2 SWEET-RL (asymmetric critic, paper #7)

[`paper-sweet-rl.md`](../notes/paper-sweet-rl.md). Multi-turn collaborative; PI vive en un **critic separado** entrenado con Bradley-Terry sobre pares (preferida, no preferida).

- ❌ El actor de SWEET-RL **nunca ve la PI**, ni directa ni indirecta. La PI se filtra a un escalar via el critic.
- ❌ El critic SE ENTRENA con BT, mientras que en PIAR no entrenamos nada extra.
- **Delta de PIAR vs SWEET-RL:** estructural y grande. PIAR usa la PI directamente via log-ratio entre dos forward passes; SWEET-RL la pasa por un Bradley-Terry training de un critic separado. **🔒 Estructural.**
- Pero ojo: SWEET-RL es la **familia conceptual** de "training-time privileged info" — comparte motivación, no mecanismo.

### 3.3 iStar (multi-turn agentic, paper #4)

[`paper-istar.md`](../notes/paper-istar.md). Implicit PRM derivado del policy mismo (Yuan-style) + RLOO + advantage normalization.

- ❌ iStar **no tiene privileged context teacher**. El "teacher" es un PRM separado entrenado vía DPO sobre outcome rankings, no un modelo con PI.
- ✅ Comparte con PIAR: action-level granularity, advantage-level normalization, β=0.05, sin KL penalty, agente ReAct multi-turn.
- **Delta de PIAR vs iStar:** PIAR reemplaza el PRM entrenado de iStar por el mismo modelo con golden en prompt. iStar es la base operacional sobre la cual PIAR se monta. **🔒 Estructural.**
- iStar es **complementario** a PIAR, no competidor. El experimento primario PIAR vs iStar (ver `PROJECT.md` "Reformulación operativa contra iStar") aísla esta diferencia.

### 3.4 PRIME (paper #6)

[`paper-prime.md`](../notes/paper-prime.md). Implicit PRM con online update via CE outcome + LOO baseline.

- ❌ PRIME **entrena el implicit PRM** con outcome label. PIAR no entrena nada.
- ❌ PRIME no usa privileged context.
- **Delta de PIAR vs PRIME:** dos deltas — (a) sin training del PRM, (b) PI en el contexto del teacher. **🔒 Estructural.**

### 3.5 Yuan 2024 — Implicit PRM (paper #3)

[`paper-yuan-implicit-prm.md`](../notes/paper-yuan-implicit-prm.md). Identidad algebraica soft-Q sobre log-ratio entre dos modelos.

- 🔧 PIAR **usa el resultado de Yuan como base teórica**, pero aplicado a modelos cuyo prompt difiere (no a modelos cuyos pesos difieren).
- **Delta:** Yuan no es un método aplicable, es una identidad algebraica que justifica matemáticamente que el log-ratio de PIAR es una soft-Q válida del reward "cuánto racionaliza el contexto golden esta trayectoria". Yuan no propone PIAR; PIAR es una instancia particular del framework de Yuan.

### 3.6 Math-Shepherd (paper #8)

[`paper-math-shepherd.md`](../notes/paper-math-shepherd.md). MC step labels con tree search.

- 🔧 Es **predecesor histórico** del paradigma de step-level reward sin step labels manuales.
- **Delta:** ortogonal. Math-Shepherd genera labels offline con MC; PIAR genera reward online con dos forward passes. Distintos costos, distintos métodos. No competidor directo.

---

## 4. Las tres apuestas empíricas de PIAR

Si PIAR es **publicable como contribución independiente del análisis de leakage**, es porque al menos UNA de las siguientes hipótesis se valida con experimentos. Sin ninguna **y** sin un buen resultado en §5, PIAR es un refinamiento técnico de π-Distill α=0.

### 4.1 Apuesta A — Action-level granularity gana sobre token-level (🟡)

**Hipótesis:** $r_{\text{PIAR}}^t$ agregado sobre el span de la acción ReAct completa produce credit assignment más limpio que KL token-level (estilo π-Distill α=0).

**Evidencia indirecta a favor:** iStar Tabla 3 — action-level + advantage-level normalization 94.7 vs token-level 90.0 en su setup (sin privileged context). La hipótesis es que el efecto se transfiere al régimen privileged-context.

**Cómo se valida:** ablation directa PIAR (action-level) vs versión token-level del mismo método, mismo modelo, mismo benchmark. Si la diferencia es < 1 punto, la apuesta no se sostiene.

### 4.2 Apuesta B — Log-ratio como reward gana sobre KL como regularizador (🟡)

**Hipótesis:** integrar la señal del teacher como step reward (con clipping y normalización de PPO/GRPO sobre el step advantage) es más estable y se combina mejor con outcome reward que el KL como término del loss con β fijo.

**Evidencia indirecta a favor:** iStar y PRIME ambos usan log-ratio en advantage; π-Distill α=0 usa KL como regularizador y reporta que **β > 0 es crítico** para estabilidad. Sugerencia (no prueba) de que el régimen reward-based tiene márgenes mejores.

**Cómo se valida:** ablation directa PIAR vs implementación literal de π-Distill α=0 sobre el mismo benchmark, mismo modelo. Si PIAR no supera por al menos 2-3 puntos consistentemente, la apuesta no se sostiene.

**Riesgo identificado:** la conexión $E_{\pi^S}[\log \pi^T - \log \pi^S] = -KL(\pi^S \| \pi^T)$ implica que las dos formulaciones son equivalentes en expectativa. La ventaja de PIAR depende de propiedades del **estimador finito** (varianza, normalización), no del objetivo en el límite.

### 4.3 Apuesta C — `π_old` reciente preserva la propiedad que OPSD-Zhao mostró con frozen θ₀ (🟡)

**Hipótesis:** mantener el teacher como `π_old` (snapshot reciente del student) en multi-turn agentic preserva la ganancia que OPSD-Zhao reportó con frozen θ₀ en single-turn math, **mientras** mantiene invariante 4 (mismos pesos del student al tiempo del log-ratio).

**Evidencia indirecta a favor:** OPSD-Zhao gana +5.7% sobre GRPO en single-turn math con frozen θ₀. π-Distill α=0 usa stop-grad sobre params current (funcionalmente equivalente a `π_old`) y reporta resultados positivos en agentic. La hipótesis es que `π_old` se sostiene.

**Riesgo identificado por PRIME:** un PRM con loss CE outcome puede degradarse a medida que el policy se aleja del SFT inicial. El teacher de PIAR mide algo distinto (delta context-induced sobre `π_old`) y en principio es más estable, pero **no está medido en multi-turn**. E.7 en `design-decisions.md` cubre la verificación.

**Cómo se valida:** ablation directa `π_old` vs frozen θ₀ vs re-snapshot cada N steps (E.3 en `design-decisions.md`). Durante RL training, comparar correlación de $r_{\text{PIAR}}^t$ con outcome a step 0 vs step N. Si `π_old` se mantiene estable → C.2 cierra como sólida. Si decae → frozen θ₀ o re-snapshot como alternativas.

---

## 5. La pregunta crítica que es ortogonal a las tres apuestas

**D.1 + D.9 — Leakage vs progreso causal.** Ver [`design-decisions.md`](design-decisions.md) D.1 y D.9.

Esta pregunta NO es sobre el delta de PIAR vs π-Distill α=0. Es sobre si **el log-ratio entre teacher con PI y student** —independiente de si se usa como reward o como KL regularizer, independiente de granularidad— **mide progreso causal o solo similitud textual con la PI**.

El problema afecta por igual a OPSD-Zhao, π-Distill α=0 y PIAR. **OPSD-Zhao no lo midió. π-Distill lo observó parcialmente** (frequency penalty sobre tokens con KL anómalo) **y reporta que la mitigación apenas mueve el performance final**.

**Si PIAR mide D.1 + D.9 rigurosamente** (versión textual via muestreo pareado + versión existencial via shuffled-golden control — "¿el método mejora aunque metamos la golden de otra tarea?"), **eso solo es contribución suficiente** incluso si las tres apuestas A/B/C no producen ganancias claras. Es la pregunta que toda la familia tiene pendiente.

---

## 6. Lo que hay que poder defender en review

Si en un review alguien pregunta "¿esto no es π-Distill α=0 con cambios técnicos?", la respuesta honesta es:

> **PIAR comparte spirit con π-Distill α=0** (same model, teacher no entrenado, student on-policy, outcome reward). **Difiere en tres aspectos técnicos** (action-level vs token-level, log-ratio-as-reward vs KL-as-regularizer, advantage normalization estilo iStar). **La importancia empírica de esos tres deltas es exactamente lo que los experimentos miden** (apuestas A, B, C). **Independientemente de A/B/C, contribuimos un análisis riguroso de leakage textual + existencial** (D.1 + D.9) **que ninguno de los papers vecinos hizo a fondo**.

Esa frase tiene que sostenerse al final del proyecto. **Si no se sostiene, el paper no se publica como independiente.**

---

## 7. Versión corta para uso interno

Cuando estés pensando en PIAR y te asalta la duda "¿no es lo mismo que self-distillation / OPSD / π-Distill?", recordá esto:

| Pregunta | Respuesta corta |
|---|---|
| ¿Es self-distillation? | No, porque es RL (reward, no loss); pero comparte el spirit. |
| ¿Es π-Distill? | Caso α=1, no. Caso α=0, **muy parecido en spirit**. Difiere en tres cosas técnicas (§2.1). |
| ¿Es OPSD-Zhao? | Mismo spirit, pero OPSD-Zhao es single-turn math con frozen θ₀. PIAR es multi-turn agentic con `π_old`. |
| ¿Es SWEET-RL? | No. SWEET-RL filtra la PI por un critic separado entrenado con BT. PIAR usa la PI directa via log-ratio. |
| ¿Es iStar? | No. iStar no usa privileged context (entrena un PRM separado con DPO sobre outcomes). PIAR se monta sobre la base operacional de iStar reemplazando ese PRM por el mismo modelo con golden en prompt. |
| ¿Es PRIME / Yuan? | No. Esos no usan privileged context y entrenan el PRM. PIAR usa privileged context y no entrena nada. |
| ¿Y entonces qué es lo nuevo? | La unión "same-model + privileged-context + log-ratio-as-reward + action-level + multi-turn agentic + `π_old`" no la propuso nadie todavía. **Pero es una unión, no un componente nuevo.** Su valor depende de los experimentos. |

---

## 8. Cuándo este doc se actualiza

- ✏️ Si aparece un paper nuevo cercano (especialmente: cualquier extensión de OPSD, π-Distill, SWEET-RL, iStar, o multi-turn privileged-context).
- ✏️ Cuando se libere el código de π-Distill (placeholder al 2026-05) — comparación directa contra ese código se vuelve posible.
- ✏️ Después de cada experimento que valide o invalide una de las apuestas A/B/C.
- ✏️ Después de D.1 / D.9 — el resultado redefine el framing del paper.
- ✏️ Si emerge una decisión de cambiar el framing (ej. dejar de competir con π-Distill α=0 y reposicionar como "extensión multi-turn + análisis de leakage").
- ✏️ Si C.2 cambia (ej. de `π_old` a frozen θ₀, o a re-snapshot cada N steps) — actualizar §1 punto 2, §2 "no es 'ellos usan frozen'", §3.1 detalle OPSD-Zhao, y §4.3 Apuesta C.
