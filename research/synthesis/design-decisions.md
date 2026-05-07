# PIAR — Decisiones de diseño (síntesis)

> **Qué es esto:** índice central de las decisiones de diseño de PIAR con su trazabilidad. Cada fila apunta al paper / análisis / issue de donde sale la decisión y su status (cerrada / inclinada / abierta).
>
> **Qué NO es:** no es una vision (eso está en `PROJECT.md`), no es un roadmap (eso está en Project v2 y CURRENT_STATE), no es notas de papers individuales (eso está en `research/notes/`).
>
> **Cuándo actualizar:** al cerrar un issue de research, al tomar una decisión nueva, al cambiar el status de una decisión existente. Los cambios significativos también van al CHANGELOG con ref `#N`.
>
> **Status legend:**
> - 🔒 **Cerrada** — decidida, no se reabre sin disparador explícito.
> - ➡️ **Inclinada** — default informado por evidencia, validación empírica pendiente.
> - 🟡 **Abierta** — pendiente de resolución.
> - 🅿 **Parked** — disponible pero no activa.

---

## A · Decisiones de stack y compute

| # | Decisión | Status | Fuente / análisis | Issue |
|---|---|---|---|---|
| A.1 | Framework base: **prime-rl + verifiers** (Prime Intellect). | 🔒 Cerrada (2026-05-04) | [`research/notes/repos-mapping.md`](../notes/repos-mapping.md) — comparación detallada vs veRL / OpenRLHF. Plan B documentado: veRL. Plan C parked: pedir código a iStar. | [#11](https://github.com/lucaspecina/piar-rl/issues/11) |
| A.2 | **Plan B reabierto** (PIAR sobre veRL fork de iStar) como alternativa concreta. | 🅿 Parked (disponible) | Verificación 2026-05-07: código de iStar liberado en [`Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar) (ICLR 2026). Estimación líneas: 30-100 vs 80-250 en Plan A. Reactivar si Plan A resulta caro al iniciar fase 4. | [#4](https://github.com/lucaspecina/piar-rl/issues/4) |
| A.3 | Compute pesado: **Azure ML (Y-TEC)** con A100/H100. | 🔒 Cerrada (2026-05-07) | Crédito Y-TEC ya disponible. Setup propio sobre máquinas existentes. **Prime Intellect Lab descartado** (agrega costo y dependencia externa sin upside claro; prime-rl/verifiers son open-source y se corren igual fuera de Lab). | (memoria del proyecto) |

---

## B · Decisiones de formulación matemática

| # | Decisión | Status | Fuente / análisis | Issue |
|---|---|---|---|---|
| B.1 | **Reward primitivo** = log-ratio escalado por β: $r_{\text{PIAR}}(\mathbf{y}) = \beta \log \pi_{\text{teacher}}(\mathbf{y} \mid x, \text{golden}) / \pi_{\text{student}}(\mathbf{y} \mid x)$. Mantener β explícito en la fórmula. | 🔒 Cerrada | [`paper-yuan-implicit-prm.md`](../notes/paper-yuan-implicit-prm.md) §2, §8. Prop 3.1 de Yuan da la igualdad soft-Q algebraica. La formulación con β explícito es necesaria para no desnormalizar la escala de los hyperparams experimentales. | [#3](https://github.com/lucaspecina/piar-rl/issues/3) |
| B.2 | **β = 0.05** como punto de partida. | ➡️ Inclinada | Yuan §4.1 y iStar (`paper-istar.md` §6) usan ambos β=0.05. Consistencia cross-paper. Tunear si la dinámica de PIAR lo requiere. | [#3](https://github.com/lucaspecina/piar-rl/issues/3), [#4](https://github.com/lucaspecina/piar-rl/issues/4) |
| B.3 | **Granularidad: action-level (no token-level).** Sumar log-ratio sobre el span de la acción ReAct completa (CoT + acción ejecutable), NO por token. | ➡️ Inclinada | iStar ablation Tabla 3 (`paper-istar.md` §8): action-level + advantage-level 94.7 vs token-level 90.0 vs reward-level 90.7 en WebShop Score. | [#4](https://github.com/lucaspecina/piar-rl/issues/4) |
| B.4 | **Combinar advantages, no rewards.** Normalizar $A^E$ (outcome) y $A^S$ (step) con stats de batch antes de combinar: $A(a_t) = A^E + \alpha A^S$. | ➡️ Inclinada | iStar (`paper-istar.md` §4 + ablation Tabla 3): advantage-level normalization gana sobre reward-level merging (94.7 vs 90.7). | [#4](https://github.com/lucaspecina/piar-rl/issues/4) |
| B.5 | **α = 1** como default entre $A^E + \alpha A^S$. | ➡️ Inclinada | iStar (`paper-istar.md` §4 + §6) reporta α=1 sin ablation explícito de α. Default razonable; tunear si la balance episode/step queda mal. | [#4](https://github.com/lucaspecina/piar-rl/issues/4) |
| B.6 | **Sin KL penalty** en el policy loss. | ➡️ Inclinada | iStar (`paper-istar.md` §4) lo omite y funciona. Default conservador para PIAR; reabrir si hay inestabilidad de training. | [#4](https://github.com/lucaspecina/piar-rl/issues/4) |
| B.7 | **Importance ratio agregado a nivel acción** en GRPO surrogate (no token-level). Internamente se usan logprobs token-level pero se agregan al span de la acción; tokens del environment se excluyen. | ➡️ Inclinada | iStar (`paper-istar.md` §4 nota técnica) — review crítico de codex confirmó la distinción. | [#4](https://github.com/lucaspecina/piar-rl/issues/4) |

---

## C · Decisiones del teacher (privileged-context)

| # | Decisión | Status | Fuente / análisis | Issue |
|---|---|---|---|---|
| C.1 | **Teacher = mismos pesos del student + golden answer en contexto.** Misma arquitectura, misma instancia del modelo; la asimetría vive solo en el prompt. | 🔒 Cerrada (invariante 4 PROJECT.md) | Tesis central de PIAR. NO negociable. Si el teacher fuera otro modelo (más grande, distilled, con LoRA exclusivo), entraríamos en la línea SWEET-RL / distillation guiada — otro paper. | (PROJECT.md inv 4) |
| C.2 | **Teacher frozen al checkpoint inicial** del student (estilo OPSD), no co-evolutivo (estilo Skill-SD). Default; cierre definitivo merece validación empírica en el régimen multi-turn log-ratio. | ➡️ Inclinada (2026-05-07) | OPSD (`paper-opsd.md` §3, §13.1): cita literal "We fix the teacher policy to be the initial policy [...] this helps stabilize training and implicitly acts as regularization to prevent excessive deviation from the initial policy." OPSD valida en single-turn distillation con KL forward; transferencia a multi-turn log-ratio merece chequeo. | [#5](https://github.com/lucaspecina/piar-rl/issues/5) |
| C.3 | **Template del prompt del teacher** estilo OPSD: "Here is a reference solution: [y\*]. After understanding the reference solution, please try to solve this problem using your own approach below: Answer:". Adaptar al setup multi-turn agentic. | ➡️ Inclinada | OPSD (`paper-opsd.md` §2 / Figura 2). Template ya iterado por OPSD; reusar como punto de partida. | [#5](https://github.com/lucaspecina/piar-rl/issues/5) |
| C.4 | **Teacher no genera tokens** — solo prefilling del student trajectory para obtener logits. Costo = 1 forward pass adicional sobre tokens del student. | 🔒 Cerrada | OPSD (`paper-opsd.md` §2) y iStar (`paper-istar.md` §2-§4) implementan así. Eficiente; sin fricción de implementación. | — |
| C.5 | **Privileged context** = solución completa con CoT, no answer literal solo. La forma específica para PIAR (CoT vs SCM vs trace exitoso) merece su propia ablation por el riesgo de leakage. | 🟡 Abierta | OPSD usa CoT completo y le va bien (`paper-opsd.md` §2). Para PIAR multi-turn agentic la mejor forma del privileged context es pregunta empírica abierta — answer literal maximiza leakage trivial, SCM lo reduce. **Ablation a hacer en fase 5.** | [#5](https://github.com/lucaspecina/piar-rl/issues/5), pendiente issue de ablation |

---

## D · Decisiones de medición y validación

| # | Decisión | Status | Fuente / análisis | Issue |
|---|---|---|---|---|
| D.1 | **Primer ablation crítico de PIAR: medir leakage vs progreso causal.** Comparar $r_{\text{PIAR}}^t$ contra un PRM entrenado o anotación humana en muestras donde la trayectoria repite texto del golden vs donde razona. Si correlaciona en correctas pero también sube en trayectorias que copian texto sin razonar, hay leakage y la métrica falla en su rol semántico. | 🔒 Cerrada como requisito | Yuan distinción algebraica vs semántica (`paper-yuan-implicit-prm.md` §4, §8). OPSD no analizó leakage (`paper-opsd.md` §10, §13.2) → gap real de literatura que PIAR debe cubrir. Si PIAR no demuestra esto, hereda la misma debilidad. | pendiente issue dedicado |
| D.2 | **Loggear distribución de contribuciones al log-ratio por tipo de token** (header ReAct vs contenido vs delimitadores). Si headers/delimitadores dominan, evaluar pointwise clipping estilo OPSD. | ➡️ Inclinada | OPSD pointwise KL clipping (`paper-opsd.md` §6, §13.3): sin clipping, tokens stylistic ("hmm", "maybe", "wait") dominan el loss y colapsa. Riesgo análogo en PIAR multi-turn con headers de acción ReAct. | pendiente issue dedicado |
| D.3 | **Replicar baseline iStar (RLOO + iStar) en WebShop** antes de modificar para PIAR. Hyperparameters conocidos (`paper-istar.md` §6). Sin baseline reproducido las comparaciones son ruido. | ➡️ Inclinada | Invariante 2 de PROJECT.md ("replicar antes de modificar") + iStar (`paper-istar.md` §10.3): código liberado, scripts ejecutables, hyperparams conocidos. Esto define la fase 3 del roadmap. | fase 3 (pendiente issues concretos) |
| D.4 | **Benchmark primario: WebShop**. Estándar de agentic RL. Después al menos un segundo (ALFWorld o SOTOPIA) para argumentar generalidad. | ➡️ Inclinada | `PROJECT.md` "Lo que PIAR quiere lograr" + iStar lo usa como primer benchmark. | (roadmap, fases 5-6) |

---

## E · Preguntas abiertas (de qué deberían salir las respuestas)

| # | Pregunta | Cómo se resuelve |
|---|---|---|
| E.1 | ¿El log-ratio teacher-context vs student-context **correlaciona con progreso causal hacia la respuesta correcta** o solo con compatibilidad textual / leakage del contexto privilegiado? | Ablation D.1 (medir leakage vs progreso causal en setup controlado). |
| E.2 | ¿Qué forma del **privileged context** (answer literal / CoT / SCM / trace exitoso) maximiza señal de progreso minimizando leakage? | Ablation C.5 después de tener D.1 montado. |
| E.3 | ¿Frozen al checkpoint inicial vs co-evolución funciona mejor en **multi-turn log-ratio**? OPSD validó frozen en single-turn KL; PIAR puede ser distinto. | Ablation directo en fase 5 si hay capacidad de compute para correr ambas variantes. |
| E.4 | ¿Hace falta **pointwise clipping estilo OPSD** en PIAR multi-turn span-level? | Decisión data-driven después de D.2 (loggear contribuciones por tipo de token). |
| E.5 | ¿El **teacher de PIAR como evaluador** sirve también como **teacher para policy improvement**? Yuan advierte que un buen PRM no implica buen policy (PRM evaluador ≠ buen actor). | Diagnóstico cuando arranquen experimentos de policy update. |
| E.6 | **Plan A (prime-rl) vs Plan B (veRL fork iStar)** — ¿se mantiene Plan A o se cambia? | Decisión al iniciar fase 4 después de leer el código de iStar como referencia. |

---

## F · Sub-decisiones derivadas que ya tienen artefacto

| # | Artefacto | Para qué |
|---|---|---|
| F.1 | `experiments/ENNN/manifest.yaml` con commit SHA + seeds + hyperparams + privileged context loggeado | Reproducibilidad por experimento (invariante 5 PROJECT.md). |
| F.2 | `research/notes/paper-<slug>.md` por paper analizado | Trazabilidad de cada decisión hacia su evidencia. |
| F.3 | Este archivo (`design-decisions.md`) | Índice central de decisiones; updatear cuando se cierra un research issue o se inclina/cierra una decisión. |

---

## G · Trazabilidad — papers analizados al 2026-05-07

| Paper | Notas | Issue | Aporte principal |
|---|---|---|---|
| Yuan 2024 — Implicit PRM | [`paper-yuan-implicit-prm.md`](../notes/paper-yuan-implicit-prm.md) | [#3](https://github.com/lucaspecina/piar-rl/issues/3) ✅ | Base teórica del log-ratio (igualdad soft-Q). Distinción algebraica vs semántica. |
| iStar — agentic RL multi-turn | [`paper-istar.md`](../notes/paper-istar.md) | [#4](https://github.com/lucaspecina/piar-rl/issues/4) ✅ | Action-level + advantage-level normalization. β=0.05, α=1, sin KL penalty. Código liberado (ICLR 2026). |
| OPSD — privileged-context teacher | [`paper-opsd.md`](../notes/paper-opsd.md) | [#5](https://github.com/lucaspecina/piar-rl/issues/5) ✅ | Same-model + golden context funciona. Teacher frozen al checkpoint inicial. Template del prompt. Riesgo leakage no analizado. |
| PRIME (pendiente) | — | [#6](https://github.com/lucaspecina/piar-rl/issues/6) ⏳ | — |
| SWEET-RL (pendiente) | — | [#7](https://github.com/lucaspecina/piar-rl/issues/7) ⏳ | — |
| Math-Shepherd (pendiente) | — | [#8](https://github.com/lucaspecina/piar-rl/issues/8) ⏳ | — |
| π-Distill (pendiente) | — | [#10](https://github.com/lucaspecina/piar-rl/issues/10) ⏳ | — |
| Síntesis cruzada (pendiente) | — | [#9](https://github.com/lucaspecina/piar-rl/issues/9) ⏳ | Cierre del epic. |
