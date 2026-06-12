# Propuesta de edición de PROJECT.md — pivot 2026-06

> **Qué es esto:** la propuesta completa de los cambios a `PROJECT.md` que el
> pivot 2026-06 requiere y que **necesitan aprobación explícita de Lucas**
> ([`pivot-2026-06.md`](pivot-2026-06.md) §11, bloque "requiere aprobación").
> **NO está aplicada** — PROJECT.md sigue reflejando el estado pre-pivot.
>
> **v2 (2026-06-12):** incorpora el review externo — cierres de los loopholes
> de los invariantes 10/11 (trayectoria-experta-como-hecho, selección como
> canal de opinión, lavado por imitación/destilación) y reordenamiento de
> HCAPO como vecino #1 tras su verificación independiente.
>
> **Cómo revisar:** cada cambio tiene bloque ANTES (texto actual) y DESPUÉS
> (texto propuesto). Los textos DESPUÉS están listos para copy-paste.
> Tracking: [#22](https://github.com/lucaspecina/piar-rl/issues/22).
>
> **Al aprobar (checklist para la sesión que aplique):**
> 1. Aplicar los cambios 1–6 a `PROJECT.md`.
> 2. Propagar a `CLAUDE.md` (sección "LA PREGUNTA" cita el texto viejo) y a
>    `README.md` si cita LA PREGUNTA o el roadmap.
> 3. Actualizar la memoria del proyecto (`project_piar_overview.md` describe
>    el delta pre-pivot).
> 4. Actualizar `design-decisions.md` N.2/N.3 (de "propuesta como invariante"
>    a "invariante 10/11 cerrado") y cerrar #22 con link al commit.

---

## Cambio 1 — Misión

### ANTES

> Investigar si **el log-ratio entre un teacher con información privilegiada
> (golden answer / SCM) y un student sin ella, sumado sobre el span de cada
> acción ReAct, sirve como step reward denso** para entrenar agentes RL
> multi-turn — y caracterizar dónde funciona y por qué.
>
> PIAR vive en la intersección de tres líneas existentes que no se cruzaron así
> todavía: **implicit PRM** (Yuan 2024), **privileged-context teacher** (OPSD
> 2026) y **agentic multi-turn RL** (iStar 2025). El delta concreto es la
> combinación, no inventar componentes nuevos.

### DESPUÉS

> Investigar si **un agente RL multi-turn puede recibir señal densa por acción
> derivada solo de información privilegiada fáctica y del propio modelo**,
> combinando dos lecturas complementarias del mismo par (modelo, PI):
>
> - **Forward (valor pragmático):** el incremento de probabilidad de la acción
>   cuando el scorer ve la PI — "¿actuaste como quien sabe?".
> - **Backward (valor epistémico):** el incremento de probabilidad de la PI
>   después de la acción — "¿ahora sabés más?".
>
> dosificando qué componentes de la PI ve el scorer según el **belief medido
> del propio student** (PI residual) — y caracterizar **en qué entornos hace
> falta cada componente y por qué**.
>
> Los componentes individuales ya existen en la literatura y la validan:
> el forward en variante (TAMTRL, OPCD, CriticSearch), el backward en search
> agents (IGPO). El delta de PIAR es **la descomposición bidireccional, la
> dosificación por belief y el mapa dónde/por qué** — un cruce de categorías
> que el survey de credit assignment (arXiv 2604.09459, 41+6 métodos) deja
> explícitamente vacío. Ver [`pivot-2026-06.md`](research/synthesis/pivot-2026-06.md)
> y la sección "Pivot 2026-06" de
> [`papers-cross-mapping.md`](research/synthesis/papers-cross-mapping.md).

---

## Cambio 2 — LA PREGUNTA

Reemplaza el bloque completo actual (LA PREGUNTA + "Reformulación operativa
contra iStar (2026-05-11)"). La reformulación 2026-05-11 no se borra de la
historia: queda en el tag `pre-pivot-2026-06` y en `piar-delta.md`; acá se
reemplaza por la versión post-pivot con una nota de lineage.

### ANTES

El bloque actual "## LA PREGUNTA" + "### Reformulación operativa contra
iStar (2026-05-11)" completo (líneas 20–54 del PROJECT.md actual).

### DESPUÉS

```markdown
## LA PREGUNTA

> **¿Puede un agente RL multi-turn recibir señal densa por acción derivada
> solo de información privilegiada fáctica y del propio modelo — combinando
> el incremento de probabilidad de la acción bajo PI (valor pragmático) con
> el incremento de probabilidad de la PI tras la acción (valor epistémico),
> dosificando qué PI ve el scorer según el belief medido del student — y en
> qué entornos hace falta cada componente y por qué?**
>
> Tres aristas que se evalúan juntas:
> 1. **Descomposición:** ¿pragmático + epistémico > cada uno solo, y el
>    balance depende de la información oculta del environment?
> 2. **Dosificación:** ¿la PI residual (gateada por belief) da señal igual o
>    mejor que la PI completa, con auto-annealing emergente?
> 3. **Sin overhead de entrenamiento:** todo con forward passes del mismo
>    modelo; ni juez entrenado, ni juez generativo, ni step labels, ni
>    rollouts extra.
>
> Aplicala al evaluar, diseñar, priorizar o revisar cualquier decisión.

### Reformulación operativa contra los baselines (2026-06-12)

Misma pregunta de fondo, planteada contra los baselines concretos:

> **¿El reward dual (±residual) supera a (a) outcome-only, (b) cada dirección
> sola [forward solo ≈ PIAR-viejo; backward solo ≈ IGPO trasplantado a
> ejecución], (c) iStar (privilegio en pesos de un juez entrenado), (d) GiGPO
> (estadística entre rollouts sin PI) — en WebShop y en ALFWorld, con los
> controles de leakage D.1/D.9 aplicados a ambas direcciones?**

Brazos experimentales y prioridades en `pivot-2026-06.md` §9 (A0–A4, B1–B2,
C1–C2). Predicciones pre-registradas y gate go/no-go (Figura 1) en §8.
Controles no-opcionales: shuffled-golden (D.9) sobre el forward Y el belief
del backward; el par mínimo outcome-label-in-prompt (C2), que post-HCAPO
sube de control interno de riqueza-vs-ubicación a **comparación directa
contra la familia hindsight publicada** (con el matiz de que C2 sí computa
el contraste dos-contextos que HCAPO no implementa).

> **Lineage:** la reformulación operativa anterior ("PIAR vs iStar como
> nearest-neighbor replacement study", 2026-05-11) quedó superada por el
> pivot 2026-06 — el forward standalone que esa versión aislaba fue publicado
> en variante (TAMTRL) y pasa de contribución a componente (brazo A1).
> Texto histórico: tag `pre-pivot-2026-06` + `research/synthesis/piar-delta.md`.
```

---

## Cambio 3 — "Lo que PIAR quiere lograr" (retoques menores)

Bullets que cambian (el resto queda igual):

### ANTES (bullet 1)

> - **Densidad de señal sin overhead**: rewards por acción sin PRM entrenado ni step labels manuales — solo dos forward passes.

### DESPUÉS (bullet 1)

> - **Densidad de señal sin overhead**: rewards por acción sin PRM entrenado,
>   sin juez generativo y sin step labels manuales — solo forward passes del
>   mismo modelo (dos por acción para el forward; belief por componente con
>   prefix caching para el backward).

### ANTES (bullet 4)

> - **Validación cuantitativa contra baselines fuertes**: WebShop primero (estándar de agentic RL), después al menos un benchmark más para argumentar generalidad.

### DESPUÉS (bullet 4)

> - **Validación cuantitativa contra baselines fuertes**: WebShop y ALFWorld
>   como benchmarks de primera clase (N.11 — en ALFWorld vive la predicción
>   del backward: estado oculto). Baselines obligatorios: iStar (privilegio en
>   pesos) y GiGPO (estadística entre rollouts sin PI, SOTA critic-free en
>   ambos benchmarks — N.10).

### ANTES (bullet 6, leakage)

> - **Análisis riguroso de leakage textual y estructural** del log-ratio teacher-privilegiado vs student. Esta es **contribución independiente del resto**: ninguno de los papers vecinos (OPSD-Zhao, OPSD-Penaloza, SWEET-RL, π-Distill) hizo este análisis a fondo. **Si las apuestas técnicas vs el vecino más cercano (π-Distill α=0) no producen ganancias medibles, este análisis solo es contribución suficiente.** Ver [`research/synthesis/piar-delta.md`](research/synthesis/piar-delta.md) §5 y `design-decisions.md` D.1 + D.9.

### DESPUÉS (bullet 6)

> - **Análisis riguroso de leakage textual y estructural**, ahora aplicado a
>   **las dos direcciones** (forward y backward). Sigue siendo contribución
>   independiente: ningún vecino (ni los nuevos — TAMTRL, IGPO, HCAPO) lo hizo
>   a fondo. **Fallback explícito del pivot (§10.4): si dual ≈ mejor
>   componente solo en ambos benchmarks, el estudio sistemático de las dos
>   direcciones + residual + leakage bidireccional es contribución de análisis
>   publicable igual.** Ver `design-decisions.md` D.1 + D.9.

---

## Cambio 4 — "Lo que NO es" (tres bullets nuevos)

Agregar al final de la lista existente (los bullets actuales quedan; el de
π-Distill y OPSD-Zhao siguen siendo correctos como historia del forward):

### DESPUÉS (agregar)

> - **No es TAMTRL.** TAMTRL (arXiv 2603.21663) usa el mismo mecanismo
>   same-model dos-contextos, pero: (i) usa la probabilidad del teacher a
>   secas, no el ratio teacher/student; (ii) su PI es un documento filtrado
>   por relevancia ("dónde mirar"), no golden estructurada en componentes;
>   (iii) vive en compresión long-context, no en agentes de decisión;
>   (iv) no tiene descomposición dual ni residual; (v) gatea el reward
>   multiplicativamente por el outcome binario — solo redistribuye crédito
>   dentro de éxitos, mientras PIAR da señal densa independiente del outcome.
>   Ver [`paper-tamtrl.md`](research/notes/paper-tamtrl.md) §8.
> - **No es la familia hints (QuestA, KnowRL, BREAD, Scaf-GRPO, ...).** En
>   toda esa familia la información entra al **prompt del student** durante el
>   rollout → contamina la política, crea train/test mismatch del input, y
>   exige retirada por schedule (QuestA) o minimalidad estática (KnowRL).
>   En PIAR la PI entra **solo al canal del reward** (invariante 11): el
>   student rollea siempre a ciegas y lo que se apaga (solo, vía el gating
>   por belief) es la señal. Ver [`paper-knowrl.md`](research/notes/paper-knowrl.md) §6–§7.
> - **No es un juez generativo (CriticSearch, C3, CCPO).** PIAR no le pide a
>   un LLM que *opine* scores post-hoc: los dos scores son cantidades
>   mecánicas de logprobs del mismo modelo (sin generación, sin rúbrica, sin
>   parsing de juicios).
> - **No es hindsight credit mecánico (HCAPO).** HCAPO es el vecino mecánico
>   más cercano (logprobs same-model, sin crítico entrenado) y el primero a
>   diferenciar en related work — por encima de TAMTRL. Las diferencias:
>   (i) su "PI" es el outcome realizado del propio rollout (consistencia con
>   *lo que pasó*); la de PIAR son hechos del dataset/simulador que el agente
>   nunca observó (consistencia con *la verdad*); (ii) su implementación
>   (Eq. 7) puntúa UN contexto de hindsight normalizado intra-trayectoria —
>   nunca computa el contraste con-PI/sin-PI que define al forward de PIAR;
>   (iii) multiplicativo sobre el return vs aditivo en el advantage; (iv) sin
>   backward, sin residual, sin mapa. El brazo C2 (outcome-label-in-prompt)
>   es la comparación directa contra esta familia. Ver
>   [`paper-menores-pivot-2026-06.md`](research/notes/paper-menores-pivot-2026-06.md) §1
>   (verificado dos veces, 2026-06-12).

---

## Cambio 5 — Invariantes nuevos 10 y 11

Agregar después del invariante 9, con el mismo formato:

### DESPUÉS (agregar)

> 10. **La PI vive en el espacio de hechos, nunca en el de acciones.** PI
>     válida = hechos sobre el **estado del mundo o el objetivo**, queryables
>     del dataset/simulador **sin ejecutar política alguna** (producto target,
>     atributos, ubicaciones de objetos). Dos exclusiones explícitas que
>     cierran los loopholes conocidos:
>
>     - **Demostraciones, trayectorias expertas y cualquier secuencia de
>       acciones quedan excluidas aunque estén en el dataset.** Una traza
>       experta es técnicamente "un hecho del dataset", pero usarla como PI
>       es exactamente la degeneración a imitación que este invariante
>       prohíbe. (Las trayectorias humanas de WebShop quedan confinadas a la
>       ablation A de C.5, reportada como tal — nunca como PI del método.)
>     - **La selección/orden de los hechos no puede ser canal de opinión.**
>       Toda función que decide qué componentes ve el scorer y en qué orden
>       depende solo de (dataset, estado del simulador, beliefs medidos bajo
>       π_old versionado) — nunca de juicios de utilidad-para-la-próxima-acción.
>       "Mejorar" el gating con heurísticas de relevancia para la acción
>       siguiente es meter política por la puerta de atrás vía curation.
>
>     Si la PI opina sobre acciones (directa o por curation), el reward
>     degenera en imitación del opinador y el método colapsa a la familia
>     juez-generativo/hints — otro paper. Extiende el invariante 5: además de
>     reproducible y verificable, la PI es *fáctica*. (Origen: N.2,
>     `pivot-2026-06.md` §4.1; cierres post-review externo 2026-06-12.)
>
> 11. **La PI influye en los pesos del student únicamente a través del
>     escalar `r_t` en el policy gradient.** Nunca como texto en el contexto
>     del student (training, eval o test), y **nunca como target de imitación,
>     distillation o fine-tuning** — generar texto condicionado a la PI y
>     destilarlo al student cumple la letra de "no está en su prompt" y viola
>     todo; queda explícitamente prohibido. La PI aparece únicamente en los
>     prompts del *scorer* (los forward passes que computan r_fwd y b_i), y
>     de ahí solo sale un escalar por acción — pocos bits, que es
>     precisamente lo que hace defendible el claim de no-contaminación.
>     Consecuencias: no hay train/test mismatch del input, no hay nada que
>     "retirar" del prompt con un schedule (lo que se apaga, vía el gating
>     por belief, es la señal), y el invariante 3 queda subsumido y
>     reforzado. Es el diferenciador estructural contra la familia hints
>     (QuestA/KnowRL/...) y contra la rama distillation (OPSD/π-Distill/OPCD).
>     (Origen: N.3, `pivot-2026-06.md` §2 y §7; cierre del lavado por
>     imitación post-review externo 2026-06-12.)

---

## Cambio 6 — Roadmap conceptual

### ANTES

La tabla actual de 8 fases (0–7), con WebShop como único benchmark primario,
ALFWorld en fase 6, y sin gate explícito antes de implementar.

### DESPUÉS

```markdown
| Fase | Estado | Objetivo |
|---|---|---|
| 0 — Bootstrap | ✅ Done | Estructura de docs + tracking + memoria. |
| 1 — Research | ✅ Done (2026-05-11) | Síntesis de los 7 vecinos originales (`papers-cross-mapping.md`, `design-decisions.md`). |
| 1b — Re-research pivot 2026-06 | 🟡 Now (vecinos consolidados 2026-06-12) | 6 vecinos nuevos (IGPO, TAMTRL, survey CA, GiGPO, KnowRL, menores) + decisiones N.1–N.12 + re-centrado en dual + residual. Epic [#19](https://github.com/lucaspecina/piar-rl/issues/19). Cierra con PROJECT.md actualizado (#22) y pre-registro de Figura 1 (#23). |
| 2 — Setup de compute | ⏳ Next | Azure ML Y-TEC (`lp-gpu-h100-x2-spot`, 2×H100 NVL). Spot → checkpointing no negociable. Sin cambios por el pivot — el pivot no requiere compute. |
| 3 — Replicación de baselines | ⏳ | iStar (B1, [#16](https://github.com/lucaspecina/piar-rl/issues/16)) **y GiGPO (B2, N.10 — trainer ya vendoreado en `code/examples/gigpo_trainer/`)** en WebShop, hyperparams de los papers. Sin baselines reproducidos las comparaciones son ruido. |
| 4 — Figura 1 (gate go/no-go) | ⏳ | Experimento día-1 sin RL: scoring forward/backward/dual sobre ~200 rollouts congelados en WebShop. Predicciones pre-registradas P1–P4 ([#23](https://github.com/lucaspecina/piar-rl/issues/23), `pivot-2026-06.md` §8). P1–P3 salen → luz verde a fase 5; cada escenario de fallo tiene salida definida. Requiere #17 (componentes g_i). |
| 5 — Implementación dual + residual | ⏳ | Modificación sobre `code/`: forward residual + belief tracker + gating + combinación en el advantage. El estimate ~150 LOC era para el forward solo — re-estimar (el backward + gating agregan superficie; `piar-implementation-points.md` necesita adendum). |
| 6 — Comparación + ablations | ⏳ | Brazos A0–A4 + B1/B2 + C1/C2 (`pivot-2026-06.md` §9) en **WebShop y ALFWorld (primera clase, N.11)**. Críticas: leakage bidireccional (D.1 + D.9), calibración de b_i (N.9), λ y normalización (N.5, N.7), thoughts en historial (N.6), gating duro vs soft (N.4). Secuencia mínima publicable según presupuesto §9. |
| 7 — Redacción | ⏳ | Draft del paper + naming del método (N.12) + related work con las diferenciaciones obligatorias (TAMTRL ×5, IGPO, HCAPO, familia hints). |
| 8 — Caso de estudio SREG (opcional) | ⏳ | SCM como información privilegiada — la cereza, no el plato principal. |
```

Nota sobre el plan temporal: el "2–3 meses de calendar time" del roadmap
anterior cubría fases 3–7 del diseño viejo (un solo método, un benchmark
primario). Con dual + residual + dos benchmarks + dos baselines, proponer
**re-estimar al cerrar la Figura 1** — el presupuesto honesto por brazo está
en `pivot-2026-06.md` §9.

---

## Lo que NO cambia (verificado contra el pivot §6)

- Invariantes 1–9 quedan textualmente como están (el 4 ya fue retocado en
  C.2 el 2026-05-11; `π_old` como scorer sigue válido para ambas direcciones).
- Jerarquía de decisión: sin cambios.
- "Lo que NO es" existente: sin cambios (los bullets de π-Distill / OPSD-Zhao
  / SWEET-RL siguen siendo correctos como historia del componente forward).
- Stack (`code/`, fork CharacterRL-iStar) y workflow: sin cambios.
