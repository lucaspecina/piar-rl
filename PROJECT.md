# PIAR — Vision

> Norte filosófico. Define qué es PIAR, por qué existe, qué fuerza, e
> invariantes que no pueden romperse. **No describe implementación ni estado actual.**
>
> Implementación: `ARCHITECTURE.md` (cuando exista) · Estado actual: `CURRENT_STATE.md` · Trabajo pendiente: [GitHub Project v2](https://github.com/users/lucaspecina/projects/5).

## Misión

Investigar si **un agente RL multi-turn puede recibir señal densa por acción
derivada solo de información privilegiada fáctica y del propio modelo**,
combinando dos lecturas complementarias del mismo par (modelo, PI):

- **Forward (valor pragmático):** el incremento de probabilidad de la acción
  cuando el scorer ve la PI — "¿actuaste como quien sabe?".
- **Backward (valor epistémico):** el incremento de probabilidad de la PI
  después de la acción — "¿ahora sabés más?".

dosificando qué componentes de la PI ve el scorer según el **belief medido
del propio student** (PI residual) — y caracterizar **en qué entornos hace
falta cada componente y por qué**.

Los componentes individuales ya existen en la literatura y la validan:
el forward en variante (TAMTRL, OPCD, CriticSearch), el backward en search
agents (IGPO). El delta de PIAR es **la descomposición bidireccional, la
dosificación por belief y el mapa dónde/por qué** — un cruce de categorías
que el survey de credit assignment (arXiv 2604.09459, 41+6 métodos) deja
explícitamente vacío. Ver [`pivot-2026-06.md`](research/synthesis/pivot-2026-06.md)
y la sección "Pivot 2026-06" de
[`papers-cross-mapping.md`](research/synthesis/papers-cross-mapping.md).

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
C1–C2). Predicciones pre-registradas y gate go/no-go (Figura 1) en §8 y en
[`figura1-prereg.md`](research/synthesis/figura1-prereg.md). Controles
no-opcionales: shuffled-golden (D.9) sobre el forward Y el belief del
backward; el par mínimo outcome-label-in-prompt (C2), que post-HCAPO sube de
control interno de riqueza-vs-ubicación a **comparación directa contra la
familia hindsight publicada** (con el matiz de que C2 sí computa el contraste
dos-contextos que HCAPO no implementa).

> **Lineage:** la reformulación operativa anterior ("PIAR vs iStar como
> nearest-neighbor replacement study", 2026-05-11) quedó superada por el
> pivot 2026-06 — el forward standalone que esa versión aislaba fue publicado
> en variante (TAMTRL) y pasa de contribución a componente (brazo A1).
> Texto histórico: tag `pre-pivot-2026-06` + `research/synthesis/piar-delta.md`.

## Lo que PIAR quiere lograr

- **Densidad de señal sin overhead**: rewards por acción sin PRM entrenado,
  sin juez generativo y sin step labels manuales — solo forward passes del
  mismo modelo (dos por acción para el forward; belief por componente con
  prefix caching para el backward).
- **Diferenciación por contexto, no por parámetros**: el "teacher" no es un modelo entrenado distinto, es el mismo modelo con más información en el prompt. La asimetría vive en el contexto, no en los pesos.
- **Plug-and-play sobre RL existente**: la salida de PIAR es un escalar por acción que entra al advantage de GRPO/PPO standard. No requiere refactor del optimizer ni del rollout loop.
- **Validación cuantitativa contra baselines fuertes**: WebShop y ALFWorld
  como benchmarks de primera clase (N.11 — en ALFWorld vive la predicción
  del backward: estado oculto). Baselines obligatorios: iStar (privilegio en
  pesos) y GiGPO (estadística entre rollouts sin PI, SOTA critic-free en
  ambos benchmarks — N.10).
- **Story explicable de las ablations**: si gana, entender por qué; si pierde, entender qué supuesto del paper falla. Ambos resultados son publicables.
- **Análisis riguroso de leakage textual y estructural**, ahora aplicado a
  **las dos direcciones** (forward y backward). Sigue siendo contribución
  independiente: ningún vecino (ni los nuevos — TAMTRL, IGPO, HCAPO) lo hizo
  a fondo. **Fallback explícito del pivot (§10.4): si dual ≈ mejor
  componente solo en ambos benchmarks, el estudio sistemático de las dos
  direcciones + residual + leakage bidireccional es contribución de análisis
  publicable igual.** Ver `design-decisions.md` D.1 + D.9.

## Lo que NO es

- **No es un nuevo framework de RL.** Modificamos quirúrgicamente lo que ya existe (verl / iStar). El delta esperado: 50–200 líneas.
- **No es trabajo de math reasoning single-turn.** Eso lo hace OPSD. PIAR vive en agentes multi-turn ReAct.
- **No es un PRM entrenable.** El teacher no se entrena; se le pasa más contexto. Esa es la apuesta diferenciadora vs iStar.
- **No es distillation con teacher más fuerte.** Si el teacher es un modelo distinto (más grande, mejor entrenado, distilled, ensembled), aunque tenga golden en contexto, ya no es PIAR — es la línea de SWEET-RL / distillation guiada por privileged info. Ahí el log-ratio mezcla efecto-de-contexto con efecto-de-modelo y la conclusión científica se diluye.
- **No es π-Distill.** π-Distill es **tres regímenes según α** (1, 0.5, 0). El caso α=1 entrena al teacher: PIAR no. **El caso α=0 ("OPSD-Penaloza")** comparte spirit con PIAR: same model, teacher no se entrena (via stop-grad sobre params compartidos), student samplea, outcome reward. **Acá el delta de PIAR es técnico, no conceptual**: (a) action-level vs token-level granularity, (b) log-ratio como reward que entra al advantage de GRPO vs KL como regularizador del loss, (c) advantage normalization estilo iStar vs β fijo. Las tres apuestas son **empíricas, no estructurales** — su validez depende de los experimentos. **Si los experimentos no muestran ventaja medible, PIAR colapsa a "refinamiento técnico de OPSD-Penaloza α=0".** Detalle completo en [`research/synthesis/piar-delta.md`](research/synthesis/piar-delta.md).
- **No es OPSD-Zhao.** Comparte spirit (same model + privileged context), pero OPSD-Zhao validó en single-turn matemática con forward KL token-level. PIAR vive en multi-turn agentic con log-ratio action-level como reward — la transición no está validada por nadie todavía.
- **No es construir un benchmark nuevo.** Usamos los existentes (WebShop, ALFWorld, SOTOPIA, τ-Bench). Eventualmente SREG como caso de estudio, pero no antes.
- **No es ajustar hyperparameters de un baseline.** Si la idea no gana por sí sola, las ablations explican por qué — no se rescata con HP search ad-hoc.
- **No es TAMTRL.** TAMTRL (arXiv 2603.21663) usa el mismo mecanismo
  same-model dos-contextos, pero: (i) usa la probabilidad del teacher a
  secas, no el ratio teacher/student; (ii) su PI es un documento filtrado
  por relevancia ("dónde mirar"), no golden estructurada en componentes;
  (iii) vive en compresión long-context, no en agentes de decisión;
  (iv) no tiene descomposición dual ni residual; (v) gatea el reward
  multiplicativamente por el outcome binario — solo redistribuye crédito
  dentro de éxitos, mientras PIAR da señal densa independiente del outcome.
  Ver [`paper-tamtrl.md`](research/notes/paper-tamtrl.md) §8.
- **No es la familia hints (QuestA, KnowRL, BREAD, Scaf-GRPO, ...).** En
  toda esa familia la información entra al **prompt del student** durante el
  rollout → contamina la política, crea train/test mismatch del input, y
  exige retirada por schedule (QuestA) o minimalidad estática (KnowRL).
  En PIAR la PI entra **solo al canal del reward** (invariante 11): el
  student rollea siempre a ciegas y lo que se apaga (solo, vía el gating
  por belief) es la señal. Ver [`paper-knowrl.md`](research/notes/paper-knowrl.md) §6–§7.
- **No es un juez generativo (CriticSearch, C3, CCPO).** PIAR no le pide a
  un LLM que *opine* scores post-hoc: los dos scores son cantidades
  mecánicas de logprobs del mismo modelo (sin generación, sin rúbrica, sin
  parsing de juicios).
- **No es hindsight credit mecánico (HCAPO).** HCAPO es el vecino mecánico
  más cercano (logprobs same-model, sin crítico entrenado) y el primero a
  diferenciar en related work — por encima de TAMTRL. Las diferencias:
  (i) su "PI" es el outcome realizado del propio rollout (consistencia con
  *lo que pasó*); la de PIAR son hechos del dataset/simulador que el agente
  nunca observó (consistencia con *la verdad*); (ii) su implementación
  (Eq. 7) puntúa UN contexto de hindsight normalizado intra-trayectoria —
  nunca computa el contraste con-PI/sin-PI que define al forward de PIAR;
  (iii) multiplicativo sobre el return vs aditivo en el advantage; (iv) sin
  backward, sin residual, sin mapa. El brazo C2 (outcome-label-in-prompt)
  es la comparación directa contra esta familia. Ver
  [`paper-menores-pivot-2026-06.md`](research/notes/paper-menores-pivot-2026-06.md) §1
  (verificado dos veces, 2026-06-12).

## Invariantes (NO negociables)

1. **Research-first, code-second.** No se toca código hasta que esté la
   síntesis cruzada de los papers vecinos. Saltarse esto lleva a
   implementaciones que no se pueden defender en review.
2. **Replicar antes de modificar.** Cuando llegue el código, primero
   reproducir el baseline exacto de iStar (o equivalente). Recién después
   modificar. Sin baseline reproducido, las comparaciones son ruido.
3. **Información privilegiada solo en training time.** El student NUNCA ve
   la golden answer / SCM en inferencia. Si esto se rompe, la propuesta
   es un benchmark contaminado.
4. **Teacher = mismos pesos del student; la asimetría vive SOLO en el contexto.**
   Esta es la tesis central de PIAR y NO se negocia.

   - **Lo no negociable:** el teacher tiene exactamente la misma arquitectura
     y los mismos pesos del student. **No es un modelo más fuerte, ni más
     grande, ni distilled, ni con LoRA/adapter exclusivo.** Si el teacher
     fuera distinto, el log-ratio mezclaría dos efectos (contexto + diferencia
     de modelo) y la conclusión científica se diluye. Esa es la línea de
     SWEET-RL / distillation guiada — otro paper.
   - **Justificación:** la fórmula
     `r(acción) = log π_teacher(y | x, golden) − log π_student(y | x)` mide,
     **cuando teacher y student son el mismo modelo**, el incremento de
     probabilidad que aporta la golden answer a esa acción. Es la única
     interpretación limpia de PIAR. Si los modelos difieren, la resta deja
     de medir solo el efecto del contexto.
   - **Sub-decisión INCLINADA (actualizada 2026-05-11):** **snapshot del
     teacher = `π_old`** (la misma snapshot reciente que se usa como
     denominador del log-ratio en iStar Eq. 1) como **default primario**;
     **frozen θ₀** (estilo OPSD-Zhao) degradado a **ablation legítima de
     estabilidad**. Ver [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md) C.2 y E.3.

     **Razón del cambio (2026-05-11, post-Codex review):** frozen θ₀ rompe
     "mismos pesos del student" después del primer update — al step t,
     θ_0 ≠ θ_t, y el log-ratio mezcla efecto-contexto con weight-drift.
     Con `π_old` en ambos términos del log-ratio (`log[π_old(a|x,golden) / π_old(a|x)]`),
     queda **pura diferencia de prompt** como única asimetría — la
     interpretación limpia que el invariante exige. OPSD-Zhao sigue siendo
     válido en single-turn KL, pero su transferencia a multi-turn log-ratio
     no preserva este invariante.

     **Cierre definitivo abierto en E.3:** ablation directo `π_old` vs
     frozen θ₀ vs re-snapshot cada N steps en fase 5, si hay compute. Si
     `π_old` se mantiene estable correlacionando con outcome → C.2 cierra
     sólida. Si decae → re-snapshot. Co-evolución estilo Skill-SD sigue
     dentro del invariante ("mismos pesos / misma arquitectura / sin adapter
     exclusivo") como variante legítima.
5. **Información privilegiada reproducible y verificable.** La golden answer /
   SCM debe ser automatable, loggable y determinística. Si requiere armado
   manual ad-hoc por trayectoria, estamos reintroduciendo step labeling por
   la puerta de atrás — y eso es justo lo que el método dice eliminar.

   - **"Determinística"** = función de (instrucción, estado, SCM) más versión
     rastreable: hash o commit del generador, seeds si aplica. Misma entrada
     + misma versión = misma salida.
   - **"Loggable"** = el artefacto privilegiado se persiste por episodio en
     `experiments/ENNN/...`, no "podríamos persistir si hiciera falta".
   - **Si la PI sale de un LLM externo** (ej. respuestas generadas por GPT-4,
     un Validator): fijar modelo + revision + params (temperature, top_p,
     max_tokens). Aceptar que perfecta determinismo puede depender de
     hardware/runtime; si eso rompe, esa fuente de PI no cumple este invariante.
6. **On-policy scoring estricto.** El **student** es quien genera la
   trayectoria / acciones. El **teacher** SOLO puntúa lo que el student hizo.
   Si en algún punto el teacher genera rollouts que se usan para entrenar al
   student, dejamos de hacer "reward por acción" y pasamos a distillation /
   behavioral cloning — otra hipótesis distinta. La trampa es fácil de colar
   sin querer (ej. "armemos rollouts del teacher para inicializar"); este
   invariante existe para que se vea explícito.
7. **Ablations claras > resultados maximales.** Más vale entender por qué
   gana o pierde que ganar por márgenes que no se explican. Los
   experimentos deben aislar el efecto del log-ratio teacher-privilegiado.
8. **Negative results se reportan honestamente.** No hay massaging de
   resultados. Si el método empata o pierde, eso es la conclusión.
9. **Documentación viva.** Toda decisión queda en GitHub Issues +
   `research/synthesis/`. Si una conclusión emerge en chat y no se
   persiste, no existe.
10. **La PI vive en el espacio de hechos, nunca en el de acciones.** PI
    válida = hechos sobre el **estado del mundo o el objetivo**, queryables
    del dataset/simulador **sin ejecutar política alguna** (producto target,
    atributos, ubicaciones de objetos). Dos exclusiones explícitas que
    cierran los loopholes conocidos:

    - **Demostraciones, trayectorias expertas y cualquier secuencia de
      acciones quedan excluidas aunque estén en el dataset.** Una traza
      experta es técnicamente "un hecho del dataset", pero usarla como PI
      es exactamente la degeneración a imitación que este invariante
      prohíbe. (Las trayectorias humanas de WebShop quedan confinadas a la
      ablation A de C.5, reportada como tal — nunca como PI del método.)
    - **La selección/orden de los hechos no puede ser canal de opinión.**
      Toda función que decide qué componentes ve el scorer y en qué orden
      depende solo de (dataset, estado del simulador, beliefs medidos bajo
      π_old versionado) — nunca de juicios de utilidad-para-la-próxima-acción.
      "Mejorar" el gating con heurísticas de relevancia para la acción
      siguiente es meter política por la puerta de atrás vía curation.

    Si la PI opina sobre acciones (directa o por curation), el reward
    degenera en imitación del opinador y el método colapsa a la familia
    juez-generativo/hints — otro paper. Extiende el invariante 5: además de
    reproducible y verificable, la PI es *fáctica*. (Origen: N.2,
    `pivot-2026-06.md` §4.1; cierres post-review externo 2026-06-12.)
11. **La PI influye en los pesos del student únicamente a través del
    escalar `r_t` en el policy gradient.** Nunca como texto en el contexto
    del student (training, eval o test), y **nunca como target de imitación,
    distillation o fine-tuning** — generar texto condicionado a la PI y
    destilarlo al student cumple la letra de "no está en su prompt" y viola
    todo; queda explícitamente prohibido. La PI aparece únicamente en los
    prompts del *scorer* (los forward passes que computan r_fwd y b_i), y
    de ahí solo sale un escalar por acción — pocos bits, que es
    precisamente lo que hace defendible el claim de no-contaminación.
    Consecuencias: no hay train/test mismatch del input, no hay nada que
    "retirar" del prompt con un schedule (lo que se apaga, vía el gating
    por belief, es la señal), y el invariante 3 queda subsumido y
    reforzado. Es el diferenciador estructural contra la familia hints
    (QuestA/KnowRL/...) y contra la rama distillation (OPSD/π-Distill/OPCD).
    (Origen: N.3, `pivot-2026-06.md` §2 y §7; cierre del lavado por
    imitación post-review externo 2026-06-12.)

## Jerarquía de decisión

Cuando objetivos compiten, priorizar en este orden:

1. **Validez del experimento** (no contaminar señal, no romper invariantes 3, 4, 5, 6, 10 y 11).
2. **Reproducibilidad** (manifest, commit SHA, seeds, artefactos privilegiados loggeados).
3. **Claridad de la story** (ablations que expliquen el resultado).
4. **Magnitud de la ganancia** (importa, pero no a costa de 1–3).
5. **Velocidad de iteración** (importante, pero no es el primer driver).

## Roadmap conceptual

| Fase | Estado | Objetivo |
|---|---|---|
| 0 — Bootstrap | ✅ Done | Estructura de docs + tracking + memoria. |
| 1 — Research | ✅ Done (2026-05-11) | Síntesis de los 7 vecinos originales (`papers-cross-mapping.md`, `design-decisions.md`). |
| 1b — Re-research pivot 2026-06 | 🟡 Now (vecinos consolidados 2026-06-12) | 6 vecinos nuevos (IGPO, TAMTRL, survey CA, GiGPO, KnowRL, menores) + decisiones N.1–N.13 + re-centrado en dual + residual. Epic [#19](https://github.com/lucaspecina/piar-rl/issues/19). Cierra con PROJECT.md actualizado (#22) y pre-registro de Figura 1 (#23). |
| 2 — Setup de compute | ⏳ Next | Azure ML Y-TEC (`lp-gpu-h100-x2-spot`, 2×H100 NVL). Spot → checkpointing no negociable. Sin cambios por el pivot — el pivot no requiere compute. |
| 3 — Replicación de baselines | ⏳ | iStar (B1, [#16](https://github.com/lucaspecina/piar-rl/issues/16)) **y GiGPO (B2, N.10 — trainer ya vendoreado en `code/examples/gigpo_trainer/`)** en WebShop, hyperparams de los papers. Sin baselines reproducidos las comparaciones son ruido. |
| 4 — Figura 1 (gate go/no-go) | ⏳ | Experimento día-1 sin RL: scoring forward/backward/dual sobre ~200 rollouts congelados en WebShop. Predicciones pre-registradas en [`figura1-prereg.md`](research/synthesis/figura1-prereg.md) ([#23](https://github.com/lucaspecina/piar-rl/issues/23)). Si pasan → luz verde a fase 5; cada escenario de fallo tiene salida definida. Requiere #17 (componentes g_i). |
| 5 — Implementación dual + residual | ⏳ | Modificación sobre `code/`: forward residual + belief tracker + gating + combinación en el advantage. El estimate ~150 LOC era para el forward solo — re-estimar (el backward + gating agregan superficie; `piar-implementation-points.md` necesita adendum). |
| 6 — Comparación + ablations | ⏳ | Brazos A0–A4 + B1/B2 + C1/C2 (`pivot-2026-06.md` §9) en **WebShop y ALFWorld (primera clase, N.11)**. Críticas: leakage bidireccional (D.1 + D.9), calibración de b_i (N.9), λ y normalización (N.5, N.7 — incl. ablation del backward sin z-norm por el caveat PBRS), thoughts en historial (N.6), gating duro vs soft vs observacional (N.4). Secuencia mínima publicable según presupuesto §9. |
| 7 — Redacción | ⏳ | Draft del paper + naming del método (N.12) + related work con las diferenciaciones obligatorias (HCAPO como vecino #1, TAMTRL ×5, IGPO, familia hints). |
| 8 — Caso de estudio SREG (opcional) | ⏳ | SCM como información privilegiada — la cereza, no el plato principal. |

Nota temporal: el "2–3 meses de calendar time" del roadmap anterior cubría
un solo método y un benchmark primario. Con dual + residual + dos benchmarks
+ dos baselines, **re-estimar al cerrar la Figura 1** — el presupuesto
honesto por brazo está en `pivot-2026-06.md` §9.
