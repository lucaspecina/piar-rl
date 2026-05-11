# PIAR — Vision

> Norte filosófico. Define qué es PIAR, por qué existe, qué fuerza, e
> invariantes que no pueden romperse. **No describe implementación ni estado actual.**
>
> Implementación: `ARCHITECTURE.md` (cuando exista) · Estado actual: `CURRENT_STATE.md` · Trabajo pendiente: [GitHub Project v2](https://github.com/users/lucaspecina/projects/5).

## Misión

Investigar si **el log-ratio entre un teacher con información privilegiada
(golden answer / SCM) y un student sin ella, sumado sobre el span de cada
acción ReAct, sirve como step reward denso** para entrenar agentes RL
multi-turn — y caracterizar dónde funciona y por qué.

PIAR vive en la intersección de tres líneas existentes que no se cruzaron así
todavía: **implicit PRM** (Yuan 2024), **privileged-context teacher** (OPSD
2026) y **agentic multi-turn RL** (iStar 2025). El delta concreto es la
combinación, no inventar componentes nuevos.

## LA PREGUNTA

> **¿Podemos darle señal densa por acción a un agente RL multi-turn sin entrenar
> reward model ni etiquetar steps, usando solo el log-ratio entre un teacher con
> información privilegiada (golden answer / SCM) y el student — y si funciona,
> dónde y por qué?**
>
> Integra tres aristas que se evalúan juntas:
> 1. **Mecanismo:** ¿sirve el log-ratio teacher-privilegiado vs student como step reward?
> 2. **Problema:** ¿se puede tener señal densa por acción sin reward model ni step labels?
> 3. **Hipótesis fuerte:** ¿el contexto privilegiado reemplaza al entrenamiento del PRM/critic?
>
> Aplicala al evaluar, diseñar, priorizar o revisar cualquier decisión.

### Reformulación operativa contra iStar (2026-05-11)

Misma pregunta de fondo, planteada contra el baseline experimental concreto:

> **iStar codifica información privilegiada (outcome rankings) en los pesos
> de un juez separado vía DPO. ¿Esa info puede reemplazarse por información
> privilegiada en el contexto del mismo modelo (golden answer en el prompt),
> produciendo igual o mejor señal per-paso con un setup más simple (sin
> entrenar juez)?**

Es una **nearest-neighbor replacement study**, NO una "única variable changed" (caveat añadido tras review crítico de Codex, 2026-05-11). La asimetría se mueve de pesos a prompt, pero también cambia **qué información** está disponible (outcome rankings binarios vs golden answer rico) y **cuánta**. Los confounds están explícitos en el diseño experimental:

- **Cantidad/tipo de información privilegiada**: PIAR ve una golden answer estructurada; iStar solo ve "ganó/perdió" agregado a nivel trayectoria. Si PIAR gana puede ser por más info, no por prompt-asymmetry per se. Mitigación parcial: shuffled-golden control (D.9).
- **Mismos pesos del student (invariante 4)**: el setup primario usa **`π_old` + golden** vs **`π_old`** (misma snapshot, dos prompts), NO frozen θ₀. Esto resuelve la tensión: si el teacher fuera θ₀ y el student θ_t (con t > 0), ya no son "mismos pesos" y el log-ratio mezcla efecto-contexto con weight-drift. Ver [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md) C.2.
- **Hyperparams α/β**: tunear solo para PIAR sería unfair contra iStar; no tunear nada puede ser unfair contra PIAR. Solución: defaults de iStar (β=0.05, α=1) como punto de partida + pequeño grid compartido pre-registrado.
- **Leakage**: el control existencial es **shuffled-golden** (meter golden de otra tarea como sanity de que PIAR no mide solo answer-conditioned textual affinity). Ver D.9. Sin este control, ningún resultado positivo de PIAR es defendible.

**Lo que el experimento puede responder honestamente**:
- Si PIAR ≥ iStar **y** shuffled-golden ≤ outcome-only → la info privilegiada en prompt es alternativa viable al PRM entrenado.
- Si PIAR < iStar → el PRM entrenado captura algo (patrones temporales, regularización implícita) que el modelo con golden en prompt no extrae.
- Si shuffled-golden ≈ PIAR real → PIAR está midiendo answer-conditioned textual affinity, no causalidad. Resultado negativo limpio.

## Lo que PIAR quiere lograr

- **Densidad de señal sin overhead**: rewards por acción sin PRM entrenado ni step labels manuales — solo dos forward passes.
- **Diferenciación por contexto, no por parámetros**: el "teacher" no es un modelo entrenado distinto, es el mismo modelo con más información en el prompt. La asimetría vive en el contexto, no en los pesos.
- **Plug-and-play sobre RL existente**: la salida de PIAR es un escalar por acción que entra al advantage de GRPO/PPO standard. No requiere refactor del optimizer ni del rollout loop.
- **Validación cuantitativa contra baselines fuertes**: WebShop primero (estándar de agentic RL), después al menos un benchmark más para argumentar generalidad.
- **Story explicable de las ablations**: si gana, entender por qué; si pierde, entender qué supuesto del paper falla. Ambos resultados son publicables.

## Lo que NO es

- **No es un nuevo framework de RL.** Modificamos quirúrgicamente lo que ya existe (verl / iStar). El delta esperado: 50–200 líneas.
- **No es trabajo de math reasoning single-turn.** Eso lo hace OPSD. PIAR vive en agentes multi-turn ReAct.
- **No es un PRM entrenable.** El teacher no se entrena; se le pasa más contexto. Esa es la apuesta diferenciadora vs iStar.
- **No es distillation con teacher más fuerte.** Si el teacher es un modelo distinto (más grande, mejor entrenado, distilled, ensembled), aunque tenga golden en contexto, ya no es PIAR — es la línea de SWEET-RL / distillation guiada por privileged info. Ahí el log-ratio mezcla efecto-de-contexto con efecto-de-modelo y la conclusión científica se diluye.
- **No es construir un benchmark nuevo.** Usamos los existentes (WebShop, ALFWorld, SOTOPIA, τ-Bench). Eventualmente SREG como caso de estudio, pero no antes.
- **No es ajustar hyperparameters de un baseline.** Si la idea no gana por sí sola, las ablations explican por qué — no se rescata con HP search ad-hoc.

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
   - **Sub-decisión INCLINADA (2026-05-07):** **frozen al checkpoint inicial**
     como primer default, después de leer OPSD ([`research/notes/paper-opsd.md`](research/notes/paper-opsd.md) §3, §13.1; ver también
     [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md) C.2). OPSD usa frozen y lo justifica empíricamente
     ("helps stabilize training and implicitly acts as regularization to
     prevent excessive deviation from the initial policy"). La alternativa
     de **co-evolución** (estilo Skill-SD) sigue como ablation legítima dentro
     del invariante 4 ("mismos pesos / misma arquitectura / sin adapter exclusivo").
     **Cierre definitivo merece validación empírica en el régimen de PIAR**
     (multi-turn log-ratio), distinto al de OPSD (single-turn KL). Documentar
     el resultado de esa validación explícitamente cuando se haga.
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

## Jerarquía de decisión

Cuando objetivos compiten, priorizar en este orden:

1. **Validez del experimento** (no contaminar señal, no romper invariantes 3, 4, 5 y 6).
2. **Reproducibilidad** (manifest, commit SHA, seeds, artefactos privilegiados loggeados).
3. **Claridad de la story** (ablations que expliquen el resultado).
4. **Magnitud de la ganancia** (importa, pero no a costa de 1–3).
5. **Velocidad de iteración** (importante, pero no es el primer driver).

## Roadmap conceptual

| Fase | Estado | Objetivo |
|---|---|---|
| 0 — Bootstrap | ✅ Done | Estructura de docs + tracking + memoria. |
| 1 — Research | 🟡 Now | Síntesis cruzada de los 7 papers vecinos en `research/synthesis/`. Decisiones consolidadas en [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md). |
| 2 — Setup de compute | ⏳ Next | Stack ya decidido (prime-rl + verifiers, [#11](https://github.com/lucaspecina/piar-rl/issues/11)). Falta setup de Azure ML (Y-TEC) sobre las máquinas con A100/H100. |
| 3 — Replicación de baseline | ⏳ | Reproducir iStar baseline en WebShop (RLOO + iStar) con los hyperparams del paper. Código liberado ([`Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar)). |
| 4 — Implementación PIAR | ⏳ | Modificación quirúrgica sobre la base — estimación 80–250 líneas en Plan A (prime-rl), 30–100 en Plan B (veRL fork iStar) si se reabre. |
| 5 — Comparación + ablations | ⏳ | PIAR vs baseline + ablations clave: leakage vs progreso causal (D.1), forma del privileged context (C.5, E.2), frozen vs co-evolución (E.3), β scaling, length norm. |
| 6 — 2do benchmark + redacción | ⏳ | Extensión a ALFWorld/SOTOPIA + draft de paper. |
| 7 — Caso de estudio SREG (opcional) | ⏳ | SCM como información privilegiada — la cereza, no el plato principal. |

Plan tentativo de la fase de código (3 a 7): 2–3 meses de calendar time.
