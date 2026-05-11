# Changelog

Historial de cambios del proyecto. Cada entrada referencia el issue de GitHub
que la motivó (formato `#N`).

## 2026-05-11

- **Reformulación operativa de LA PREGUNTA contra iStar** en `PROJECT.md`. Mantiene LA PREGUNTA abstracta original y agrega sub-sección concreta: "¿La info privilegiada de iStar (outcome rankings codificados en los pesos del juez vía DPO) puede reemplazarse por info privilegiada en el contexto del mismo modelo (golden answer en el prompt), produciendo igual o mejor señal per-paso con un setup más simple?". Define el experimento primario PIAR vs iStar en WebShop con setup intacto, cambiando solo la fuente del término privilegiado del log-ratio. (Refs #2, #4)
- **Documentación amigable de iStar** en [`research/notes/paper-istar.md`](research/notes/paper-istar.md) §15: el método sin fórmulas (3 copias del SFT base, ejemplo del laberinto, DPO ≠ SFT, cómo emerge el score por acción, truco matemático de Yuan). Para futuras sesiones que necesiten releer iStar sin volver a descifrar la matemática. (Refs #4)
- **Dudas conceptuales y debilidades teóricas de iStar** documentadas en `paper-istar.md` §16: (1) el juez no es un jugador competente, solo provee señal per-paso; (2) la info del juez es re-codificación de outcomes, no info nueva; (3) los propios autores reconocen que separar PRM del policy es opcional, no necesario; (4) el teorema de Yuan tiene supuestos no blindados empíricamente. Implicaciones para PIAR: la apuesta es más fuerte que lo que iStar reconoce como abierto — además de unificar PRM+policy, reemplaza el entrenamiento por info en el prompt. Tres riesgos específicos enumerados (leakage textual, info demasiado densa, dependencia del golden definido). (Refs #4)
- **CLAUDE.md actualizado**: puntero desde "LA PREGUNTA" hacia la reformulación operativa en `PROJECT.md` y hacia `paper-istar.md` §15-§16.
- **Repo `CharacterRL-iStar` verificado operacionalmente**: estructura completa con 7 trainers ejecutables (iStar, RLOO, GRPO, REINFORCE++, PPO, GiGPO, PRIME) en WebShop + Sokoban. Modelo base Qwen2.5-7B-Instruct. Hardware target 8×H100/A100 (compatible con Y-TEC). Framework veRL (fork de Alibaba). Hyperparams loggeados en los scripts. Refuerza viabilidad del Plan B (#11) si se reabre la decisión de framework.

### Parte 2 — post review crítico de Codex

- **Tightening de la "Reformulación operativa" en `PROJECT.md`**: Codex pegó duro contra la claim de "única variable changed". Reescrita honestamente como **nearest-neighbor replacement study** con los confounds nombrados explícitamente (cantidad/tipo de info disponible, weight drift si frozen, hyperparams α/β). Agregada mención al shuffled-golden control (D.9) como sanity de leakage. (Refs #2, #4)
- **Actualizada decisión C.2 en `design-decisions.md`**: snapshot del teacher cambia de "frozen θ₀" (decisión 2026-05-07 estilo OPSD) a **`π_old` (snapshot reciente) como primario; frozen θ₀ como ablation de estabilidad**. Razón: frozen θ₀ rompe "mismos pesos del student" (invariante 4) después del primer update (θ_0 ≠ θ_t) y mezcla efecto-contexto con weight-drift. Con `π_old` en ambos términos del log-ratio, queda **pura diferencia de prompt** como única asimetría. OPSD seguía siendo válido en single-turn KL pero la transferencia a multi-turn log-ratio no preserva el invariante. (Refs #5)
- **Nueva decisión D.9 en `design-decisions.md`**: **shuffled-golden control como sanity de leakage existencial** (no opcional). Correr PIAR con goldens shuffleadas (de otras tareas) — si el reward o el training mejora vs outcome-only con shuffled goldens, PIAR mide answer-conditioned textual affinity, no progreso causal. Sin este control, ningún resultado positivo es defendible. Complementa a D.1 (que mide leakage a nivel muestreo pareado). (Refs #2)
- **Actualizada E.3** para reflejar el cambio en C.2: la pregunta abierta pasa de "frozen vs co-evolución" a "`π_old` vs frozen θ₀".
- **CURRENT_STATE.md refrescado**: lista de papers cerrados actualizada (eran 2 visibles como cerrados, ahora son 7 — los 6 restantes del epic). Mención al reframe post-iStar y a la verificación del repo `CharacterRL-iStar`.
- **GitHub Issues restructurados post-review**:
  - Cerrado #4 (iStar) como `completed`. Trabajo hecho, bug de tracking corregido (CHANGELOG ya lo daba cerrado, GitHub seguía abierto).
  - Cerrado #12 (contactar autores iStar) como `completed`. Código liberado al público, criterio de cierre alcanzado sin necesidad de contacto.
  - Retitulado #13 a "Plan A POC — log-ratio teacher-student como rubric de verifiers (step-level reward)" para clarificar que depende de la decisión Plan A vs Plan B (queda parked).
  - Abierto **#14** (design): "Decidir Plan A (prime-rl) vs Plan B (fork CharacterRL-iStar) post-reformulación 2026-05-11".
  - Abierto **#15** (research): "Leakage vs progreso causal — D.1 (muestreo pareado) + D.9 (shuffled-golden control)".
  - Abierto **#16** (research, blocked): "Fase 3 — Replicar baseline iStar WebShop con CharacterRL-iStar". Bloqueado por #14 y setup de Azure ML.
  - Todos agregados al Project v2 board en `Todo`.
- **Codex MCP validado como segunda opinión técnica**: encontró confounds reales en la reformulación de hoy (la oversold-única-variable + la tensión C.2/invariante 4 + la falta de shuffled-golden). Todas las correcciones aplicadas.

## 2026-05-07

- **Research — Yuan 2024 (Implicit PRM) consolidado** en [`research/notes/paper-yuan-implicit-prm.md`](research/notes/paper-yuan-implicit-prm.md). Insight clave: Proposition 3.1 es identidad algebraica de soft-Q (vale para cualquier par de distribuciones autoregresivas, no requiere training). Aplicada a PIAR, $r_{\text{PIAR}}$ es soft-Q válida del reward "cuánto racionaliza el contexto golden esta trayectoria"; el gap con un PRM clásico es **semántico**, no matemático. (#3 cerrado)
- **Research — iStar (ICLR 2026) consolidado** en [`research/notes/paper-istar.md`](research/notes/paper-istar.md). Insights clave: (1) **action-level + advantage-level normalization** es el setup ganador (94.7 vs 90.0 token-level vs 90.7 reward-level). (2) **β = 0.05** y $\alpha = 1$ default. (3) **Sin KL penalty.** (4) **Código liberado en [`Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar)** — reabre Plan B (PIAR sobre veRL fork de iStar) como opción concreta vs Plan A (prime-rl). (5) iStar **no considera privileged-context teachers** — PIAR está en territorio que iStar no exploró. (#4 cerrado)
- **Research — OPSD (Self-Distilled Reasoner) consolidado** en [`research/notes/paper-opsd.md`](research/notes/paper-opsd.md). Insights clave: (1) Privileged-context teacher con same-model funciona en single-turn math (+5.7% vs GRPO en Qwen3-1.7B). (2) **Teacher frozen al checkpoint inicial** justificado empíricamente como regularizador — inclina la sub-decisión del invariante 4 hacia frozen como primer default. (3) Template del prompt del teacher reusable (Figura 2). (4) **OPSD no analiza leakage** → gap real de literatura que PIAR debe cubrir. (5) Forward KL token-level en OPSD vs log-ratio span-level en PIAR — diferencia operativa importante. (#5 cerrado pendiente)
- **Creado [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md)** — índice central de decisiones de diseño con trazabilidad a papers, issues y status (cerrada / inclinada / abierta / parked). Incluye 8 decisiones inclinadas o cerradas + 6 preguntas abiertas con criterio de resolución.
- **Actualizado PROJECT.md**: invariante 4 sub-decisión inclinada hacia frozen (con referencia a OPSD); roadmap fase 2 ajustado (stack ya decidido, foco en Azure ML setup).
- **Actualizado README.md**: tabla de fases consistente con stack decidido. Nueva entrada en navegación apuntando a design-decisions.md.
- **Actualizado CLAUDE.md**: trigger table extendida con regla de mantenimiento de design-decisions.md cuando hay decisión nueva o cambio de status.
- **Codex MCP validado** como segunda opinión técnica en review de notas. Encontró bugs reales en cada paper (framing algebraic-vs-semantic de Yuan, errores en Δ de SOTOPIA, framing del invariante 4 con iStar, "mismos pesos" en OPSD ignorando que después del step 0 son θ_0 vs θ_t). Todas las correcciones aplicadas antes de cerrar.

## 2026-05-04

- Bootstrap del proyecto con tracking GitHub-based: repo `lucaspecina/piar-rl`,
  Project v2 #5 ("PIAR Roadmap") con campo `Status`, docs raíz (README,
  PROJECT, CURRENT_STATE, CLAUDE, CHANGELOG, AUTORESEARCH), skills
  (`tracking/`, `test/`, `status/`), estructura `research/{notes,synthesis,examples,archive}/`,
  memoria del proyecto inicial. (#1)
- Decisión de stack: **prime-rl + verifiers** (Prime Intellect) como base de implementación. Plan B documentado: veRL. Plan C parked: contactar autores iStar. (#11)
