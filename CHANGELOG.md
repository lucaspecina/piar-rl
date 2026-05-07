# Changelog

Historial de cambios del proyecto. Cada entrada referencia el issue de GitHub
que la motivó (formato `#N`).

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
