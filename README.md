# PIAR — Privileged Implicit Action Reward

Research project sobre RL para LLM agents. Propone usar el log-ratio entre un teacher con golden answer en contexto y el student sin ella, sumado sobre el span de cada acción ReAct, como step reward — sin entrenar reward model ni etiquetar steps.

## Estado del proyecto

| Fase | Paradigma | Estado |
|---|---|---|
| 0 — Bootstrap | Estructura + tracking + memoria | ✅ Done |
| 1 — Research | Síntesis cruzada de papers vecinos | ✅ Done (2026-05-11) |
| 2 — Setup de compute | Stack decidido (fork CharacterRL-iStar vendoreado en [`code/`](code/)); falta Azure ML Y-TEC | ⏳ Next |
| 3 — Replicación de baseline | iStar baseline en WebShop (#16) | ⏳ |
| 4 — Implementación PIAR | Modificación quirúrgica de `code/` (~30–100 líneas) | ⏳ |
| 5 — Comparación + ablations | PIAR vs iStar + leakage controls D.1/D.9 (#15) | ⏳ |
| 6 — 2do benchmark + redacción | Sokoban (incluido en `code/`) + draft | ⏳ |
| 7 — Caso de estudio SREG | SCM como info privilegiada (opcional) | ⏳ |

## Cómo navegar este repo

| Si querés... | Andá a |
|---|---|
| Vision e invariantes | [`PROJECT.md`](PROJECT.md) |
| Qué corre HOY | [`CURRENT_STATE.md`](CURRENT_STATE.md) |
| Operativa Claude Code | [`CLAUDE.md`](CLAUDE.md) |
| Workflow operativo de tracking | [`.claude/skills/tracking/SKILL.md`](.claude/skills/tracking/SKILL.md) |
| Roadmap y trabajo pendiente | [Project v2 #5](https://github.com/users/lucaspecina/projects/5) · `gh issue list -R lucaspecina/piar-rl` |
| Historial de cambios | [`CHANGELOG.md`](CHANGELOG.md) |
| Decisiones de diseño | [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md) |
| Delta deep-dive vs π-Distill α=0 (vecino más cercano) | [`research/synthesis/piar-delta.md`](research/synthesis/piar-delta.md) |
| Mapeo de implementación PIAR sobre `code/` (fase 4 prep) | [`research/synthesis/piar-implementation-points.md`](research/synthesis/piar-implementation-points.md) |
| Notas de papers | [`research/notes/`](research/notes/) |
| Research consolidado | [`research/synthesis/`](research/synthesis/) |

## Setup

```bash
git clone https://github.com/lucaspecina/piar-rl.git
cd piar-rl
# Por ahora no hay deps — el proyecto está en fase research/papers (sin código).
# Ver CURRENT_STATE.md para el detalle de qué existe hoy.
```

## Estructura

```
piar-rl/
├── README.md            # Este archivo
├── PROJECT.md           # Vision, LA PREGUNTA, invariantes
├── CLAUDE.md            # Operativa Claude Code
├── CURRENT_STATE.md     # Qué corre hoy
├── CHANGELOG.md         # Historial con refs #N
├── AUTORESEARCH.md      # Config autoresearch (OFF)
├── code/                # Fork vendoreado de CharacterRL-iStar (Apache-2.0).
│                        # Base de implementación de PIAR. Ver code/NOTICE.md.
├── experiments/         # Reproducibilidad (gitignored ENNN/*)
├── research/
│   ├── notes/           # Dumps de papers, debates
│   ├── synthesis/       # Conclusiones consolidadas
│   ├── examples/        # Ejemplos canónicos
│   └── archive/         # Notas obsoletas
└── .claude/skills/      # tracking, test, status
```

## Sobre `code/`

`code/` contiene una copia vendoreada de `CharacterRL-iStar/` (Tongyi-ConvAI, Apache-2.0, ICLR 2026). Es la base sobre la que se implementará PIAR — un fork directo del paper de iStar modificado quirúrgicamente para reemplazar el PRM aprendido por el mismo modelo con golden answer en el prompt.

Detalle de procedencia, licencia y modificaciones esperadas en [`code/NOTICE.md`](code/NOTICE.md).

## Tracking

Source of truth: [GitHub Project v2 — PIAR Roadmap](https://github.com/users/lucaspecina/projects/5).
Issues + sub-issues nativas. Ver [`.claude/skills/tracking/`](.claude/skills/tracking/) para el workflow operativo.
