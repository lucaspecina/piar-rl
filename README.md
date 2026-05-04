# PIAR — Privileged Implicit Action Reward

Research project sobre RL para LLM agents. Propone usar el log-ratio entre un teacher con golden answer en contexto y el student sin ella, sumado sobre el span de cada acción ReAct, como step reward — sin entrenar reward model ni etiquetar steps.

## Estado del proyecto

| Fase | Paradigma | Estado |
|---|---|---|
| 0 — Bootstrap | Estructura + tracking + memoria | ✅ Done |
| 1 — Research | Síntesis cruzada de papers vecinos | 🟡 Now |
| 2 — Decisión de framework | verl vs alternativas + Azure ML | ⏳ Next |
| 3 — Replicación de baseline | iStar baseline en WebShop | ⏳ |
| 4 — Implementación PIAR | Modificación quirúrgica (~50–200 líneas) | ⏳ |
| 5 — Comparación + ablations | PIAR vs baseline + ablations clave | ⏳ |
| 6 — 2do benchmark + redacción | ALFWorld/SOTOPIA + draft | ⏳ |
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
| Research consolidado | [`research/synthesis/`](research/synthesis/) · [`research/notes/`](research/notes/) |

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
├── experiments/         # Reproducibilidad (gitignored ENNN/*)
├── research/
│   ├── notes/           # Dumps de papers, debates
│   ├── synthesis/       # Conclusiones consolidadas
│   ├── examples/        # Ejemplos canónicos
│   └── archive/         # Notas obsoletas
└── .claude/skills/      # tracking, test, status
```

## Tracking

Source of truth: [GitHub Project v2 — PIAR Roadmap](https://github.com/users/lucaspecina/projects/5).
Issues + sub-issues nativas. Ver [`.claude/skills/tracking/`](.claude/skills/tracking/) para el workflow operativo.
