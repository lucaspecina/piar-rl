# PIAR — Privileged Implicit Action Reward

Research project sobre RL para LLM agents. **Post-pivot 2026-06**: propone un step reward bidireccional derivado solo de información privilegiada fáctica y del propio modelo — **forward pragmático** (Δ logprob de la acción con PI en el prompt del scorer) + **backward epistémico** (Δ belief sobre los componentes de la PI) — con **PI residual** gateada por el belief medido del student (auto-annealing sin schedule). Sin reward model entrenado, sin juez generativo, sin step labels. El claim: la descomposición, la dosificación y el mapa dónde/por qué — los componentes sueltos ya están publicados (TAMTRL, IGPO); el cruce no.

## Estado del proyecto

| Fase | Paradigma | Estado |
|---|---|---|
| 0 — Bootstrap | Estructura + tracking + memoria | ✅ Done |
| 1 — Research | Síntesis de los 7 vecinos originales | ✅ Done (2026-05-11) |
| 1b — Re-research pivot 2026-06 | 6 vecinos nuevos + re-centrado dual+residual ([pivot doc](research/synthesis/pivot-2026-06.md), epic #19) | 🟡 Now |
| 2 — Setup de compute | Stack decidido (fork CharacterRL-iStar vendoreado en [`code/`](code/)); falta Azure ML Y-TEC | ⏳ Next |
| 3 — Replicación de baselines | iStar (B1, #16) **y GiGPO (B2)** en WebShop | ⏳ |
| 4 — Figura 1 (gate go/no-go) | Scoring sobre rollouts congelados, [pre-registro congelado](research/synthesis/figura1-prereg.md) (#23) | ⏳ |
| 5 — Implementación dual + residual | Modificación sobre `code/` (re-estimar LOC) | ⏳ |
| 6 — Comparación + ablations | Brazos A0–A4 + B1/B2 + C1/C2 en WebShop y ALFWorld; leakage bidireccional (#15) | ⏳ |
| 7 — Redacción | Draft + naming + related work (HCAPO vecino #1) | ⏳ |
| 8 — Caso de estudio SREG | SCM como info privilegiada (opcional) | ⏳ |

## Cómo navegar este repo

| Si querés... | Andá a |
|---|---|
| Vision e invariantes (post-pivot, 11 invariantes) | [`PROJECT.md`](PROJECT.md) |
| **El pivot 2026-06 (formalización del método)** | [`research/synthesis/pivot-2026-06.md`](research/synthesis/pivot-2026-06.md) |
| Qué corre HOY | [`CURRENT_STATE.md`](CURRENT_STATE.md) |
| Operativa Claude Code | [`CLAUDE.md`](CLAUDE.md) |
| Workflow operativo de tracking | [`.claude/skills/tracking/SKILL.md`](.claude/skills/tracking/SKILL.md) |
| Roadmap y trabajo pendiente | [Project v2 #5](https://github.com/users/lucaspecina/projects/5) · `gh issue list -R lucaspecina/piar-rl` |
| Historial de cambios | [`CHANGELOG.md`](CHANGELOG.md) |
| Decisiones de diseño | [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md) |
| Delta deep-dive vs π-Distill α=0 (histórico pre-pivot) | [`research/synthesis/piar-delta.md`](research/synthesis/piar-delta.md) |
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

`code/` contiene una copia vendoreada de `CharacterRL-iStar/` (Tongyi-ConvAI, Apache-2.0, ICLR 2026). Es la base sobre la que se implementará PIAR: el fork trae los trainers de iStar (baseline B1) **y de GiGPO (baseline B2)** listos, y el reward dual+residual se inyectará en el mismo hook del advantage donde iStar mete su step reward (sin tocar `code/` hasta fase 3 — invariante 2).

Detalle de procedencia, licencia y modificaciones esperadas en [`code/NOTICE.md`](code/NOTICE.md).

## Tracking

Source of truth: [GitHub Project v2 — PIAR Roadmap](https://github.com/users/lucaspecina/projects/5).
Issues + sub-issues nativas. Ver [`.claude/skills/tracking/`](.claude/skills/tracking/) para el workflow operativo.
