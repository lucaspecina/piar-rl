# PIAR — Claude Code Configuration

## START HERE — Read these docs first

1. **README.md** — Entry point. Navegación + setup rápido.
2. **PROJECT.md** — Vision, LA PREGUNTA, invariantes, jerarquía de decisión.
3. **CURRENT_STATE.md** — Qué corre HOY, qué se está construyendo.
4. **`.claude/skills/tracking/SKILL.md`** — Workflow operativo de issues + Project v2.
5. **CHANGELOG.md** — Historial con refs `#N` a issues de GitHub.
6. **`research/synthesis/`** — Conclusiones consolidadas (cuando existan).
7. **ARCHITECTURE.md** — (Deferred. Crear cuando haya 3+ módulos con contratos.)

## LA PREGUNTA

> **¿Podemos darle señal densa por acción a un agente RL multi-turn sin entrenar
> reward model ni etiquetar steps, usando solo el log-ratio entre un teacher con
> información privilegiada (golden answer / SCM) y el student — y si funciona,
> dónde y por qué?**
>
> Aplicala al evaluar, diseñar, priorizar o revisar cualquier decisión.

> **Reformulación operativa contra iStar (2026-05-11):** versión concreta del
> experimento primario en [`PROJECT.md`](PROJECT.md). Aísla "fuente de la
> asimetría" (pesos del juez entrenado en iStar vs golden en el prompt en PIAR)
> como única variable. Ver también [`research/notes/paper-istar.md`](research/notes/paper-istar.md) §15-§16 para la explicación amigable y las dudas teóricas.

## Where to find what

| Necesito... | Ir a... |
|---|---|
| Vision e invariantes | `PROJECT.md` |
| Estado HOY del sistema | `CURRENT_STATE.md` |
| **Decisiones de diseño + trazabilidad** | [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md) |
| Roadmap / trabajo pendiente | [Project v2 #5](https://github.com/users/lucaspecina/projects/5) · `gh issue list -R lucaspecina/piar-rl` |
| Workflow de tracking (cómo crear/cerrar issues, sub-issues, etc.) | `.claude/skills/tracking/SKILL.md` + `commands.md` + `reference.md` |
| Historial de cambios | `CHANGELOG.md` |
| Trabajo en curso de un paper | Issue activo en GitHub + `research/notes/paper-<slug>.md` |
| Conclusiones consolidadas | `research/synthesis/` |
| Ejemplos canónicos worked-out | `research/examples/` |
| Cómo se trabaja acá | Este archivo |

## Project overview

Research project sobre RL para LLM agents. Propone usar el log-ratio de logprobs
entre un teacher con golden answer en contexto (privileged-context) y el student
sin ella, sumado sobre el span de cada acción ReAct, como step reward. Cruza tres
líneas existentes: implicit PRM (Yuan), privileged-context teacher (OPSD) y
agentic multi-turn RL (iStar). Fase actual: **research/papers, sin código**.

## Environment setup

Por ahora no hay código → no hay environment de runtime. Cuando arranque la fase
de implementación, el plan tentativo es:

- **Dev local:** Windows + conda/uv (TBD).
- **Compute pesado:** Azure ML (Y-TEC) con A100 / H100. Ver skill
  `azure-ml-connect` (user-level) cuando llegue el momento.
- **Frameworks objetivo:** PyTorch, verl (base de iStar), vLLM, transformers.

## Tech stack

Decisión operativa al 2026-05-04 (ver análisis completo en
[`research/notes/repos-mapping.md`](research/notes/repos-mapping.md) y trazabilidad en [#11](https://github.com/lucaspecina/piar-rl/issues/11)):

- **[`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl)** (Prime Intellect, Apache-2.0) — framework de RL agentic asincrónico. FSDP2 + vLLM + multi-turn nativo. Heredado de SREG.
- **[`verifiers`](https://github.com/PrimeIntellect-ai/verifiers)** (Prime Intellect, MIT) — librería de environments + rubrics. Define el harness, dataset y reward function.
- **PyTorch** — base obvia.
- **vLLM** — sampling rápido para rollouts (integrado en prime-rl).

**Plan B documentado:** `verl` (ByteDance) si por alguna razón prime-rl no encaja
para PIAR. **Plan C parked:** contactar autores iStar para acceso temprano al
código (su repo no está liberado al 2026-05).

## Project structure

```
piar-rl/
├── README.md            # Entry point + navegación
├── PROJECT.md           # Vision, LA PREGUNTA, invariantes
├── CLAUDE.md            # Este archivo
├── CURRENT_STATE.md     # Qué corre hoy
├── CHANGELOG.md         # Historial de cambios con refs #N
├── AUTORESEARCH.md      # Config de autoresearch (OFF por defecto)
├── experiments/         # Reproducibilidad — ENNN/manifest.yaml (gitignored ENNN/*)
├── research/
│   ├── notes/           # Dumps de papers, debates (efímeros)
│   ├── synthesis/       # Conclusiones consolidadas (canon)
│   ├── examples/        # Ejemplos canónicos worked-out
│   └── archive/         # Notas obsoletas
└── .claude/skills/
    ├── tracking/        # Workflow Project v2 — SKILL, commands, reference
    ├── test/            # Correr tests (placeholder hasta haya código)
    └── status/          # Overview rápido del estado del proyecto
```

## Code conventions

- **Comunicación: español siempre** (a menos que el usuario cambie a inglés).
- Cuando llegue el código: **type hints, docstrings cortos en castellano,
  ruff/black**. Definir cuando se concrete.
- **NO crear ARCHITECTURE.md** hasta que haya 3+ módulos reales con contratos
  entre ellos.

## Commands

Aún no aplica (sin código). Cuando arranque dev:
- `/test` — correr tests del proyecto.
- `/status` — overview del estado actual (board + commits).
- `/tracking` — workflow para issues / Project v2.

## Quality assurance

Tres niveles (ver user-level `dev-workflow/quality-levels.md`):
- **Nivel 1** — pre-commit: tests + lint (cuando haya código).
- **Nivel 2** — system validation: replicar baseline iStar end-to-end en WebShop.
- **Nivel 3** — external validation: márgenes claros vs baseline en benchmarks
  estándar + ablations que expliquen la ganancia.

## Issue tracking — GitHub Project v2

**Source of truth = [PIAR Roadmap (Project v2 #5)](https://github.com/users/lucaspecina/projects/5).**
Toda issue abierta tiene `Status` (Todo / In Progress / Done) en el board.

Workflow operativo completo en `.claude/skills/tracking/`:
- `SKILL.md` — modelo mental (Epic / Issue), reglas, templates de body, labels.
- `commands.md` — recipes exactos por situación (crear, empezar, cerrar, linkear sub-issues, promover a epic).
- `reference.md` — Project ID, field IDs, option IDs, GraphQL templates.

**Resumen de lo crítico:**
- **1 issue concreta = 1 entregable** (1 PR cuando hay código, o 1 doc en `research/` en fase research).
- **Sub-issues vía API nativa de GitHub** (NO "Part of #N" en body).
- **Status `In Progress` se mueve AL EMPEZAR**, no al final.
- **Labels acotados** a 5: `bug`, `blocked`, `parked`, `research`, `design`. Sin `area:*` ni `prio:*`.
- **Template de body obligatorio**: Contexto / Detalle técnico / Criterio de cierre.
- **Razones de cierre**: `completed` (default) vs `not planned` (descartado, remover del board).
- **Cross-link** issues ↔ `research/notes/` y `research/synthesis/`.

> **Nota sobre Worktree**: el campo `Worktree` del Project v2 NO está creado
> todavía. Se agregará cuando arranque la fase de código y haya sesiones
> paralelas. Ver `tracking/commands.md` recipe #10.

## Epics activos

| # | Epic | Status | Sub-issues | Notas |
|---|------|--------|------------|-------|
| ~~[#2](https://github.com/lucaspecina/piar-rl/issues/2)~~ | ~~Research — síntesis de papers vecinos~~ | ✅ Done (2026-05-11) | #3, #4, #5, #6, #7, #8, #9, #10 todos cerrados | Cierre del epic: ver `research/synthesis/papers-cross-mapping.md` + `design-decisions.md`. |

> Esta tabla se mantiene sincronizada con el Project v2. Si se crea/cierra un
> epic o cambia el criterio: actualizar acá en el mismo commit.
>
> **Estado al 2026-05-11**: sin epics activos. Próximo posible epic = "Fase 4 — Implementar PIAR sobre Plan B" cuando arranque la implementación.

## Commit workflow — MANDATORIO

1. **VALIDATE** — tests + lint (cuando aplique).
2. **PRESENT** — explicar en español qué cambió y por qué. **Esperar aprobación.**
3. **DOCS** — actualizar todos los docs afectados (ver trigger table abajo).
4. **COMMIT** — con `Co-Authored-By: Claude Opus 4.7 (1M context)` y refs `#N`
   (issues de GitHub) en el mensaje.

**Regla de oro: NUNCA commitear sin aprobación explícita del usuario, salvo en
modo autoresearch.** Esto es no-negociable.

Convenciones de commit:
- `feat: ...`, `fix: ...`, `docs: ...`, `chore: ...`, `research: ...`.
- Refs: `Refs #N <descripción>` (no cierra) o `Closes #N` (cierra al merge).

## Document maintenance — trigger table

Después de cada cambio significativo, escanear esta tabla:

| Qué cambió | Qué actualizar |
|---|---|
| Trabajé en un issue | Mover Status a `In Progress`, comentar al cerrar |
| Cerré un issue | Comentario final con link a artefacto. Status → `Done`. CHANGELOG si tocó código. |
| Issue nuevo | Crear en GitHub + agregar al Project v2 + linkear como sub-issue si aplica |
| Corrí un experimento | `experiments/ENNN/manifest.yaml` + comentario en issue |
| Agregué/saqué archivo | CLAUDE.md project structure + CURRENT_STATE.md |
| Cambié API o función | ARCHITECTURE.md si existe + CURRENT_STATE.md |
| Agregué dependencia | `pyproject.toml` (cuando exista) + CLAUDE.md tech stack |
| Cambié convención | CLAUDE.md inmediatamente |
| Cambió scope/visión | PROJECT.md primero, propagar a CLAUDE.md y al Project v2 |
| Research deep done | `research/notes/paper-<slug>.md` + comentario en el issue |
| Conclusión de research | `research/synthesis/` + cerrar issue con link |
| Research → decisión | Agregar / actualizar fila en `research/synthesis/design-decisions.md`. Si toca un invariante → propagar a `PROJECT.md`. |
| Decisión cambia de status (abierta → inclinada → cerrada, o viceversa) | Updatear fila en `research/synthesis/design-decisions.md` con fecha + fuente. CHANGELOG con ref `#N` si fue disparada por cierre de issue. |
| Creé/cerré epic | Actualizar tabla "Epics activos" arriba |

Ver user-level `dev-workflow/doc-maintenance.md` para reglas completas y cleanup.

## Cleanup y mantenimiento

- "Actualizar" significa el ECOSISTEMA completo: docs, skills, memorias, scripts, configs.
- Si un cambio dejó código/tests/scripts obsoletos → **borrarlos** (git tiene historia).
- Si un doc referencia algo que ya no existe → **arreglar la referencia**.
- Al cerrar issues: evaluar si `research/notes/` puede pasar a `research/archive/`
  cuando la conclusión ya vive en `research/synthesis/` o en `PROJECT.md`.

## Autoresearch

- Config en `AUTORESEARCH.md` (ON / OFF + run config).
- Branch: `autoresearch/<topic>-<fecha>` desde branch base explícita.
- Stop conditions OBLIGATORIAS antes de activar.
- Comentarios en los issues = memoria persistente que sobrevive compactación.
- Ver user-level `dev-workflow/autoresearch.md` para protocolo completo.

## Codex collaboration

Si `mcp__codex__codex` está disponible:
- **MANDATORIO:** code review post-implementación, antes de presentar al usuario.
- **RECOMENDADO:** strategy pre-implementación en tareas complejas.
- **SKIP:** doc-only, fixes triviales, cuando el usuario lo dice.
- Pedirle SHORT y CRITICAL. Bugs reales se arreglan, nits = deuda.
- Ver user-level skill `codex-collab` y `codex-base-instructions.md`.

## Git conventions

- Branch principal: `main`.
- Branch de feature/issue: `issue/<N>-<slug-corto>` (ej. `issue/3-yuan-implicit-prm`).
- Branch de research autónomo: `autoresearch/<topic>-<fecha>`.
- Commits con ref a issues: `Mensaje del cambio (Refs #N)` o `Closes #N` en PR body.
- Co-Authored-By al final de commits con asistencia de Claude.

## Comunicación

- **Idioma:** español (siempre, salvo que el usuario cambie a inglés).
- **Tono:** directo, técnico, accesible. Sin filler.
- **Formato:** explicaciones cortas, bullets, tablas. Después de un commit,
  sugerir 1-3 próximos pasos. Mantener momentum.
