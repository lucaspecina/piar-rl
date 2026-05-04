---
name: tracking
description: USE WHENEVER creating, editing, closing, labeling, linking, or organizing GitHub issues, epics, sub-issues, or Project v2 board fields (Status). ALSO when asked about epics, roadmap, priorities, or "what's next". Covers issue templates, sub-issue linking via native API, Project v2 Status updates (GraphQL), epic promotion, close reasons, and concurrent session coordination. This project's tracking source of truth is GitHub Project v2 — not raw Issues.
---

# PIAR tracking workflow

**SOURCE OF TRUTH = GitHub Project v2 "PIAR Roadmap"**
https://github.com/users/lucaspecina/projects/5

Every open issue MUST appear on the board with `Status` set. If you touch an issue and don't sync the board, the board breaks and other sessions lose visibility.

For exact commands: see `commands.md`. For IDs and GraphQL templates: see `reference.md`.

## Modelo mental (los 2 conceptos)

| Concepto | Qué es | Vida útil |
|---|---|---|
| **Epic** | Meta concreta con criterio de cierre. No es label ni tema paraguas. Agrupa sub-issues. | Semanas/meses |
| **Issue** | Sub-issue de un epic (puede tener hijos) o issue concreta (1 PR / 1 task). | Días |

Reglas clave:
- **Anidación permitida dentro de un epic.** Estructura típica:
  ```
  Epic: Research — síntesis de papers vecinos (#2)
    -> Yuan et al. 2024 — Implicit PRM (#3)        <-- sub-issue (1 task)
    -> iStar (#4)                                  <-- sub-issue (1 task)
    -> OPSD (#5)                                   <-- sub-issue (1 task)
    ...
  ```
  GitHub soporta anidación de sub-issues sin límite.
- **La hoja (la que cierra un PR o produce el artefacto) sigue siendo "1 issue = 1 PR / 1 entregable"**. Lo que tiene hijos es un agrupador conceptual.
- **Sub-issues vía API nativa** de GitHub (NO "Part of #N" en body). Aplica a cualquier nivel.
- **Un issue puede ser standalone** (sin epic padre) para one-offs: bugs, docs, research parkeado, fixes.
- **Cuándo splittear trabajo de un epic a otro epic separado**: cuando el trabajo es **ortogonal** al epic actual (toca otro componente o fase) y tiene su propio criterio de cierre.

> **Nota sobre Worktree**: este proyecto no usa todavía el campo `Worktree` del Project v2. Cuando arranque la fase de código y haya múltiples sesiones en paralelo, se puede agregar (ver `commands.md` → "Agregar campo Worktree"). Por ahora, sesiones lineales en branches.

## Arranque de sesión (primer check cuando la tarea toca tracking)

```bash
# 1. Dónde estoy
pwd && git branch --show-current

# 2. Qué hay activo en el board (Status por item)
# Ver commands.md -> "query del board"

# 3. Qué epic / issue corresponde a la tarea actual
# Ver tabla "Epics activos" en CLAUDE.md + gh api /repos/lucaspecina/piar-rl/issues/<EPIC>/sub_issues
```

## El campo obligatorio del board

| Campo | Valores | Cuándo cambia |
|---|---|---|
| **Status** | `Todo` / `In Progress` / `Done` | Crear issue → `Todo`. Empezar trabajo → `In Progress` (mover AL EMPEZAR, no al final). Mergear PR / cerrar issue → `Done` (auto). |

**Prioridad** = orden manual en la columna `Todo` (drag & drop o reorder vía API). **NO es label.**

## Template de body (obligatorio para epic o issue)

```markdown
## Contexto (para humanos)

<1-3 frases en español: qué es, por qué importa, cuándo lo harías>

## Detalle técnico (para Claude / sesiones)

<jerga, refs a código / paper / arxiv ID, decisiones, edge cases, links a research/>

## Criterio de cierre

<qué tiene que pasar para estar hecho>
```

## Convenciones de título

- **Epic**: `Epic · <meta>` (ej: `Epic · Research — síntesis de papers vecinos`).
  - Cuando arranque la fase de código y haya worktrees, el formato volverá a ser `Epic · <worktree> · <meta>`.
- **Sub-issue / standalone**: descriptivo, sin prefijo, < 70 chars.

## Labels (5 — no crear nuevos sin consultar)

| Label | Cuándo |
|---|---|
| `bug` | Bug real |
| `blocked` | Esperando dependencia (comentar qué bloquea) |
| `parked` | Idea abierta pero no activa |
| `research` | Análisis o síntesis, no produce código |
| `design` | Requiere diseño/decisión antes de codear |

Sin `area:*` ni `prio:*` (orden en Todo cubre prioridad).

## Crear un epic (reactivo, no predictivo)

Crear epic SOLO si hay 3+ sub-issues concretos + semanas de trabajo + criterio de cierre claro.

**Reactivo, no predictivo**: empezar con issues sueltos; promover a epic cuando emerge el patrón (ver commands.md → "Promover sub-issue a epic").

**Cuándo agrupar sub-issues dentro de un epic (nivel intermedio)**: si dentro de un epic se ven varios sub-issues que comparten un objetivo común pero no lo suficiente para otro epic separado, crearlos como sub-issues-con-hijos. GitHub soporta el anidamiento nativo.

## Razones de cierre — Completed vs Not planned

- `gh issue close <N> --reason completed` — Se hizo. Va a la columna `Done`. **Usar esto para la mayoría.**
- `gh issue close <N> --reason "not planned"` — No se va a hacer (scope change, duplicado, replanteo). Queda cerrado pero NO aparece en Done. **Remover del board** con GraphQL `deleteProjectV2Item` para que no polucione.

## Comentarios en issues (obligatorios)

- **Al cerrar**: párrafo explicando qué se hizo + link al PR mergeado (cuando aplique) + link a `research/synthesis/` cuando produjo conclusión.
- **Al bloquear**: qué lo bloquea y qué lo destrabaría.
- **Scope change**: si cambia alcance, documentar decisión y por qué.
- **Hand-off entre sesiones**: estado actual + qué falta.

NO comentar para updates triviales ni discusiones largas (esas van a `research/notes/`).

## Flujo de trabajo: GitHub vs filesystem

**Regla**: GitHub Issues = superficie externa (visibilidad, tracking). Filesystem = superficie interna (pensamiento, debates, diseño, investigación).

| Info | Dónde |
|---|---|
| Trabajo concreto con criterio de cierre (1+ días, 1+ PR / artefacto) | **GitHub Issue** |
| Exploración / investigación / debate en curso | `research/notes/<scratch>.md` (efímero, puede borrarse) |
| Conclusiones estables que guían decisiones | `research/synthesis/<doc>.md` (canon, perdura) |
| Ejemplos canónicos worked-out | `research/examples/<name>.md` |
| Contexto sobre usuario / proyecto / cómo trabajamos | `~/.claude/projects/<slug>/memory/` (cross-session) |
| TODOs efímeros de la sesión actual | Task tool (no persiste) |
| Progreso en issue específico | Comentario del issue |
| Debate sobre cambios de código | Comentario del PR |

**Flujo natural (va izquierda-a-derecha, no al revés):**

```
exploración/debate      diseño/síntesis        trabajo concreto
research/notes/    ->   research/synthesis/ -> GitHub Issue
```

Una idea entra por `notes/`. Si se decanta → pasa a `synthesis/`. Cuando hay unidad PR-sized o entregable claro → recién ahí se vuelve issue. **No crear issues para ideas vagas ni investigación abierta** (poluciona el board).

**Cross-linking** es el pegamento:
- Body de issue linkea a `research/synthesis/foo.md` cuando la justificación vive ahí.
- Comentario de issue linkea a `research/notes/debate_X.md` si hubo debate.
- Docs de synthesis citan issues que motivaron / cerraron.

**Casos concretos:**
- Bug encontrado → issue con `bug`, directo (skip notes).
- "¿Y si probamos X?" → `research/notes/ideas.md`, NO issue todavía.
- Lectura de paper → issue con label `research` (sub-issue del epic de research) → notas en `research/notes/paper-<slug>.md` → conclusiones en `research/synthesis/` al cerrar.
- "Deberíamos rediseñar X" → debate en `notes/` → cuando se decanta → `synthesis/` → si requiere N PRs → epic con sub-issues linkeando al synthesis.

No crear issues para preguntas, discusiones, o trabajo < 1 día.

## Sesiones concurrentes (múltiples Claude en paralelo — fase futura)

- El board es el punto de sync. Todas leen/escriben al mismo.
- Antes de empezar: verificar `Status`. Si está `In Progress`, buscar otro issue.
- Mover a `In Progress` **al empezar, no al final**. Otras sesiones necesitan ver que está tomado.
- Sesiones distintas = branches distintas. Nunca 2 sesiones misma branch.
- Cuando arranque esta fase, evaluar si conviene agregar el campo `Worktree` al Project v2.

## Issue workflow (código + PRs)

Cuando llegue la fase de código:

- **1 issue concreta = 1 PR**.
- Branch: `issue/NNN-short-slug`.
- PR body empieza con `Closes #NNN`.
- Commits referencian con `Refs #NNN <descripción>` (no cierra).
- Squash merge preferido.

## Issue workflow (research — fase actual)

- **1 sub-issue de research = 1 paper leído + notas + synthesis (cuando aplique)**.
- No hay PR (no hay código). El "entregable" es la doc en `research/notes/` y/o `research/synthesis/`.
- Al cerrar: comentario con link al synthesis o a la nota.
- Mover a `In Progress` al empezar la lectura.

## Flujo end-to-end

1. Project board → elegir top del `Todo`.
2. Mover Status → `In Progress` (ver commands.md).
3. Trabajar (leer, codear, escribir).
4. Si hubo código: PR con `gh pr create` + body `Closes #NNN`. Merge → issue cierra → Project mueve a `Done` (auto).
5. Si fue research puro: comentario al issue con link al artefacto + `gh issue close <N> --reason completed`.

## Mantener CLAUDE.md "Epics activos" sincronizada

La tabla en CLAUDE.md debe reflejar el estado real. Si creás/cerrás epic o cambia criterio: actualizar la tabla en el mismo PR (o commit, en fase research).

## Checklist antes de commit que toque tracking

- [ ] Issues cerradas → Status `Done` (auto, verificar).
- [ ] Issues nuevas → Status `Todo` + agregadas al board.
- [ ] Issues en curso → Status `In Progress`.
- [ ] Sub-issues linkeadas vía API nativa.
- [ ] Tabla "Epics activos" en CLAUDE.md refleja el estado actual.

## Referencias

- **`commands.md`** — Recipes exactos por situación (crear, empezar, cerrar, linkear, promover, agregar campo Worktree cuando aplique).
- **`reference.md`** — Project ID, field IDs, option IDs, GraphQL templates, query de refresh.
