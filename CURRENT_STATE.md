# PIAR — Estado actual

> Banner duro: **HOY no corre nada de código.** PIAR está en fase de research
> consolidado de papers vecinos. La síntesis del cross-paper-mapping todavía no
> está hecha. Cuando esté, recién ahí pasa a la fase de implementación.

## 1. Qué corre HOY

**Cero código.** No hay environment, no hay deps, no hay modelos, no hay
experimentos. Lo único que existe es la **infraestructura de tracking y docs**:

- **Repo público** `lucaspecina/piar-rl` con docs raíz (README, PROJECT,
  CURRENT_STATE, CLAUDE, CHANGELOG, AUTORESEARCH) y workflow definido.
- **GitHub Project v2** ["PIAR Roadmap"](https://github.com/users/lucaspecina/projects/5)
  como source of truth del trabajo: campo `Status` (Todo / In Progress / Done).
- **Sistema de issues + sub-issues nativos** de GitHub. Epic activo: research
  de papers vecinos.
- **Skill `tracking/`** local (`.claude/skills/tracking/`) con SKILL.md,
  commands.md y reference.md (IDs del Project v2 #5).
- **Skills básicos** del proyecto: `/test` (placeholder hasta que haya código)
  y `/status` (overview del estado del board + commits).
- **Memoria del proyecto** configurada en `~/.claude/projects/<slug>/memory/`
  con entries de project, user, feedback y reference.
- **Estructura de research**: `research/{notes,synthesis,examples,archive}/`
  vacíos pero listos.

## 2. Cómo usar el sistema hoy

**Setup local:**
```bash
git clone https://github.com/lucaspecina/piar-rl.git
cd piar-rl
# No hay deps todavía. No hay environment. Solo docs.
```

**Para entender el proyecto (3 pasos):**
1. Leer `PROJECT.md` (vision + LA PREGUNTA + invariantes).
2. Leer este archivo (`CURRENT_STATE.md`) — ya estás acá.
3. Mirar el [Project v2](https://github.com/users/lucaspecina/projects/5) o `gh issue list -R lucaspecina/piar-rl` para ver qué está en `Todo` / `In Progress` / `Done`.

**Para consultar conclusiones de papers ya leídos:**
- `research/synthesis/` cuando los issues de research empiecen a cerrar.
- Mientras tanto, los issues activos tienen el progreso en sus comentarios.

**Para arrancar trabajo nuevo:**
- Leer `.claude/skills/tracking/SKILL.md` — el workflow operativo.
- Tomar un issue del top de `Todo`, mover a `In Progress`, trabajar, cerrar.

## 3. Qué se está construyendo

**Foco actual: fase 1 del roadmap (research consolidado).**

Epic activo: **Research — síntesis de papers vecinos a PIAR** (ver Project v2).
Sub-issues:

1. ✅ **Yuan 2024 — Implicit PRM** (#3, cerrado 2026-05-07). Notas: [`research/notes/paper-yuan-implicit-prm.md`](research/notes/paper-yuan-implicit-prm.md).
2. ✅ **iStar — log-ratio en agentes multi-turn** (#4, cerrado 2026-05-07). Notas: [`research/notes/paper-istar.md`](research/notes/paper-istar.md). Código liberado, ICLR 2026.
3. ⏳ OPSD / Self-Distilled Reasoner — privileged-context teacher (#5). **Próximo.** Resuelve sub-decisión frozen vs co-evolución del invariante 4.
4. ⏳ π-Distill — vecino conceptual (#10).
5. ⏳ PRIME — framework completo de RL con implicit PRM (#6).
6. ⏳ SWEET-RL — critic asimétrico privilegiado (#7).
7. ⏳ Math-Shepherd — predecesor histórico (#8).
8. ⏳ Síntesis cruzada + delta de PIAR explícito (#9, cierre del epic).

**Criterio de cierre del epic**: cada paper tiene un doc en
`research/notes/paper-<slug>.md` y la síntesis cruzada está en
`research/synthesis/papers-cross-mapping.md` con el delta de PIAR aislado.

**Todavía NO se está construyendo**: nada de código, nada de environment,
nada de Azure ML, nada de experimentos. Esas fases (2–7 del roadmap)
arrancan recién cuando el epic actual cierre.

## 4. Donde mirar para qué

| Si querés... | Andá a |
|---|---|
| Vision e invariantes | `PROJECT.md` |
| Estado del proyecto (este archivo) | `CURRENT_STATE.md` |
| Operativa Claude Code (workflow) | `CLAUDE.md` |
| Workflow operativo de tracking | `.claude/skills/tracking/SKILL.md` |
| Roadmap, trabajo pendiente, prioridades | [Project v2](https://github.com/users/lucaspecina/projects/5) o `gh issue list -R lucaspecina/piar-rl` |
| Historia de cambios | `CHANGELOG.md` |
| Notas pesadas de papers leídos | `research/notes/` |
| Conclusiones consolidadas | `research/synthesis/` |
| Ejemplos canónicos | `research/examples/` |
| Skills del proyecto | `.claude/skills/{tracking,test,status}/` |
| Implementación target (no existe todavía) | `ARCHITECTURE.md` (deferred hasta 3+ módulos) |
