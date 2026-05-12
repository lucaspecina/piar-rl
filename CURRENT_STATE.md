# PIAR — Estado actual

> Banner (2026-05-11): **Fase 1 (research) cerrada. Fase 2 (setup compute) próxima.**
> Código de iStar vendoreado en [`code/`](code/) como base de PIAR, sin modificaciones
> propias todavía. **Nada corrido aún** — falta Azure ML Y-TEC operacional.

## 1. Qué corre HOY

**Nada corrido todavía.** Tenemos el código de iStar vendoreado en `code/` (commit
upstream `81af1429f306a0fc9f84302528f32821cc2f273b`, fork de `Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`,
Apache-2.0) pero no se instaló el environment ni se corrió ningún training. Lo que existe es la **infraestructura de tracking, docs, research consolidado, y el codebase base sin tocar**:

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

**Foco actual: cierre de la fase 1 (research consolidado) + reformulación operativa de LA PREGUNTA contra iStar.**

Epic activo: **Research — síntesis de papers vecinos a PIAR** (#2). Estado de sub-issues:

1. ✅ **Yuan 2024 — Implicit PRM** (#3). Notas: [`paper-yuan-implicit-prm.md`](research/notes/paper-yuan-implicit-prm.md).
2. ✅ **iStar — log-ratio en agentes multi-turn** (#4). Notas: [`paper-istar.md`](research/notes/paper-istar.md) — incluye §15 (versión amigable sin formulas) y §16 (dudas conceptuales). Código liberado, ICLR 2026.
3. ✅ **OPSD / Self-Distilled Reasoner** (#5). Notas: [`paper-opsd.md`](research/notes/paper-opsd.md).
4. ✅ **PRIME** (#6). Notas: [`paper-prime.md`](research/notes/paper-prime.md).
5. ✅ **SWEET-RL** (#7). Notas: [`paper-sweet-rl.md`](research/notes/paper-sweet-rl.md).
6. ✅ **Math-Shepherd** (#8). Notas: [`paper-math-shepherd.md`](research/notes/paper-math-shepherd.md).
7. ✅ **π-Distill** (#10). Notas: [`paper-pi-distill.md`](research/notes/paper-pi-distill.md).
8. ⏳ **Síntesis cruzada + delta de PIAR explícito** (#9) — único pendiente, cierre del epic.

Las 9 decisiones cerradas/inclinadas y 8 controles de medición salieron consolidados en [`research/synthesis/design-decisions.md`](research/synthesis/design-decisions.md).

**Reformulación operativa post-iStar (2026-05-11)**: LA PREGUNTA quedó re-anclada contra el baseline experimental concreto (PIAR vs iStar con setup intacto, cambiando solo cómo se obtiene el término privilegiado del log-ratio). Ver `PROJECT.md` "Reformulación operativa contra iStar". Documentado también en `paper-istar.md` §15-§16 (versión amigable + dudas teóricas reconocidas).

**Verificación operacional del repo de iStar (2026-05-11)**: `Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar` es maduro y ejecutable. 7 trainers (iStar, RLOO, GRPO, REINFORCE++, PPO, GiGPO, PRIME) × 2 environments (WebShop, Sokoban). Modelo base Qwen2.5-7B-Instruct, hardware target 8×H100/A100 (compatible con Y-TEC). Framework veRL (fork de Alibaba). Esto reabre Plan B como opción concreta vs Plan A (prime-rl) — decisión pendiente.

**Todavía NO se está construyendo**: nada de código, nada de environment,
nada de Azure ML, nada de experimentos. Esas fases arrancan al cerrar #9 + decidir Plan A/B + definir el privileged context concreto para WebShop.

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
