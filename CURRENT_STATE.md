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
- **Sistema de issues + sub-issues nativos** de GitHub. Epic de research (#2)
  cerrado 2026-05-11. Sin epics activos al 2026-05-12.
- **Skill `tracking/`** local (`.claude/skills/tracking/`) con SKILL.md,
  commands.md y reference.md (IDs del Project v2 #5).
- **Skills básicos** del proyecto: `/test` (placeholder hasta que haya código)
  y `/status` (overview del estado del board + commits).
- **Memoria del proyecto** configurada en `~/.claude/projects/<slug>/memory/`
  con entries de project, user, feedback y reference.
- **Research consolidado**: 7 paper notes en `research/notes/paper-*.md` +
  síntesis cruzada (`papers-cross-mapping.md`), índice de decisiones
  (`design-decisions.md`) y deep-dive del vecino α=0 (`piar-delta.md`)
  en `research/synthesis/`.
- **Codebase base**: fork de `CharacterRL-iStar` (Apache-2.0) vendoreado
  en `code/`. Sin modificaciones propias todavía.

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
- `research/synthesis/papers-cross-mapping.md` — síntesis multi-paper + delta de PIAR.
- `research/synthesis/design-decisions.md` — índice de decisiones de diseño (cerradas / inclinadas / abiertas).
- `research/synthesis/piar-delta.md` — deep-dive del vecino más cercano (π-Distill α=0).
- `research/notes/paper-*.md` — notas pesadas por paper.

**Para arrancar trabajo nuevo:**
- Leer `.claude/skills/tracking/SKILL.md` — el workflow operativo.
- Tomar un issue del top de `Todo`, mover a `In Progress`, trabajar, cerrar.

## 3. Qué se está construyendo

**Foco actual (2026-05-12): fase 2 — setup de compute Azure ML Y-TEC.** La fase 1 (research) cerró 2026-05-11.

### Fase 1 — Research (✅ cerrada 2026-05-11)

Epic #2 cerrado. 7 papers vecinos consolidados:

1. ✅ **Yuan 2024 — Implicit PRM** (#3). Notas: [`paper-yuan-implicit-prm.md`](research/notes/paper-yuan-implicit-prm.md).
2. ✅ **iStar — log-ratio en agentes multi-turn** (#4). Notas: [`paper-istar.md`](research/notes/paper-istar.md) — incluye §15 (versión amigable sin formulas) y §16 (dudas conceptuales). Código liberado, ICLR 2026.
3. ✅ **OPSD / Self-Distilled Reasoner** (#5). Notas: [`paper-opsd.md`](research/notes/paper-opsd.md).
4. ✅ **PRIME** (#6). Notas: [`paper-prime.md`](research/notes/paper-prime.md).
5. ✅ **SWEET-RL** (#7). Notas: [`paper-sweet-rl.md`](research/notes/paper-sweet-rl.md).
6. ✅ **Math-Shepherd** (#8). Notas: [`paper-math-shepherd.md`](research/notes/paper-math-shepherd.md).
7. ✅ **π-Distill** (#10). Notas: [`paper-pi-distill.md`](research/notes/paper-pi-distill.md).
8. ✅ **Síntesis cruzada + delta de PIAR explícito** (#9). Consolidado en [`papers-cross-mapping.md`](research/synthesis/papers-cross-mapping.md). Deep-dive del vecino más cercano (π-Distill α=0) en [`piar-delta.md`](research/synthesis/piar-delta.md) (2026-05-12).

Las 9 decisiones cerradas/inclinadas y 9 controles de medición consolidados en [`design-decisions.md`](research/synthesis/design-decisions.md). **Reformulación operativa de LA PREGUNTA contra iStar** anclada en `PROJECT.md`. Stack decidido (#14 cerrado): **fork de `CharacterRL-iStar` vendoreado en `code/`** (Plan B). Plan A (prime-rl + verifiers) descartado.

### Fase 2 — Setup compute (⏳ Now)

Necesario antes de fase 3 (replicar baseline iStar) y fase 4 (implementar PIAR):

- Acceso operacional a Azure ML Y-TEC con la VM `lp-gpu-h100-x2-spot` (2×H100 NVL, 188GB VRAM). Spot → checkpointing no negociable.
- Instalar `code/requirements.txt` (Python 3.12, torch 2.6, vllm 0.8.5, flash-attn 2.7.4) + WebShop env separado (Python 3.10, `code/README.md`).
- Bajar modelo base Qwen2.5-7B-Instruct.

### Issues abiertos relevantes

- [#13](https://github.com/lucaspecina/piar-rl/issues/13) — **parked** (POC Plan A, congelado al descartar Plan A).
- [#15](https://github.com/lucaspecina/piar-rl/issues/15) — research: leakage D.1 + D.9 (a ejecutar en fase 5).
- [#16](https://github.com/lucaspecina/piar-rl/issues/16) — research, **blocked** por setup Azure ML: replicar baseline iStar.
- [#17](https://github.com/lucaspecina/piar-rl/issues/17) — design: spec estructurada como privileged context para WebShop. Extracción puede empezar sin compute.

**Todavía NO se está construyendo**: nada de environment instalado, nada de training corrido, nada de modificaciones propias sobre `code/`. La barrera bloqueante es Azure ML Y-TEC.

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
