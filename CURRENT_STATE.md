# PIAR — Estado actual

> Banner (2026-06-12): **Pivot 2026-06 — el proyecto se re-centró en el
> reward bidireccional (forward pragmático + backward epistémico) + PI
> residual gateada por belief.** El delta original (forward log-ratio
> standalone) murió como contribución (TAMTRL/OPCD/CriticSearch); el espejo
> backward está tomado (IGPO). Fuente de verdad del pivot:
> [`research/synthesis/pivot-2026-06.md`](research/synthesis/pivot-2026-06.md).
> **Fase research parcialmente reabierta** (vecinos nuevos consolidados
> 2026-06-12); el pivot es 100% docs + paper notes + pre-registro, no
> requiere compute. Código de iStar vendoreado en [`code/`](code/) sin
> modificaciones propias. **Nada corrido aún** — falta Azure ML Y-TEC
> operacional.

## 1. Qué corre HOY

**Nada corrido todavía.** Tenemos el código de iStar vendoreado en `code/` (commit
upstream `81af1429f306a0fc9f84302528f32821cc2f273b`, fork de `Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`,
Apache-2.0) pero no se instaló el environment ni se corrió ningún training. Lo que existe es la **infraestructura de tracking, docs, research consolidado, y el codebase base sin tocar**:

- **Repo público** `lucaspecina/piar-rl` con docs raíz (README, PROJECT,
  CURRENT_STATE, CLAUDE, CHANGELOG, AUTORESEARCH) y workflow definido.
- **GitHub Project v2** ["PIAR Roadmap"](https://github.com/users/lucaspecina/projects/5)
  como source of truth del trabajo: campo `Status` (Todo / In Progress / Done).
- **Sistema de issues + sub-issues nativos** de GitHub. Epic de research (#2)
  cerrado 2026-05-11. Epic activo al 2026-06-12: **Pivot 2026-06**.
- **Skill `tracking/`** local (`.claude/skills/tracking/`) con SKILL.md,
  commands.md y reference.md (IDs del Project v2 #5).
- **Skills básicos** del proyecto: `/test` (placeholder hasta que haya código)
  y `/status` (overview del estado del board + commits).
- **Memoria del proyecto** configurada en `~/.claude/projects/<slug>/memory/`
  con entries de project, user, feedback y reference.
- **Research consolidado**: 13 paper notes en `research/notes/paper-*.md`
  (7 de la fase 1 + 6 del pivot 2026-06: IGPO, TAMTRL, survey de credit
  assignment, GiGPO, KnowRL y nota conjunta de menores) + síntesis cruzada
  (`papers-cross-mapping.md`, con sección "Pivot 2026-06"), índice de
  decisiones (`design-decisions.md`, secciones A–N) y deep-dive del vecino
  α=0 (`piar-delta.md`, histórico del diseño pre-pivot) en
  `research/synthesis/`. **Fuente de verdad del pivot:**
  [`pivot-2026-06.md`](research/synthesis/pivot-2026-06.md).
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

**Foco actual (2026-06-12): pivot 2026-06** — re-centrado del proyecto en el
reward bidireccional + PI residual, 100% docs + paper notes + pre-registro
(no requiere compute). En paralelo (cuando se destrabe): fase 2 — setup de
compute Azure ML Y-TEC.

### Pivot 2026-06 (🟡 Now)

Fuente de verdad: [`pivot-2026-06.md`](research/synthesis/pivot-2026-06.md).
Resumen: el forward log-ratio standalone murió como contribución (TAMTRL +
OPCD + CriticSearch); el backward está tomado (IGPO). Lo libre y ahora
central: **(a)** descomposición dual pragmático/epistémico, **(b)** PI
residual gateada por el belief medido del student, **(c)** mapa dónde/por qué
por environment. GiGPO entra como baseline obligatorio (N.10); ALFWorld sube
a benchmark de primera clase (N.11).

Hecho al 2026-06-12 (epic "Pivot 2026-06" en el board):

1. ✅ **Vecinos nuevos consolidados** (6 notas): [`paper-igpo.md`](research/notes/paper-igpo.md), [`paper-tamtrl.md`](research/notes/paper-tamtrl.md), [`paper-survey-ca.md`](research/notes/paper-survey-ca.md), [`paper-gigpo.md`](research/notes/paper-gigpo.md), [`paper-knowrl.md`](research/notes/paper-knowrl.md), [`paper-menores-pivot-2026-06.md`](research/notes/paper-menores-pivot-2026-06.md).
2. ✅ **Síntesis actualizada**: sección "Pivot 2026-06" en [`papers-cross-mapping.md`](research/synthesis/papers-cross-mapping.md) (mapa + veredicto de novedad) y decisiones N.1–N.12 en [`design-decisions.md`](research/synthesis/design-decisions.md).
3. ✅ **PROJECT.md actualizado post-pivot** (aprobado por Lucas y aplicado 2026-06-12, commit `acb799f`): LA PREGUNTA dual+residual, invariantes 10/11, roadmap con Figura 1 como gate. Registro del cambio en [`proposal-project-md.md`](research/synthesis/proposal-project-md.md). (#22 cerrado)
4. 🟡 **#17 en curso**: componentes g_i + wrappers + formato canónico definidos en [`pi-webshop.md`](research/synthesis/pi-webshop.md); falta la validación contra el dataset real (descarga bloqueada — requiere Python/VM).
5. ✅ **Pre-registro de Figura 1 congelado** ([`figura1-prereg.md`](research/synthesis/figura1-prereg.md), #23 cerrado): umbrales fijados 2026-06-12 con los defaults del review externo, bajo delegación explícita de Lucas. Es el experimento día-1 de compute y el gate go/no-go de la implementación.

La fase 1 original (7 vecinos: Yuan, iStar, OPSD, PRIME, SWEET-RL,
Math-Shepherd, π-Distill — epic #2) cerró 2026-05-11; sus notas y
[`piar-delta.md`](research/synthesis/piar-delta.md) siguen siendo válidas
como historia del diseño pre-pivot. Stack decidido (#14): **fork de
`CharacterRL-iStar` vendoreado en `code/`**.

### Fase 2 — Setup compute (⏳ bloqueada, en paralelo)

Necesario antes de la replicación de baselines y la Figura 1:

- Acceso operacional a Azure ML Y-TEC con la VM `lp-gpu-h100-x2-spot` (2×H100 NVL, 188GB VRAM). Spot → checkpointing no negociable.
- Instalar `code/requirements.txt` (Python 3.12, torch 2.6, vllm 0.8.5, flash-attn 2.7.4) + WebShop env separado (Python 3.10, `code/README.md`).
- Bajar modelo base Qwen2.5-7B-Instruct.

### Issues abiertos relevantes

- **Epic Pivot 2026-06** con sub-issues de paper notes (cerradas), edición de docs y pre-registro de Figura 1 — ver el [board](https://github.com/users/lucaspecina/projects/5).
- [#17](https://github.com/lucaspecina/piar-rl/issues/17) — design, **promovido a prioridad alta por el pivot**: componentes g_i + formato canónico (N.1, N.8). Primer ladrillo del método; la extracción puede empezar sin compute.
- [#15](https://github.com/lucaspecina/piar-rl/issues/15) — research: leakage D.1 + D.9, ahora aplicado a *ambas* direcciones (forward y backward).
- [#16](https://github.com/lucaspecina/piar-rl/issues/16) — research, **blocked** por setup Azure ML: replicar baseline iStar (sigue vigente como B1).
- [#13](https://github.com/lucaspecina/piar-rl/issues/13) — cerrado `not planned` (POC Plan A) desde 2026-05-11; sin cambios.

**Todavía NO se está construyendo**: nada de environment instalado, nada de
training corrido, nada de modificaciones propias sobre `code/`. La barrera
bloqueante de compute es Azure ML Y-TEC; el pivot avanza sin ella.

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
