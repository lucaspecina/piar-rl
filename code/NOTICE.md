# Procedencia del código en `code/`

Este directorio contiene una **copia vendoreada** del subdirectorio
`CharacterRL-iStar/` del repositorio público de Alibaba Tongyi-ConvAI.

## Origen

- **Repositorio upstream**: [`Tongyi-ConvAI/Qwen-Character`](https://github.com/Tongyi-ConvAI/Qwen-Character)
- **Subdirectorio**: `CharacterRL-iStar/`
- **Commit SHA al momento del vendoring**: `81af1429f306a0fc9f84302528f32821cc2f273b`
- **Fecha de vendoring**: 2026-05-11
- **Paper original**: [iStar — Agentic Reinforcement Learning with Implicit Step Rewards](https://arxiv.org/abs/2509.19199), Liu et al. (Alibaba), aceptado en ICLR 2026.

## Licencia

El código upstream se distribuye bajo **Apache License 2.0** — ver el archivo
[`LICENSE`](LICENSE) en este mismo directorio. Las modificaciones que PIAR aplique
a este código se publican bajo la misma licencia.

## Por qué vendoring (no fork)

Decisión documentada el 2026-05-11 (ver
[`research/synthesis/design-decisions.md`](../research/synthesis/design-decisions.md) A.2
y [`CHANGELOG.md`](../CHANGELOG.md)):

- Modificación esperada chica (~30-100 líneas) → un repo entero para eso es overkill.
- Reproducibilidad del paper se simplifica con una sola URL (`lucaspecina/piar-rl`).
- iStar es paper publicado con código liberado — no esperamos updates upstream relevantes.
- El git log de `piar-rl` queda como historia única e integrada del proyecto.

Existe también un fork separado en [`lucaspecina/Qwen-Character`](https://github.com/lucaspecina/Qwen-Character)
mantenido como snapshot referencial sin desarrollo activo (podría removerse en el
futuro si no se le encuentra utilidad).

## Atribución requerida (Apache-2.0 §4)

> Copyright 2025 Alibaba Group (Tongyi-ConvAI).
> Licensed under the Apache License, Version 2.0 — see LICENSE for full text.

Modificaciones aplicadas por PIAR (lucaspecina/piar-rl) sobre el código original
se registran en los commits del repositorio y se documentan en
[`CHANGELOG.md`](../CHANGELOG.md) a partir del 2026-05-11.

## Qué hace este código (resumen rápido)

Es el framework de RL agentic + implementación del método iStar. Hereda de
[veRL](https://github.com/volcengine/verl) (incluido como `verl/` adentro,
también con modificaciones propias de Alibaba) y agrega:

- `istar/` — implementación del método iStar (PRM aprendido vía DPO trayectorial).
- `agent_system/` — environment wrappers (WebShop, Sokoban), rollout loop multi-turn, reward manager.
- `examples/` — 7 trainers ejecutables (iStar, RLOO, GRPO, REINFORCE++, PPO, GiGPO, PRIME) × 2 envs.
- `gigpo/` — implementación de GiGPO (otro baseline).
- `recipe/` — variantes adicionales (DAPO, R1, SPIN, SPPO, PRIME).
- `verl/` — fork de veRL con modificaciones de Alibaba.

## Qué va a cambiar PIAR sobre este código (esperado)

La modificación primaria es **cambiar cómo se computa el término privilegiado
del log-ratio en el step reward**:

- **iStar (actual)**: `r(action) = β · log[π_φ(a|s) / π_old(a|s)]` donde `π_φ` es
  un PRM separado entrenado con DPO trayectorial sobre outcome rankings.
- **PIAR (modificación)**: `r(action) = β · log[π_old(a|s, golden) / π_old(a|s)]`
  donde el mismo modelo (snapshot `π_old`) se usa en ambos términos, con la
  diferencia única de inyectar el `golden` (spec estructurada del producto target)
  en el prompt del término "privilegiado".

Estimación de líneas a tocar: 30-100. Archivos principales esperados:

- `istar/core_istar.py` — donde se computa el step reward.
- `agent_system/reward_manager/episode.py` — donde se aplica al training.
- `agent_system/multi_turn_rollout/rollout_loop.py` — inyección del golden solo en el forward pass del teacher.

(Estos paths son tentativos; se confirman al arrancar fase 4.)
