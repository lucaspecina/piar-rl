# Mapeo de repos disponibles + estrategia de implementación base

**Issue:** [#11](https://github.com/lucaspecina/piar-rl/issues/11)
**Fecha:** 2026-05-04
**Status:** In Progress

> Pasada superficial de los repos asociados a los papers vecinos a PIAR + del
> ecosistema de RL para LLM agents, para decidir cuál usar como base de
> implementación cuando llegue la fase de código. NO es lectura profunda de
> cada paper — eso queda para los sub-issues del epic #2.

---

## Hallazgos críticos

> 1. **iStar (el plan A original) NO tiene código liberado todavía** al
>    2026-05-04. El paper dice "code will be available soon" desde septiembre
>    2025 pero el repo público no está. Esto **fuerza un cambio de plan base**.
>
> 2. **π-Distill tampoco**: el repo dice "We will make the code available ASAP,
>    we are appending legal approval".
>
> 3. **SREG (el proyecto vecino del usuario) ya decidió `verifiers + prime-rl`
>    como su framework de RL** (`sreg_training_transfer_protocol.md`). Eso
>    genera coherencia con la infraestructura que ya conoce el usuario. Es
>    **la opción más razonable para PIAR**.

---

## Tabla comparativa

### Frameworks de RL (las bases candidatas)

| Repo | Owner | License | Activity | Multi-turn agentic | Notas |
|---|---|---|---|---|---|
| **prime-rl** ([PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)) | Prime Intellect | **Apache-2.0** | v0.5.0 mar 2026, 1.3k ⭐, 1880 commits — **muy activo** | ✅ **Nativo** (Wordle, Wiki Search, SWE, MiniMax-M2.5) | FSDP2 + vLLM + asincrónico. Hackeable, modular (TOML). Built for 1000+ GPUs pero también escala a 1. **Integración nativa con `verifiers`**. |
| **verifiers** ([PrimeIntellect-ai/verifiers](https://github.com/PrimeIntellect-ai/verifiers)) | Prime Intellect | **MIT** | v0.1.13 abr 2026, 4.1k ⭐ — muy activo | ✅ Multi-turn protocols, trajectory tracking | Librería de environments + rubrics. API: `load_environment()` → `vf.Environment`. Rubrics: `async def func(completion, answer) -> float`. **Environments Hub** con cientos de envs. |
| **verl** ([verl-project/verl](https://github.com/verl-project/verl)) | ByteDance Seed | Apache-2.0 | v0.7.1 mar 2026, 21.1k ⭐ — muy activo | Experimental | Backbone de iStar. Foco en math/code (AIME, GSM8K, Codeforces). **Multi-turn experimental, no nativo**. |
| **OpenRLHF** | OpenRLHF | (varios forks) | activo | Sí | Base del fork de SWEET-RL. Más maduro pero más opinionated. |

### Repos de papers vecinos

| Repo | Paper / autor | Framework | Último commit | Estado | Reusable para PIAR | Bloqueo |
|---|---|---|---|---|---|---|
| **iStar** | Cui et al. (#4) | verl-based | **Sin repo público** al 2026-05-04 | ❌ No usable hoy | (no inspeccionable) | Código no liberado. |
| **PRIME / ImplicitPRM** ([PRIME-RL/ImplicitPRM](https://github.com/PRIME-RL/ImplicitPRM)) | Cui/Yuan (#3 + #6) | Custom + DPO/CE/KTO/NCA | Diciembre 2024 — inactivo (12 commits, 171 ⭐) | ⚠️ Útil como ref matemática, no como base | Fórmula `r = β log π_θ/π_ref`. Loss objectives multi-variantes. | Sin Docker/tests. No agentic. |
| **OPSD** ([siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD)) | Zhao (#5) | Custom + TRL GOLD Trainer + accelerate | Marzo 2026 — activo (7 commits) | ✅ Útil como referencia de privileged-context teacher | Per-token JSD + KL clipping. Scripts shell. | **Usa LoRA en el teacher** — choca con invariante #4 si lo copiamos literal. No agentic. |
| **SWEET-RL** ([facebookresearch/sweet_rl](https://github.com/facebookresearch/sweet_rl)) | Meta (#7) | Custom (fork OpenRLHF) + DPO + DeepSpeed | 15 commits, 266 ⭐ | ⚠️ Limitado | Step-level reward con asymmetric info. Best-of-N preference scripts. | **License CC-By-NC** (research ok, comercial no). ColBench no estándar. |
| **Tinker cookbook** ([thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation)) | Thinking Machines | Atado a **Tinker training API (proprietary)** | Activo | Patrón conceptual — `compute_logprobs()`, per-token reverse KL | **Plataforma proprietary** — no ejecutable sin acceso a Tinker. |
| **π-Distill** ([Emilianopp/Privileged-Information-Distillation-and-Self-Distillation](https://github.com/Emilianopp/Privileged-Information-Distillation-and-Self-Distillation)) | Penaloza (#10) | (no especificado) | 3 commits, código pendiente legal | ❌ No usable hoy | (pendiente) | "Code available ASAP, appending legal approval". |

---

## Análisis: cómo encaja cada framework con los invariantes de PIAR

| Invariante | prime-rl + verifiers | verl | SWEET-RL |
|---|---|---|---|
| #4 Teacher = mismos pesos del student | ✅ Compatible (no impone teacher; podemos definir cómputo custom) | ✅ Compatible | ⚠️ Usa critic separado (no aplica directo) |
| #5 PI reproducible y loggable | ✅ verifiers expone reward function donde podemos inyectar y loggear PI | ⚠️ Hay que armar la pieza | ✅ training-time info nativa |
| #6 On-policy scoring estricto | ✅ prime-rl es on-policy nativo; verifiers expone trajectory + completion en la rubric | ✅ GRPO/PPO are on-policy | ✅ Compatible |
| Multi-turn agentic | ✅ **Nativo** (no experimental) | Experimental | Sí pero ColBench-only |

---

## Recomendación de base

### Plan A — **`prime-rl` + `verifiers` (Prime Intellect stack)**

**Por qué es la mejor opción:**
1. **Coherencia con SREG.** El proyecto vecino del usuario ya eligió este stack (`sreg_training_transfer_protocol.md`). Lecciones, scripts, configs y memoria muscular se transfieren.
2. **Multi-turn agentic nativo.** No experimental, no port. Wordle, Wiki Search, SWE training, MiniMax-M2.5 son ejemplos canónicos del repo.
3. **Verifiers es la pieza correcta para definir el environment + rubric.** API limpia: `load_environment()` + rubrics async como funciones que reciben `(completion, answer)`. Eso encaja con PIAR donde la "rubric" sería el cómputo del log-ratio teacher-student.
4. **Environments Hub** con cientos de envs (MATH, Wordle, SWE, deep research). No tenemos que portar todo a mano. WebShop está listado como "infrastructure-intensive" pero soportable; o podemos arrancar con envs más livianos primero.
5. Apache-2.0 + MIT, ambos open. Hackeable, modular.
6. FSDP2 + vLLM ya integrado — escala bien a Azure ML A100/H100.

**Lo que tenemos que escribir nosotros (gaps reales):**

1. **Cómputo del log-ratio teacher-student por acción ReAct.** Esto es el delta de PIAR; no está en ningún framework por diseño. ~50-200 líneas según el plan original.
2. **Privileged-context teacher dentro de la rubric / harness.** Tomar el approach de OPSD (golden answer en prompt) **sin LoRA exclusivo en teacher** — implementar como dynamic sync (mismo modelo, dos forward passes con prompts distintos) o frozen-by-clone (sin adapter).
3. **Step-level reward hooks.** A verificar al cerrar #5 OPSD: si verifiers solo expone reward por trayectoria por defecto, escribir un Environment custom que sí exponga rewards por acción. Es trabajo pero no es bloqueante.
4. **Logging del artefacto privilegiado por episodio** (invariante #5). Hook en la rubric o en el harness.
5. **Toggle frozen vs dynamic sync** del teacher como flag de configuración (invariante #4 sub-decisión abierta).

### Plan B — **verl** (si por alguna razón prime-rl no encaja)

verl es la base subyacente de iStar (cuando se libere). Más maduro en math/code pero **multi-turn agentic es experimental, no nativo**. Mayor stars (21k vs 1.3k) pero menos especializado en agentic.

### Plan C — **Esperar / contactar autores iStar**

Mantener como deuda. Cuando iStar libere su código, compararlo con lo que armamos sobre prime-rl. Si su implementación es mejor, evaluar migrar (debería ser trivial porque iStar usa verl, que es similar arquitectónicamente).

---

## Recomendación concreta (action items)

1. **Adoptar `prime-rl + verifiers` como framework base de PIAR.** Promover esta decisión a `PROJECT.md` fase 2 del roadmap, y a la sección "Tech stack" de `CLAUDE.md`.
2. **Crear sub-issue futuro (cuando llegue fase 4):** "POC del log-ratio teacher-student como rubric de verifiers". Verificar si es viable con la API actual o si hay que extender.
3. **Crear issue standalone con label `parked`:** "Contactar autores iStar para code release timeline". Cuando llegue, comparar con el POC sobre prime-rl.
4. **Ajustar epic #2 sub-issues:**
   - **#5 (OPSD):** además de leer paper, **inspeccionar el repo `siyan-zhao/OPSD`** y decidir si copiamos el LoRA-en-teacher (rompería invariante #4) o vamos por dynamic sync sin adapter.
   - **#3 (Yuan) + #6 (PRIME):** además de leer papers, **inspeccionar el repo `PRIME-RL/ImplicitPRM`** para entender la fórmula del log-ratio en código.
   - **#7 (SWEET-RL):** además de leer paper, **inspeccionar el repo `facebookresearch/sweet_rl`** como ejemplo de "asymmetric privileged" en agentic RL.

---

## Gaps a llenar nosotros (independiente del repo base)

1. ✅ Backbone RL → **prime-rl** (no escribimos nosotros).
2. ✅ Environment definition + rubric API → **verifiers** (no escribimos nosotros).
3. ❌ **Cómputo del log-ratio teacher-student por acción ReAct.** Esto es nuestro.
4. ❌ **Privileged-context formatting.** Nuestro (con OPSD como referencia conceptual).
5. ❌ **Span aggregation con length normalization.** Nuestro.
6. ❌ **Toggle frozen vs dynamic sync** como flag. Nuestro.
7. ⚠️ **Agentic environments específicos** (WebShop, ALFWorld). Verifiers tiene Hub con muchos envs; verificar cuáles están listos. Si WebShop no está, portarlo.

---

## Implicancias para el epic #2

- **Sub-issues #4 (iStar) y #10 (π-Distill):** leer paper igual, pero **ajustar criterio de cierre** — el comentario al cerrar debería incluir "código no disponible al 2026-05; sugerencia: contactar autores / re-evaluar al 2026-Qx".
- **Sub-issues #3 (Yuan), #6 (PRIME):** leer paper + read del repo PRIME-RL/ImplicitPRM (ya inspeccionado superficialmente). Profundizar en la fórmula del log-ratio y cómo se aplica a step level.
- **Sub-issue #5 (OPSD):** leer paper + read del repo siyan-zhao/OPSD (ya inspeccionado). **Decisión sobre invariante #4 sub-decision (LoRA-en-teacher vs sin LoRA).**
- **Sub-issue #7 (SWEET-RL):** leer paper + repo. Sirve como referencia.
- **Sub-issue #8 (Math-Shepherd):** lectura liviana, sin repo.
- **Sub-issue #9 (síntesis cruzada):** heredar conclusiones de este doc + sumar lo que emerja de los reads profundos.

---

## Referencias

**Prime Intellect (stack recomendado):**
- [prime-rl GitHub](https://github.com/PrimeIntellect-ai/prime-rl) — framework de RL agentic
- [verifiers GitHub](https://github.com/PrimeIntellect-ai/verifiers) — environments + rubrics
- [Environments Hub overview](https://docs.primeintellect.ai/tutorials-environments/environments)
- [INTELLECT-3 blog](https://www.primeintellect.ai/blog/intellect-3) — context del stack en producción

**Repos de papers vecinos:**
- [PRIME-RL/ImplicitPRM](https://github.com/PRIME-RL/ImplicitPRM) — Yuan implicit PRM
- [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD) — privileged-context teacher
- [facebookresearch/sweet_rl](https://github.com/facebookresearch/sweet_rl) — asymmetric critic
- [Emilianopp/Privileged-Information-Distillation-and-Self-Distillation](https://github.com/Emilianopp/Privileged-Information-Distillation-and-Self-Distillation) — π-Distill (pendiente release)
- [thinking-machines-lab/tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation) — on-policy distillation pattern

**Backbone alternativo:**
- [verl-project/verl](https://github.com/verl-project/verl) — base de iStar (Plan B)

**Decisión heredada de SREG:**
- `synthetic-research-envs/research/synthesis/sreg_training_transfer_protocol.md` — confirma `verifiers + prime-rl` como stack canónico
