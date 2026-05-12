# PIAR — Puntos de implementación quirúrgica sobre `code/`

> **Qué es esto:** mapeo end-to-end del pipeline de iStar vendoreado en `code/` con los puntos exactos a modificar para llegar a PIAR. Cada referencia es `archivo:línea`.
>
> **Qué NO es:** no es PR todavía, no es código pegable. Es el plano para que el implementador (humano o agente) pueda atacar la fase 4 sin re-explorar el repo.
>
> **Pre-requisito de lectura:** `PROJECT.md` (invariantes 1-5), `research/synthesis/design-decisions.md` (especialmente A.2, B.1-B.7, C.1-C.5, D.9).
>
> **Decisión arquitectónica central que justifica esta estrategia:** `dp_rm.py:161` ya computa `q = rm_log_labels - ref_log_labels`. El framework de iStar es estructuralmente un log-ratio — solo cambia **qué modelo y qué prompt** alimentan cada término. PIAR es un reemplazo de inputs, no un rewrite.
>
> ## ⚠️ Estado post-review Codex (2026-05-12)
>
> Este doc fue revisado críticamente por Codex después de su primera versión. **5 hallazgos relevantes** que cambian el framing en partes específicas — están consolidados en la **§6 — Adendum post-review Codex** al final del doc. Cuando veas un ⚠️ inline, te lleva al item de la §6 que corrige esa parte.
>
> **Resumen de los cambios más importantes**:
> - La "Opción A" de §2.2 (mantener el RM worker, solo cambiar inputs) es **inválida** — rompe el invariante 4 sin querer. Solo Opción B (usar el actor para los dos forward passes) es correcta. Ver §6.2.
> - El LOC estimate de 30-100 / 40-65 es **optimista al borde de falso**. Estimación corregida: **100-180 LOC**. Ver §6.4.
> - `update=none` **no** salta el init del reward model — solo evita el `optimizer.step()`. Sigue cargando un Qwen2.5-7B extra en GPU. Ver §6.1.
> - Goal injection requiere ~25-40 LOC de plumbing real, no 10-15. Ver §6.3.
> - El score se computa token-level y se colapsa al final → posible leakage de tokens `<think>` privilegiados. Cruza con Apuesta A de [`piar-delta.md`](piar-delta.md) §4.1. Ver §6.5.

---

## Sección 1 · Pipeline actual de iStar (cómo se computa `rm_scores`)

### 1.1 Entry point y cableado

| # | Paso | Archivo:Línea | Qué hace |
|---|---|---|---|
| 1 | Trainer entry | `code/examples/istar_trainer/run_webshop.sh:19` | Llama `verl.trainer.main_ppo` con `algorithm.adv_estimator=istar_rloo`, `reward_model.enable=True`, `reward_model.model.path=$model_path` (mismo modelo que el policy), `reward_model.model.loss_type=eto`, `reward_model.model.update=after`, `reward_model.step_granularity=step`. |
| 2 | Worker registration | `code/verl/trainer/main_ppo.py:133-134` | Si `reward_model.enable`, importa `ISTARRewardModelWorker` (no el `RewardModelWorker` default de verl) y lo registra como `Role.RewardModel`. |
| 3 | Reward manager | `code/verl/trainer/main_ppo.py:142-150` | Instancia `EpisodeRewardManager` (de `agent_system.reward_manager.episode`). Este es el **outcome reward fn** (rule-based), no el PRM. |
| 4 | Trajectory collector | `code/verl/trainer/main_ppo.py:159-160` | Instancia `TrajectoryCollector` que orquesta el rollout multi-turn ReAct. |

### 1.2 Construcción del PRM (modelo `π_φ` separado)

| # | Paso | Archivo:Línea | Qué hace |
|---|---|---|---|
| 5 | RM worker init | `code/istar/rm_fsdp_workers.py:236-265` | `init_model()`: construye dos modelos via `_build_reward_ref_model_optimizer` — `self.reward_module` (π_φ, **trainable**) y `self.ref_module` (π_ref, frozen, opcional). Crea `DataParallelISTARRewardModel`. |
| 6 | Modelo trainable | `code/istar/rm_fsdp_workers.py:115-126` | `reward_module = AutoModelForCausalLM.from_pretrained(config.model.path, ...)` — se carga **el mismo checkpoint que el policy** (Qwen2.5-7B-Instruct) y se le va a aplicar DPO/ETO loss. |
| 7 | Ref module opcional | `code/istar/rm_fsdp_workers.py:165-181` | Si `config.model.ref_path` está seteado, carga `ref_module` (frozen). En `run_webshop.sh:53` está `reward_model.model.ref_path=null` → **no se usa ref_module externo**; el log-ratio se hace contra `micro_batch["old_log_probs"]` directamente (línea `dp_rm.py:158`). |
| 8 | FSDP wrapping | `code/istar/rm_fsdp_workers.py:183-195` | Envuelve `reward_module` con `FullyShardedDataParallel` para sharding. Mismo wrapping para `ref_module` si existe. |
| 9 | Optimizer | `code/istar/rm_fsdp_workers.py:214-219` | `optim.AdamW(reward_module.parameters(), lr=config.model.optim.lr, ...)`. **El PRM se entrena.** |

### 1.3 Rollout multi-turn + outcome reward

| # | Paso | Archivo:Línea | Qué hace |
|---|---|---|---|
| 10 | Loop principal | `code/verl/trainer/ppo/ray_trainer.py:1089-1149` | Por epoch y batch: `multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train=True)`. |
| 11 | Multi-turn loop | `code/agent_system/multi_turn_rollout/rollout_loop.py:263-389` | `vanilla_multi_turn_loop`: reset envs → loop max_steps: build obs (con history) → `actor_rollout_wg.generate_sequences(batch_input)` → `envs.step(text_actions)` → acumular en `total_batch_list`. Termina con `success_evaluator`. |
| 12 | Outcome reward | `code/agent_system/reward_manager/episode.py:33-122` | `EpisodeRewardManager.__call__`: lee `episode_rewards` per-episode (rule-based, 10.0 si won), pone scalar en la **última posición válida** del response → `reward_tensor[i, valid_response_length - 1] = score`. Guarda `data.batch["acc"]` (escalar per-trajectory). |

### 1.4 Cómputo de `rm_scores` (step reward del PRM)

| # | Paso | Archivo:Línea | Qué hace |
|---|---|---|---|
| 13 | Llamada al RM | `code/verl/trainer/ppo/ray_trainer.py:1242-1268` | Si `self.use_rm`: según `update_style` (`after`, `reverse`, `none`) llama `update_rm`, `update_rm_eto` o `compute_rm_score`. Con `run_webshop.sh` está `update=after` + `loss_type=eto` → llama **`update_rm_eto`** que entrena el PRM Y devuelve scores. |
| 14 | Forward por micro-batch | `code/istar/dp_rm.py:53-208` | `_forward_micro_batch`: el corazón del cómputo. Forward del `reward_module` → `rm_log_labels` (log probs por token, línea `dp_rm.py:91-94`). Si hay `ref_module` forward también, sino `ref_log_labels = micro_batch["old_log_probs"]` (`dp_rm.py:158`). |
| 15 | **EL log-ratio** | `code/istar/dp_rm.py:161` | `q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]` — diff de log-probs, este es el `q_t` algebraico de Yuan. |
| 16 | Escalado por β | `code/istar/dp_rm.py:170-189` | `r = q * beta` (con `beta_train=0.05` desde `run_webshop.sh:58`). Si `lambda > 0` aplica GAE; con default `lambda=0.0` queda `r = q * 0.05`. |
| 17 | Agregación step-level | `code/istar/dp_rm.py:199-204` | Con `step_granularity=step` (`run_webshop.sh:64`): suma `r[:max_positions[i]]` y lo asigna a `max_positions[i] - 1` (última posición válida del step). Esto produce `token_level_score` con UN valor escalar por trayectoria activa, en la última posición. |
| 18 | DPO/ETO update | `code/istar/dp_rm.py:377-448` | `update_rm_eto`: además de computar scores, computa `compute_eto_loss_rm` (rm_utils.py:43-135) — DPO loss trajectory-level pareando trayectorias exitosas/fallidas dentro del mismo `uid` (env_group). `loss.backward()` + `optimizer.step()`. |
| 19 | Return al trainer | `code/istar/rm_fsdp_workers.py:351-393` | `update_rm_eto` worker wrapper retorna `DataProto({"rm_scores": rm_scores}, meta_info={"metrics": ...})`. Se hace `batch = batch.union(reward_output)` en `ray_trainer.py:1265`. |

### 1.5 Llegada a `core_istar.py`

| # | Paso | Archivo:Línea | Qué hace |
|---|---|---|---|
| 20 | Advantage compute | `code/verl/trainer/ppo/ray_trainer.py:374-382` | `core_istar.compute_istar_rloo_advantage(data, step_advantage_w=1.0)`. |
| 21 | Lee rm_scores | `code/istar/core_istar.py:33-44` | `step_rewards = data.batch['rm_scores']` + `token_level_rewards = data.batch['token_level_rewards']`. Computa `step_rloo_reward` + `episode_rloo_reward` y los combina con `step_advantage_w=1.0`. **Este archivo NO se toca.** |

**Conclusión del trace:** todo el cableado downstream del PRM (`core_istar.py`, `ray_trainer.py:1242-1290`) consume `data.batch["rm_scores"]` como un opaque tensor `(bs, response_length)`. Si nosotros **reemplazamos** el cálculo upstream pero **respetamos la forma del tensor**, downstream no se entera.

---

## Sección 2 · Hot spots para PIAR (qué cambia exactamente)

> **Estrategia general:** mantener todo el wiring de `Role.RewardModel`, el worker FSDP y la firma de `rm_scores`. Cambiar internamente: (a) no hay PRM entrenable, ambos forward pass los hace **el mismo modelo** = una snapshot reciente de `π_old` (decisión C.2); (b) la asimetría vive en el `input_ids` del numerador, que incluye el golden serializado.

### 2.1 Bypass del PRM training (eliminar DPO update)

#### Cambio 2.1.1 — `ray_trainer.py:1242-1268` (driver de update_style)

**Actual (`code/verl/trainer/ppo/ray_trainer.py:1242-1263`):**
```python
if self.use_rm:
    update_style = self.config.reward_model.model.get("update", "none")
    if update_style == "none":
        reward_output = self.rm_wg.compute_rm_score(batch)
    elif update_style == "after":
        if self.config.reward_model.model.loss_type == "eto":
            reward_output = self.rm_wg.update_rm_eto(batch)
        else:
            reward_output = self.rm_wg.update_rm(batch)
    elif update_style == "reverse":
        # ...
```

**Propuesto:** dos opciones aparentemente equivalentes — **⚠️ pero ver §6.1 antes de elegir Opción A**.

**Opción A — config-only**: cambiar en `code/examples/istar_trainer/run_webshop.sh:56` el flag `reward_model.model.update=after` → `reward_model.model.update=none`. El branch `update_style == "none"` (`ray_trainer.py:1244-1245`) ya llama `compute_rm_score` puro sin DPO update.

> **⚠️ Caveat post-Codex (§6.1):** Opción A **NO salta el init del reward model**. `main_ppo` igual instancia `Role.RewardModel`, `ray_trainer` llama `rm_wg.init_model()`, y `rm_fsdp_workers.py` carga modelo + optimizer + scheduler + checkpoint manager (`code/istar/rm_fsdp_workers.py:115-265`). Solo se evita `optimizer.step()`. Costo: ~14 GB de VRAM extra ocupados por un Qwen2.5-7B duplicado e inútil. Si Opción A se mantiene, agregar guards en `main_ppo.py` para no instanciar el RM worker cuando `update=none` (~10-15 LOC extra). Si se va por Opción B-real (§6.2), el RM worker entero deja de tener sentido y se puede bypasear.

**Opción B — code-level**: agregar `elif update_style == "piar":` que llame un nuevo `compute_piar_rm_score`. Más explícito pero más LOC. **Esto NO es la "Opción B preferida" que aparece en §2.2 — son dos cosas distintas.** Esta variante 2.1 sigue usando el RM worker.

**Justificación:** PIAR no entrena el numerador. El mismo modelo `π_old` se usa con y sin golden — el log-ratio mide solo diferencia de contexto. **Recomendado: ver §2.2 + §6.2 — el cambio operativo importante NO está en este nivel, está en el de §2.2.**

#### Cambio 2.1.2 — `rm_fsdp_workers.py:215-219` (optimizer)

**Actual (`code/istar/rm_fsdp_workers.py:214-219`):**
```python
reward_optimizer = optim.AdamW(
    reward_module.parameters(),
    lr=config.model.optim.lr,
    betas=config.model.optim.get("betas", (0.9, 0.999)),
    weight_decay=config.model.optim.get("weight_decay", 1e-2),
)
```

**Propuesto:** con Opción A de 2.1.1, el optimizer queda construido pero **nunca se llama**. Cero líneas a tocar para el código que existe. **⚠️ Pero ver §6.1**: el optimizer está cargado en VRAM aunque no se invoque. Si Opción A se mantiene **y** la memoria es restrictiva: guard `if config.model.get("update", "none") == "none":` para skippear `reward_optimizer` + `reward_lr_scheduler` (~3-5 LOC), o mejor, bypass entero del RM worker desde `main_ppo`. Si se va por la opción correcta (§2.2 Opción B = usar el actor, no el RM worker), todo esto queda muerto sin acción adicional.

### 2.2 Reescritura del log-ratio: PRM frozen → snapshot policy con golden

#### Cambio 2.2.1 — `rm_fsdp_workers.py:80-234` (modelo cargado en RM worker)

**Actual:** `_build_reward_ref_model_optimizer` carga `reward_module` desde `config.model.path` (línea 89, 120-126), que es **el mismo path que el policy** (`run_webshop.sh:11+54`: `reward_model.model.path=$model_path`).

**Sub-opciones — ⚠️ ver §6.2: solo una es correcta**:

**❌ Opción A (INVÁLIDA) — sync explícito de pesos del actor al RM worker** antes de cada `compute_rm_score`. Sería agregar un `@register` method que copie state_dict del actor al RM (~15-25 LOC). **Por qué no funciona**: el RM worker se carga UNA VEZ desde `model.path` y queda **desacoplado** del actor que PPO sigue actualizando. Sin sync explícito el numerador deja de ser `π_old(a|s, golden)` y pasa a ser `π_init(a|s, golden)` (modelo del step 0 con golden). Después del primer update PPO, `θ_inicial ≠ θ_t`, y el log-ratio mezcla "info que aporta el golden" con "weight-drift inicial→actual" — **se rompe el invariante 4 sin querer** (mismos pesos del student). Esto es exactamente la razón por la cual C.2 cambió de frozen θ₀ a `π_old` el 2026-05-11. Mantener el RM worker estático reintroduce el problema en la implementación. Para que Opción A funcione habría que agregar el sync explícito en cada step — entonces ya tenés dos modelos en VRAM Y un sync overhead — peor que B en todas las dimensiones.

**✅ Opción B (la única correcta) — eliminar `reward_module` y reusar el actor.** En vez de tener dos modelos en GPU, usar `self.actor_rollout_wg.compute_log_prob(batch_with_golden)` directamente (existe ya en `code/verl/workers/fsdp_workers.py:665-705`). Esto pone toda la lógica de PIAR fuera del `ISTARRewardModelWorker`. Como el actor SÍ se actualiza con PPO, sus weights son `θ_t = π_old` por definición. El log-ratio queda puro: `log[π_old(a|s, golden)] - log[π_old(a|s)]`. Invariante 4 preservado. Ver Cambio 2.2.3.

#### Cambio 2.2.2 — `dp_rm.py:53-208` (`_forward_micro_batch`)

**Actual:** computa `rm_log_labels` con `self.reward_module(input_ids=micro_batch["input_ids"], ...)` (línea 101-106) y `ref_log_labels = micro_batch["old_log_probs"]` (línea 158, default cuando `ref_module is None`).

**Con Opción B (la única correcta):** `dp_rm.py` entero queda como **dead code** accesible solo bajo `update=after` legacy. La llamada `self.rm_wg.compute_rm_score(batch)` se reemplaza por un nuevo método `compute_piar_step_reward` (~50-80 LOC) que internamente hace dos llamadas a `actor_rollout_wg.compute_log_prob` — una con prompt + golden, otra con prompt sin golden — y computa `q = teacher_log_probs - student_log_probs` con el mismo agregado step-level que `dp_rm.py:199-204`. Ver Cambio 2.2.3.

**⚠️ Riesgo verificable (§6.1)**: `dp_rm.py:226-229` hace `self.ref_module.eval()` sin guard `if self.ref_module is not None`. En el config actual (`run_webshop.sh:53`: `ref_path=null`), ese código no se ejecuta porque `compute_rm_score` ni se llama bajo Opción B. **Pero si se entra por `update=none` con el RM worker cargado (Opción A o un híbrido)**, hay riesgo de crash `NoneType.eval()`. Mitigación: ir directo a Opción B; o agregar el guard si se quiere tocar lo menos posible.

#### Cambio 2.2.3 — Nueva función `compute_piar_step_reward` (recomendado)

**Justificación:** evita mantener dos modelos en GPU. El step reward se calcula como dos forward passes consecutivos del actor. Costo: **2× forward pass del policy por rollout step** (igual que `update_rm_eto` hoy: hace un forward del reward_module + uno del ref_module si existe; sin ref_module reusa `old_log_probs` y hace solo uno; con PIAR hacemos siempre dos forward del actor).

**Ubicación propuesta:** nuevo archivo `code/istar/piar_step_reward.py` (~50-80 LOC) o como método dentro del trainer (~30 LOC inline).

**Pseudocódigo (~40 LOC reales):**
```python
def compute_piar_step_reward(batch, actor_rollout_wg, golden_token_ids,
                             beta=0.05, step_granularity="step"):
    """Computa rm_scores como log-ratio teacher-vs-student del mismo policy.
    El teacher batch tiene golden inyectado en el prompt; el student no."""
    # 1) Student forward: batch tal cual viene del rollout
    student_out = actor_rollout_wg.compute_log_prob(batch)
    student_log_probs = student_out.batch["old_log_probs"]  # (bs, resp_len)

    # 2) Build teacher batch: inyectar golden en input_ids/attention_mask/position_ids
    teacher_batch = inject_golden_into_prompt(batch, golden_token_ids)

    # 3) Teacher forward
    teacher_out = actor_rollout_wg.compute_log_prob(teacher_batch)
    teacher_log_probs = teacher_out.batch["old_log_probs"]  # (bs, resp_len)

    # 4) Log-ratio (mismo modelo, prompts distintos → mide info gain del golden)
    q = teacher_log_probs - student_log_probs  # (bs, resp_len)

    # 5) Step-level aggregation (replica `dp_rm.py:199-204`)
    response_mask = batch.batch["response_mask"]
    rm_scores = torch.zeros_like(q)
    r = q * beta
    if step_granularity == "step":
        max_positions = response_mask.sum(-1)
        for i in range(batch.batch_size):
            rm_scores[i, max_positions[i] - 1] = r[i, :max_positions[i]].sum()
    else:  # token-level fallback
        rm_scores = r * response_mask

    return rm_scores
```

**Justificación:** preserva exactamente la firma `rm_scores: (bs, response_length)` que consume `core_istar.py`. No toca `core_istar.py`. Reusa toda la infra de `compute_log_prob` ya testeada en verl.

#### Cambio 2.2.4 — `ray_trainer.py:1242-1268` (callsite del cómputo)

**Si Opción B con `compute_piar_step_reward`:** reemplazar las 27 líneas del if/elif (`ray_trainer.py:1242-1268`) por ~5 líneas:
```python
if self.use_rm:
    golden_ids = batch.non_tensor_batch.get("golden_token_ids", None)
    rm_scores = compute_piar_step_reward(
        batch, self.actor_rollout_wg, golden_ids,
        beta=self.config.reward_model.model.beta_train,
    )
    batch.batch["rm_scores"] = rm_scores
```

**Justificación:** colapsa el dispatch a un solo path. El nombre `use_rm` se mantiene para no cambiar la lógica de gating downstream.

### 2.3 Mantener invariante de shape de `rm_scores`

**Mantener:** `rm_scores.shape == (bs, response_length)` con el escalar agregado en la última posición válida (`max_positions[i] - 1`) cuando `step_granularity=step` (decisión B.3 — action-level).

**Justificación:** `core_istar.py:122` hace `step_rewards.sum(dim=-1)` — si la agregación cambia, los rewards se duplican o se anulan. Mantener la convención literal.

---

## Sección 3 · Mecanismo de golden injection

### 3.1 Dónde vive el goal/spec

**Estructura del goal** (`code/agent_system/environments/env_package/webshop/webshop/web_agent_site/engine/goal.py:48-58`, función `get_human_goals`):

```python
{
    'asin': 'B07XXX...',                  # product ID literal (LEAK CONTROL — D.9 upper bound)
    'category': '...',
    'query': 'search query usado en NL',
    'name': 'Product Display Name',
    'product_category': 'Electronics > ...',
    'instruction_text': 'I am looking for a ... lower than X dollars',  # NL prompt
    'attributes': ['brand:Samsung', 'color:black', ...],  # spec estructurada — D.9 candidato
    'price_upper': 50.0,
    'goal_options': {'size': 'medium', 'color': 'red', ...},
    'weight': 1
}
```

**Persistencia durante el episodio** (`code/agent_system/environments/env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py:519-528`): cuando `reset(session=idx)` se llama, el `SimServer.receive` setea `self.user_sessions[session_id] = {'goal': goal, 'done': False}` (línea 521). El goal vive ahí hasta el reset siguiente.

**⚠️ Plumbing real (§6.3)**: el `'get_goals'` que expone `envs.py:71-72` devuelve la **lista global** de todos los goals del dataset (`env.server.goals`), no el goal del episodio actual. Lo que necesitamos vive en `env.server.user_sessions[session_id]['goal']` y nadie lo expone hoy. Plus: el `reset/step` del worker pipe no devuelve el goal en `info` (`code/agent_system/environments/env_package/webshop/envs.py:60-72`, `rollout_loop.py:328-356`). Costo real: **25-40 LOC** distribuidas en:

- `envs.py:71-72` — extender el handler con `'get_current_goal'` + `'get_current_goals_all_envs'` (multi-process). ~8-10 LOC.
- `WebshopMultiProcessEnv` (`envs.py`, clase) — método `get_current_goals()` que recolecta de cada subproceso via pipe. ~8-10 LOC.
- `WebshopEnvironmentManager` (`env_manager.py`) — atributo `self.current_goals: list[dict]` actualizado en `reset` y propagado a `step`. ~8-10 LOC.
- `rollout_loop.py:336-339` — pasar `current_goals` al batch como `non_tensor_batch['golden_dict']`. ~3-5 LOC.

Total: 25-40 LOC, no 10-15 como decía la primera versión del doc.

### 3.2 Cómo serializar el golden al prompt del teacher

**Decisión C.5 (design-decisions.md):** el privileged context es la **spec estructurada** (no la trayectoria humana, no el product ID literal, no la instrucción NL ambigua).

**Campos a serializar:** `attributes` + `goal_options` + `price_upper` + opcionalmente `product_category`. **NO** incluir `asin` ni `name` (leakage upper bound, reservado para ablation).

**Template propuesto (estilo OPSD, decisión C.3):**

```
[GOLDEN SPEC — for teacher only, do not surface to the user]
Target product specification:
- Attributes: {brand:Samsung, color:black, capacity:256GB}
- Options: {size: medium, color: red}
- Price upper bound: $50.0
- Category: Electronics > Audio > Headphones
Use this specification as a reference when planning your next action.
[/GOLDEN SPEC]

You are an expert autonomous agent operating in the WebShop e-commerce environment.
Your task is to: {task_description}.
...
```

**Dónde se inyecta:**

- **Punto de inyección — Opción A (recomendada): en `compute_piar_step_reward` momento de construir el `teacher_batch`.** No tocar el rollout. La inyección sucede SOLO durante el cálculo del log-ratio (numerador). Garantiza invariante 1 (rollout student sin golden) y mantiene el código de rollout intacto.
- **Punto de inyección — Opción B: en el `WebshopEnvironmentManager.build_text_obs`** con un flag `is_teacher_view`. Más invasivo, riesgo de filtración al rollout.

**Recomendado:** Opción A. ~15-25 LOC en una función helper `inject_golden_into_prompt(batch, golden_dict, tokenizer)` que:
1. Reconstruye el text del prompt original a partir de `batch.batch["prompts"]` decodificado.
2. Inserta el bloque `[GOLDEN SPEC ...]` al inicio del prompt (antes del system/user message del WebShop).
3. Re-tokeniza y arma `input_ids`/`attention_mask`/`position_ids` con el nuevo prompt + la response existente del student.
4. Devuelve un `DataProto` con la misma `responses` pero `input_ids` extendido.

### 3.3 Cuándo se inyecta

| Momento | ¿Con golden? | Justificación |
|---|---|---|
| Reset env / rollout student step | **NO** | Invariante 1: el student nunca ve el golden. Si lo viera, no aprende nada útil. |
| Forward `student_log_probs` en `compute_piar_step_reward` | **NO** | Es el denominador del log-ratio. |
| Forward `teacher_log_probs` en `compute_piar_step_reward` | **SÍ** (mismo modelo) | Es el numerador. La asimetría vive solo acá. |
| Loss del PPO / advantage compute | **NO** | El advantage se computa sobre el rollout original sin golden. |

### 3.4 Logging para D.9 (shuffled-golden control, decisión cerrada como requisito)

Capturar en el batch `non_tensor_batch["golden_dict"]` el goal serializado del episodio, y en el manifest del experimento (`experiments/ENNN/manifest.yaml`) loggear si el golden usado es:
- `mode=correct` (default) — golden del propio episodio.
- `mode=shuffled` — golden de OTRO episodio del mismo batch (rotación). Esto activa D.9.

Implementación: ~5 LOC para shuffling en `compute_piar_step_reward` cuando `config.algorithm.piar.shuffle_golden=True`.

---

## Sección 4 · Riesgos / cosas a verificar al implementar

### 4.1 Acoplamiento del PRM con el resto del pipeline

| Riesgo | Verificación |
|---|---|
| `compute_dpo_accuracy` / `compute_dpo_abs_accuracy` se invocan en `rm_fsdp_workers.py:290-294, 332-336, 377-381` y emiten métricas que el logger espera. | Si bypaseamos `update_rm_eto` con `update_style=none`, esas métricas quedan vacías. **Acción:** verificar que `SwanLab/console` no rompa con `dpo_acc` ausente; si rompe, agregar dummy values o filtrar el dict (~3 LOC). |
| `core_istar.py` usa `data.batch["rm_scores"]` Y `data.non_tensor_batch["uid"]`, `["traj_uid"]`, `data.batch["response_mask"]`. | Mantener todos esos campos. `compute_piar_step_reward` no los toca. |
| `apply_invalid_action_penalty` (`ray_trainer.py:1272-1276`) modifica `rm_scores` (o `token_level_rewards`, hay que verificar). | Leer la implementación. Si suma penalty a `rm_scores`, está bien — PIAR no cambia el contrato. |
| Métrica `reward_model/raw_reward` y `reward_model/reward` (`dp_rm.py:264-265, 369-372`) usan `q.sum(dim=-1).mean()`. | Replicar en `compute_piar_step_reward` para que SwanLab no se queje. ~2 LOC. |

### 4.2 FSDP / eficiencia de dos forward passes

| Riesgo | Verificación |
|---|---|
| `actor_rollout_wg.compute_log_prob` necesita cargar el actor en GPU. Llamarlo 2× (student + teacher) duplica el load/offload si `param_offload=True`. | Con `param_offload=False` (default en `run_webshop.sh:37`) los pesos quedan residentes — costo es solo 2× forward, sin re-load. Mantener `param_offload=False`. **Verificar también:** en `run_webshop.sh:49`, `actor_rollout_ref.ref.fsdp_config.param_offload=True` para el ref policy — eso es de un modelo distinto, no afecta. |
| El teacher batch tiene `input_ids` más largo (prompt + golden + response) que el student batch (prompt + response). `max_prompt_length=4096` (`run_webshop.sh:25`). Si el goal serializado supera ~200-400 tokens, podemos pasarnos del límite y truncar trayectorias. | **Acción:** loggear distribución de longitudes del golden serializado y del prompt combinado. Si pasa 4096, subir `max_prompt_length` (ya está en 4096; quizás 5120 alcance). |
| Tensor parallelism del actor (`tensor_model_parallel_size=1` en `run_webshop.sh:40`). Si se sube TP, las dos forward passes secuenciales no se aceleran. | OK con TP=1 por ahora. Si escalamos a TP>1 para 70B, evaluar batching del teacher forward para amortizar setup cost. |
| Memoria: dos forward residentes en bf16. Qwen2.5-7B + actor + (rm bypassed) + ref policy + vLLM rollout cache. | El RM bypass (no entrenar) libera memoria. `gpu_memory_utilization=0.6` (`run_webshop.sh:42`) deja headroom para vLLM. **Verificar con `nvidia-smi` en el primer run.** |

### 4.3 Inyección del golden — vía env o via dataset offline

| Opción | Pros | Contras |
|---|---|---|
| **Vía env (Opción A propuesta)**: getter `get_current_goal` en cada step, propagado a `non_tensor_batch["golden_dict"]` en el rollout loop. | Goal siempre sincronizado con el episodio real. D.9 (shuffled) se implementa shuffleando el atributo en el batch. | ~25-40 LOC en `envs.py` + `env_manager.py` + `rollout_loop.py` (⚠️ §6.3 — más plumbing que lo estimado originalmente). |
| **Vía dataset offline**: pre-extraer golden por `session_idx` en preprocessing y joinear al batch. | Cero cambios al env durante rollout. | Requiere modificar `examples/data_preprocess/prepare.py` y el dataset parquet. Más fricción, peor para iteración rápida. |

**Recomendado:** Vía env. **~25-40 LOC totales** (revisado post-Codex, ver §3.1 + §6.3) para propagar el golden_dict por episodio hasta el batch.

### 4.6 Riesgos nuevos identificados post-review Codex (2026-05-12)

| Riesgo | Severidad | Mitigación |
|---|---|---|
| **Sync actor/RM rompe invariante 4** — si se elige Opción A de §2.2 (mantener `reward_module`), el numerador queda en `π_init` en vez de `π_old` después del primer PPO update. Mezcla "info que aporta el golden" con "weight-drift inicial→actual". **Esto reintroduce el problema que C.2 cerró el 2026-05-11.** | 🔴 Crítico (rompe invariante) | Ir directo a Opción B de §2.2 (usar el actor para los dos forward passes). NO mantener el RM worker como teacher. Ver §6.2. |
| **`update=none` carga el RM igual** — `main_ppo` instancia `Role.RewardModel`, `ray_trainer` llama `rm_wg.init_model()`, `rm_fsdp_workers.py:115-265` carga modelo + optimizer + scheduler + checkpoint manager. Solo se evita `optimizer.step()`. Costo: ~14 GB VRAM extra ocupados por un Qwen2.5-7B duplicado e inútil. | 🟠 Alto (memoria + plata en spot) | Con Opción B de §2.2 se bypasea natural: el RM worker no se usa. Si por error queda activo: agregar guard en `main_ppo.py` para no instanciar `Role.RewardModel` cuando `update=none` (~10-15 LOC). Ver §6.1. |
| **Crash latente `ref_module=None`** — `dp_rm.py:226-229` hace `self.ref_module.eval()` sin guard. Con `update=none` ese path no se invoca, pero es una mina si se cambia el flow. | 🟡 Verificable (no bloquea Opción B) | Si Opción B: irrelevante (`dp_rm.py` queda muerto). Si por algún motivo se invoca: agregar `if self.ref_module is not None:` antes de la línea 226 (~1 LOC). Ver §6.1. |
| **Token-flow del `rm_scores` puede premiar el `<think>` privilegiado** — `dp_rm.py` calcula log-ratio sobre **todos los tokens de la respuesta** y después suma al final. Si la respuesta es `<think>El golden dice X, busco X</think>Acción: X`, el reward acumula sobre el reasoning que solo tiene sentido con golden en contexto — puede empujar al policy a optimizar razonamiento condicionado al golden, no la decisión de acción. Cruza con Apuesta A de [`piar-delta.md`](piar-delta.md) §4.1. | 🟠 Alto (riesgo científico) | Loggear contribución del log-ratio por tipo de token: think vs action vs other. Si la masa del reward está en `<think>` y no en `Acción:`, considerar (a) computar `rm_scores` solo sobre el span de la acción ReAct (no el `<think>`), o (b) reportar como hallazgo independiente. Ver §6.5. |

### 4.4 Otras cosas a verificar

- **`compute_log_prob` del actor usa `temperature`** (`fsdp_workers.py:682`). Verificar que para el teacher pass se use el mismo temperature que el student (sin scaling extra). Importante para que la fórmula del log-ratio sea consistente con la teoría de Yuan.
- **`use_fused_kernels`** (`dp_rm.py:46-49`): si está activo, `output.log_probs` viene pre-computado por el kernel — verificar consistencia con `compute_log_prob` del actor (que NO usa fused necesariamente). Si difieren, los logprobs del teacher (via reward_module + fused) vs los del student (via actor + non-fused) pueden tener mismatch numérico. **Acción:** usar el mismo path para ambos (preferiblemente `actor.compute_log_prob` para los dos).
- **`response_mask`** (computado en `ray_trainer.py:1160`): la agregación step-level usa `max_positions = response_mask.sum(-1)` por sample. Confirmar que el teacher batch (con golden más largo en el prompt) **NO** cambia la response_mask. La response sigue siendo idéntica byte-a-byte; solo el prompt crece.
- **Multi-turn:** el batch que llega al RM tiene una fila por **step de cada trayectoria** (`rollout_loop.py:368-369`: `total_batch_list[i].append(batch_list[i])`). Cada fila tiene su propio prompt (ya con history acumulada). El golden se inyecta a TODOS los steps de la trayectoria — mismo goal por episodio. **No re-extraerlo en cada step**; cachearlo por `traj_uid`.

### 4.5 Cosas que NO se rompen (verificadas durante el trace)

- `core_istar.py` consume solo `rm_scores` + `token_level_rewards` + `response_mask` + `uid/traj_uid`. Mientras mantengamos shape y semántica, intocable.
- `apply_invalid_action_penalty` y `apply_kl_penalty` se aplican sobre `token_level_scores` (outcome) — no sobre `rm_scores`. PIAR no las toca.
- El rollout loop (`rollout_loop.py`) es agnostic al PRM — no se entera del cambio.
- El reward_manager (`episode.py`) computa solo outcome reward — no se entera del cambio.

---

## Sección 5 · Estimación de líneas tocadas + roadmap de implementación

### 5.1 Conteo de LOC por archivo (revisado post-Codex 2026-05-12)

> ⚠️ La versión original de esta tabla decía 40-65 LOC in-place o 80-160 LOC con archivo nuevo. Codex marcó esa estimación como "optimista al borde de falso". Tabla corregida abajo. Ver §6.4.

| Archivo | Cambios | LOC est. | Tipo |
|---|---|---|---|
| `code/examples/istar_trainer/run_webshop.sh` | `update=after` → `update=none`. Agregar `algorithm.piar.shuffle_golden=False`, `algorithm.piar.golden_template=opsd`. | 3-5 | config |
| `code/verl/trainer/main_ppo.py` | Guard para no instanciar `Role.RewardModel` cuando `update=none` (evitar 14 GB VRAM desperdiciados, §6.1). | 10-15 | extensión |
| `code/verl/trainer/ppo/ray_trainer.py:1242-1268` | Reemplazar dispatch por llamada a `compute_piar_step_reward`. | 20-25 net | reemplazo |
| `code/istar/piar_step_reward.py` (nuevo) | Función `compute_piar_step_reward` + helper `inject_golden_into_prompt` + token-flow logging (§6.5). | 80-120 | nuevo archivo |
| `code/agent_system/environments/env_package/webshop/envs.py:60-72` | Extender handler con `'get_current_goal'` + lógica multi-process en clase `WebshopMultiProcessEnv` (§3.1 + §6.3). | 16-20 | extensión |
| `code/agent_system/environments/env_manager.py` (`WebshopEnvironmentManager`) | Atributo `current_goals` actualizado en `reset` + `step` + propagación al `info` (§6.3). | 8-12 | extensión |
| `code/agent_system/multi_turn_rollout/rollout_loop.py:328-356` | Pasar `current_goals` al batch como `non_tensor_batch['golden_dict']`. Cachear por `traj_uid`. | 5-8 | extensión |
| `code/istar/rm_fsdp_workers.py` | Si se mantiene el RM worker (NO recomendado): guard `if self.ref_module is not None` en `dp_rm.py:226-229` (§6.1). Con Opción B: irrelevante. | 0-3 | conditional |
| `code/istar/dp_rm.py` | Dead code bajo Opción B. **Nada que tocar.** | 0 | sin cambios |
| `code/istar/core_istar.py` | **Nada.** | 0 | sin cambios |

**Total estimado revisado: 142-208 LOC**. Si querés ser optimista y skippear el guard en `main_ppo`, y minimizar el token-flow logging, podés bajar a ~120-160 LOC. **Honestidad: la promesa original "30-100 LOC" se sobrepasa**. La razón es que el plumbing real (env-side goal propagation + main_ppo guard + token-flow logging) tiene más fricción que lo estimado inicialmente.

**Por qué se sobrepasa el rango original**:
- Goal injection vía env: subestimado (15 LOC → 25-40 LOC real). Codex §6.3.
- Bypass real del RM worker: requiere editar `main_ppo.py`, no solo el config. §6.1.
- Token-flow logging para validar Apuesta A: 10-20 LOC extra para diagnosticar si el reward se concentra en `<think>` o en acción. §6.5.

**Lectura honesta**: el research sigue siendo defendible, pero **la promesa "30-100 líneas" del NOTICE.md y del primer mapeo no se cumple**. Estimación realista: **~150 LOC totales** para una implementación correcta y verificable. Actualizar NOTICE.md + design-decisions.md A.2 si se quiere mantener la promesa visible alineada con la realidad.

### 5.2 Roadmap de implementación (orden propuesto)

| Orden | Tarea | Bloqueante de | Tiempo est. |
|---|---|---|---|
| **0** | **Replicar baseline iStar end-to-end en WebShop** (decisión D.3). Confirmar que `run_webshop.sh` corre, loggea, converge a ≥iStar paper numbers. Sin esto, no hay piso de comparación. | TODO | 1-3 días de compute + debug |
| 1 | Loggear longitudes de spans de acción ReAct (decisión D.5). Decidir si activar length normalization desde el inicio. | Punto 5 | 0.5 día |
| 2 | Implementar getter `get_current_goal` en envs + propagación al batch (paso "env-side wiring"). Verificar que el goal correcto baja por sample. | Punto 4 | 0.5-1 día |
| 3 | Implementar `inject_golden_into_prompt` y testear standalone: dado un prompt + un dict de goal, producir el input_ids del teacher con el bloque [GOLDEN SPEC] insertado. Confirmar token count razonable. | Punto 4 | 0.5 día |
| 4 | Implementar `compute_piar_step_reward` y cablearlo en `ray_trainer.py`. Cambiar `run_webshop.sh:56` a `update=none`. Correr 1 step y verificar shapes + valores razonables del log-ratio (no NaN, no todo cero). | Punto 5 | 1 día |
| 5 | **Smoke run**: 1 epoch completo. Verificar que (a) `rm_scores` no NaN, (b) `dpo_acc` métrica no rompe el logger, (c) memoria estable, (d) advantage compute funciona, (e) gradient flow al policy. | Punto 6 | 1-2 días |
| 6 | Full training run. Comparar contra baseline iStar (paso 0). | Punto 7 | 3-5 días compute |
| 7 | **D.9 — shuffled-golden control** (decisión cerrada como requisito). Re-correr con `shuffle_golden=True` y comparar. Si shuffled ≥ correct, hay leakage masivo y abortar. | Punto 8 | 3-5 días compute |
| 8 | **D.1 — leakage en muestras pareadas** (decisión cerrada como requisito). Analizar trayectorias correctas que copian texto del golden vs las que razonan. | Producción del paper | 2-3 días análisis |

### 5.3 Riesgo residual de la estimación (revisado post-Codex)

Si al implementar aparece:
- **Mismatch de logprobs** entre `actor.compute_log_prob` y `reward_module` (por kernel fused vs non-fused, o ulysses sharding distinto): irrelevante bajo Opción B (un solo path para los dos términos). **Mitigación ya aplicada en la estimación.**
- **El env worker pipe no expone bien el goal** (es Flask subyacente, puede haber estado mutable raro entre steps): puede sumar 10-15 LOC de defensive coding. Total ~160-220 LOC.
- **Token-flow del reward concentrado en `<think>`**: si la diagnóstica de §6.5 confirma el problema, agregar action-span masking (~15-25 LOC) para que `rm_scores` capture solo los tokens de la acción real, no el reasoning privilegiado. Total ~175-240 LOC.
- **Chat template de Qwen2.5-Instruct rompe inyección inline del golden**: el formato `<|im_start|>system\n...<|im_end|>` puede forzar a meter el golden como system message separado en vez de inline. ~5-10 LOC de template handling.
- **La inyección textual rompe el chat template** (Qwen2.5-Instruct usa `<|im_start|>system\n...<|im_end|>` etc.): puede que el golden tenga que ir como system message separado, no inline en user. Sumar 5-10 LOC de template handling.

**Conclusión honesta:** 30-100 LOC se sostiene como **mínimo viable** (Opción A en todo). El path recomendado (Opción B con archivo nuevo) está en **80-160 LOC** — un poquito sobre el límite alto, pero estructuralmente mucho más limpio y trazable. Lucas decide.

---

## Apéndice — Referencia rápida de paths críticos

```
ENTRY:     code/examples/istar_trainer/run_webshop.sh:1-88
MAIN:      code/verl/trainer/main_ppo.py:32-186 (con import iStar RM en :133)
TRAINER:   code/verl/trainer/ppo/ray_trainer.py
  · loop:      :1089-1320
  · RM call:   :1242-1268
  · adv:       :1283-1290 (calls core_istar)
RM WORKER: code/istar/rm_fsdp_workers.py
  · init:      :236-265
  · score:     :267-304 (compute_rm_score)
  · update:    :306-393 (update_rm, update_rm_eto)
DP_RM:     code/istar/dp_rm.py
  · forward:   :53-208 (THE LOG-RATIO at :161)
  · score:     :226-267 (compute_rm_score, calls forward)
  · update:    :269-448 (update_rm, update_rm_eto, NOT NEEDED for PIAR)
CORE_ISTAR:code/istar/core_istar.py (CONSUMER ONLY, do not touch)
  · rloo:      :28-149
ROLLOUT:   code/agent_system/multi_turn_rollout/rollout_loop.py
  · loop:      :263-389
  · gather:    :200-261
ENV:       code/agent_system/environments
  · base:      base.py:19-156
  · manager:   env_manager.py:306-450 (WebshopEnvironmentManager)
  · envs:      env_package/webshop/envs.py:9-247 (worker pipe + multi-process)
  · goal:      env_package/webshop/webshop/web_agent_site/engine/goal.py:16-66
  · sim:       env_package/webshop/webshop/web_agent_site/envs/web_agent_text_env.py:282-519
PROMPT:    code/agent_system/environments/prompts/webshop.py:1-29
REWARD:    code/agent_system/reward_manager/episode.py:23-122 (outcome ONLY, not PRM)
ACTOR LP:  code/verl/workers/fsdp_workers.py:665-705 (compute_log_prob — reusable for PIAR)
```

---

## Sección 6 · Adendum post-review Codex (2026-05-12)

> **Contexto:** la primera versión de este doc (commit `4f3505a`) fue revisada críticamente por Codex (`gpt-5.4`) el 2026-05-12. El review identificó 5 hallazgos relevantes que cambian el framing en partes específicas del doc. Esta sección es la **fuente de verdad** para esas correcciones. Cada hallazgo lleva un marker ⚠️ inline en el cuerpo del doc apuntando acá.

### 6.1 `update=none` no salta el init del RM worker

**Lo que decía la v1:** cambiar `update=after` → `update=none` en `run_webshop.sh:56` bypassea el training del PRM (1 línea de config).

**Lo que es cierto:** `update=none` evita `optimizer.step()` y `update_rm_eto`, pero **NO evita la instanciación del RM worker**. `main_ppo.py:133-134` igual importa `ISTARRewardModelWorker` y lo registra como `Role.RewardModel`. `ray_trainer.py` igual llama `rm_wg.init_model()`. `rm_fsdp_workers.py:115-265` igual:

- Carga `reward_module = AutoModelForCausalLM.from_pretrained(config.model.path, ...)` → **+14 GB VRAM** (Qwen2.5-7B duplicado).
- Carga `reward_optimizer = optim.AdamW(...)` → memoria de Adam states (~2× params en bf16).
- Carga `lr_scheduler` y `checkpoint_manager` → overhead administrativo.

**Implicación:** si dejamos `update=none` sin más, estamos cargando un Qwen2.5-7B inútil + optimizer states en VRAM. En `Standard_NC80adis_H100_v5` (2×H100 NVL = 192 GB) tenemos headroom, pero igual es desperdicio + se come KV cache de vLLM.

**Mitigación correcta:** con Opción B de §2.2 (usar el actor), el RM worker entero deja de tener sentido. Bypass desde `main_ppo.py`: guard `if config.reward_model.enable and config.reward_model.model.update != "none":` antes de registrar `Role.RewardModel`. ~10-15 LOC.

**Adicionalmente:** `dp_rm.py:226-229` hace `self.ref_module.eval()` sin guard. Con `ref_path=null` y `update=none`, ese codepath no debería invocarse (porque `compute_rm_score` no se llama). Pero si por accidente se entra: crash `NoneType.eval()`. Guard de 1 LOC `if self.ref_module is not None:` para safety.

### 6.2 Opción A de §2.2 (mantener RM worker como teacher) es INVÁLIDA

**Lo que decía la v1:** dos sub-opciones de §2.2 — Opción A (sync state_dict del actor al RM antes de cada step, ~15-25 LOC) y Opción B (usar el actor directamente, recomendada).

**Lo que es cierto:** Opción A **rompe el invariante 4** (mismos pesos del student) si se implementa sin sync explícito. El `reward_module` se carga UNA VEZ desde `model.path` (`rm_fsdp_workers.py:115-126`) y **queda desacoplado** del actor que PPO sigue actualizando. Después del primer update:

- Actor: `θ_t` (snapshot reciente, `π_old` por definición operacional).
- `reward_module`: `θ_0` (snapshot inicial, congelado por accidente).

El log-ratio computado: `log[π_θ₀(a|s, golden)] - log[π_θ_t(a|s)]`. **Esto mezcla "efecto del contexto golden" con "weight drift desde el SFT inicial"**. Es exactamente el problema que la decisión C.2 (2026-05-11) cerró al pasar de "frozen θ₀" a "`π_old` snapshot reciente". Mantener el RM worker estático reintroduce el bug en la implementación.

**Si querés rescatar Opción A**, habría que agregar sync explícito `actor.state_dict() → reward_module.load_state_dict()` antes de cada `compute_rm_score`. Eso:
- Suma ~15-25 LOC.
- Mantiene dos modelos en VRAM (no ahorra memoria vs Opción B).
- Suma overhead de copia per-step (bandwidth NVMe/PCIe si los modelos están sharded).
- Es estrictamente peor que Opción B en todas las dimensiones.

**Veredicto:** ❌ Opción A es inválida. **Solo Opción B (usar `actor_rollout_wg.compute_log_prob` para los dos forward passes) preserva invariante 4 y es eficiente.** El doc en su v1 las presentaba como alternativas equivalentes "preferir B" — pero la realidad es que A está mal, no es solo menos elegante.

### 6.3 Goal injection requiere 25-40 LOC, no 10-15

**Lo que decía la v1:** ~10-15 LOC para propagar el `golden_dict` desde `web_agent_text_env.py:519-521` al `non_tensor_batch`.

**Lo que es cierto:** el plumbing tiene más fricción del que reportó la v1.

- El getter actual `'get_goals'` en `envs.py:71-72` devuelve la **lista global** de todos los goals del dataset (`env.server.goals`), no el goal del episodio actual. Hay que agregar un getter nuevo `'get_current_goal'` que lea `env.server.user_sessions[session_id]['goal']`.
- `WebshopMultiProcessEnv` corre los envs en subprocesos via worker pipe. El método nuevo tiene que recolectar de TODOS los subprocesos (no solo uno).
- `WebshopEnvironmentManager` (`env_manager.py:306-450`) no propaga el goal en `info` del `step` ni del `reset`. Hay que agregar atributo `self.current_goals: list[dict]` actualizado en ambos.
- `rollout_loop.py:328-356` no expone goal al batch — hay que pasar `current_goals` como `non_tensor_batch['golden_dict']`.

**Total real:** 25-40 LOC distribuidas en 4 archivos. No es bloqueante ni difícil, pero la estimación original era plana.

### 6.4 LOC estimate revisado: 100-180 (probable 150)

**Lo que decía la v1:** 40-65 LOC in-place o 80-160 LOC con archivo nuevo. "Cumple promesa 30-100 estrictamente".

**Lo que es cierto:** revisada con los items 6.1 + 6.2 + 6.3, la estimación honesta es **142-208 LOC totales** (ver tabla §5.1 actualizada). Si se buscan corners cuts (skippear el guard en `main_ppo`, minimizar token-flow logging): 120-160 LOC. **La promesa "30-100 LOC" del NOTICE.md y design-decisions.md A.2 no se cumple — actualizar esos docs.**

**Por qué se sobrepasa el rango original:**
- Goal injection real: +15-25 LOC sobre el estimate (§6.3).
- Bypass del RM worker desde `main_ppo`: +10-15 LOC nuevas (§6.1).
- Token-flow logging para Apuesta A: +10-20 LOC (§6.5).
- LOC del archivo nuevo `piar_step_reward.py` más realista: 80-120 (no 60-90), porque incluye logging + handler de chat template + cachear golden por traj_uid.

**Honestidad para el paper:** "30-100 LOC" se sostiene como **mínimo viable** si aceptamos un PoC con corners cuts y sin logging diagnóstico. Para la versión **publicable** (con verificación de Apuesta A + bypass limpio del RM worker), la cifra honesta es **~150 LOC**. Esto no invalida el research — sigue siendo una modificación quirúrgica de un orden de magnitud menor que reimplementar iStar — pero la promesa fuerte "30-100" requiere asterisco.

### 6.5 Token-flow del `rm_scores` cruza con Apuesta A

**Lo que decía la v1:** la decisión B.3 (action-level granularity) está cerrada porque `dp_rm.py:199-204` colapsa el reward a la última posición válida del step.

**Lo que es cierto:** el reward **se compone token por token** sobre toda la respuesta (`q = teacher_log_probs - student_log_probs` en `dp_rm.py:161` produce un tensor `(bs, response_length)`), y después se **agrega via suma** al asignarlo a `max_positions[i] - 1`. Es decir: cada token de la respuesta contribuye linealmente al reward final del step.

**Implicación científica:** si la respuesta del agente tiene estructura `<think>...razonamiento...</think>Acción: X`, el log-ratio acumula sobre TODOS los tokens del `<think>` + la acción. Como el `<think>` puede contener referencias explícitas al golden ("El target dice Samsung, busco Samsung"), el reward puede **premiar el razonamiento condicionado al golden** más que la decisión de acción.

**Conexión con Apuesta A (`piar-delta.md` §4.1):** la Apuesta A dice "action-level granularity gana sobre token-level". Pero como está implementado, **el actual "action-level" es realmente "todos los tokens del response colapsados al final"** — semánticamente token-level con post-agregación. La granularidad estricta action-level requeriría masking del `<think>` para que solo los tokens de la acción real contribuyan al reward.

**Acción propuesta**:
1. **Diagnóstico primero**: durante el smoke run, loggear contribución del log-ratio por tipo de token (think / action / other) — qué % de la masa del reward viene de cada categoría. ~10-20 LOC en `compute_piar_step_reward`.
2. **Si el reward se concentra en `<think>`**: agregar masking de `<think>` tokens en el agregado step-level (~15-25 LOC). Esto cambia la implementación operacional de Apuesta A.
3. **Si el reward se concentra en `Acción:`** (poco probable, pero posible): el setup actual ya valida B.3 sin más cambios. Documentar.

**Cruza con D.1 (`design-decisions.md`)** — el análisis de leakage textual incluiría ahora dos dimensiones: (a) leakage en la decisión de acción (¿copia text del golden literalmente?), (b) leakage en el reward signal (¿la masa del log-ratio premia el razonamiento o la acción?).

### 6.6 Veredicto general de Codex

Textual del review:

> *"La dirección general es sólida. La versión 'config-only + input replacement' no lo es."*

**Lectura:** el método PIAR es defendible. La implementación requiere más cuidado del que la v1 del mapeo reflejaba. Las correcciones 6.1-6.5 alinean el plan operacional con la realidad del codebase.

**Lo que NO cambia post-Codex:**
- `dp_rm.py:161` sigue siendo el log-ratio framework — PIAR sigue siendo "reemplazo de inputs" en spirit.
- `core_istar.py` sigue intocable.
- La arquitectura general (actor con dos forward passes + golden injection vía env) es correcta.
- El roadmap de §5.2 sigue siendo válido.

**Lo que SÍ cambia:**
- Opción A de §2.2 marcada como inválida.
- LOC estimate de 40-65 → ~150.
- Plumbing del golden subestimado (10-15 → 25-40).
- Riesgo nuevo identificado: token-flow puede premiar `<think>` privilegiado.
- Bypass real del RM worker requiere `main_ppo.py` edit, no solo config.

### 6.7 Próximos pasos sugeridos para fase 4

Actualizar antes de arrancar implementación:
1. **`NOTICE.md`** — cambiar "modificación esperada chica (~30-100 líneas)" a "~150 líneas".
2. **`design-decisions.md` A.2** — alinear LOC estimate.
3. **Issue #16** — agregar criterio de cierre: además de replicar baseline iStar, validar que `update=none` con el bypass del RM worker no rompa el smoke run.
4. **Nuevo issue D.10 (candidato)** — token-flow analysis como prerequisito de Apuesta A. Cierra una pregunta científica antes de gastar GPU-días.
