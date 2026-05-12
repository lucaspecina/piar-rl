# PIAR — Puntos de implementación quirúrgica sobre `code/`

> **Qué es esto:** mapeo end-to-end del pipeline de iStar vendoreado en `code/` con los puntos exactos a modificar para llegar a PIAR (~30-100 LOC). Cada referencia es `archivo:línea`.
>
> **Qué NO es:** no es PR todavía, no es código pegable. Es el plano para que el implementador (humano o agente) pueda atacar la fase 4 sin re-explorar el repo.
>
> **Pre-requisito de lectura:** `PROJECT.md` (invariantes 1-5), `research/synthesis/design-decisions.md` (especialmente A.2, B.1-B.7, C.1-C.5, D.9).
>
> **Decisión arquitectónica central que justifica esta estrategia:** `dp_rm.py:161` ya computa `q = rm_log_labels - ref_log_labels`. El framework de iStar es estructuralmente un log-ratio — solo cambia **qué modelo y qué prompt** alimentan cada término. PIAR es un reemplazo de inputs, no un rewrite.

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

**Propuesto:** dos opciones equivalentes (preferir A por mínimo cambio de código):

**Opción A — config-only**: cambiar en `code/examples/istar_trainer/run_webshop.sh:56` el flag `reward_model.model.update=after` → `reward_model.model.update=none`. El branch `update_style == "none"` (`ray_trainer.py:1244-1245`) ya llama `compute_rm_score` puro sin DPO update. **Cero líneas de código tocado en `ray_trainer.py`.** Eliminación del training del PRM sin perder el path de scoring.

**Opción B — code-level**: agregar `elif update_style == "piar":` que llame un nuevo `compute_piar_rm_score`. Más explícito pero más LOC.

**Justificación:** PIAR no entrena el numerador. El mismo modelo `π_old` se usa con y sin golden — el log-ratio mide solo diferencia de contexto. Recomendado: **Opción A**.

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

**Propuesto:** con Opción A de 2.1.1, el optimizer queda construido pero **nunca se llama** (porque ni `update_rm` ni `update_rm_eto` se invocan). Cero líneas a tocar. Si se quiere ser estrictos y ahorrar memoria: agregar guard `if config.model.get("update", "none") == "none":` y skipear `reward_optimizer` + `reward_lr_scheduler` (~3-5 LOC). **Recomendado: no tocar todavía** — sumar guard solo si la memoria es un cuello de botella real.

### 2.2 Reescritura del log-ratio: PRM frozen → snapshot policy con golden

#### Cambio 2.2.1 — `rm_fsdp_workers.py:80-234` (modelo cargado en RM worker)

**Actual:** `_build_reward_ref_model_optimizer` carga `reward_module` desde `config.model.path` (línea 89, 120-126), que es **el mismo path que el policy** (`run_webshop.sh:11+54`: `reward_model.model.path=$model_path`).

**Propuesto:** mantener la carga inicial (el modelo ES el mismo). Lo crítico es que **antes de cada llamada a `compute_rm_score`** se **sincronicen los pesos** del `reward_module` con los del `actor_rollout_wg` (snapshot reciente `π_old`). Hay dos sub-opciones:

**Opción A — sync explícito de pesos del actor al RM worker** antes de cada `compute_rm_score`. Requiere agregar un `@register` method que copie state_dict del actor al RM. ~15-25 LOC nuevas en `rm_fsdp_workers.py` + ~3-5 LOC en `ray_trainer.py` que invoca el sync.

**Opción B (recomendada) — eliminar `reward_module` y reusar el actor.** En vez de tener dos modelos en GPU, usar `self.actor_rollout_wg.compute_log_prob(batch_with_golden)` directamente (existe ya en `code/verl/workers/fsdp_workers.py:665-705`). Esto pone toda la lógica de PIAR fuera del `ISTARRewardModelWorker`. Ver Cambio 2.2.3.

#### Cambio 2.2.2 — `dp_rm.py:53-208` (`_forward_micro_batch`)

**Actual:** computa `rm_log_labels` con `self.reward_module(input_ids=micro_batch["input_ids"], ...)` (línea 101-106) y `ref_log_labels = micro_batch["old_log_probs"]` (línea 158, default cuando `ref_module is None`).

**Sub-opciones según 2.2.1:**

- **Si Opción A de 2.2.1:** la firma de `_forward_micro_batch` no cambia, pero el `input_ids` del micro_batch que llega ahora **incluye el golden inyectado** (ver Sección 3). El log-ratio `q = rm_log_labels - ref_log_labels` (`dp_rm.py:161`) pasa a ser exactamente PIAR: `log[π_old(a|s, golden)] - log[π_old(a|s)]`. **Cero líneas tocadas en `dp_rm.py`** si el batch ya llega con el input correcto. La inyección del golden se hace **antes** de pasar el batch al RM worker.

- **Si Opción B (preferida):** se reemplaza la llamada `self.rm_wg.compute_rm_score(batch)` por un nuevo método `compute_piar_step_reward` (~30-50 LOC) que internamente hace dos llamadas a `actor_rollout_wg.compute_log_prob` — una con prompt + golden, otra con prompt sin golden — y computa `q = teacher_log_probs - student_log_probs` con el mismo agregado step-level que `dp_rm.py:199-204`. Ver Cambio 2.2.3 abajo.

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

**Acceso disponible:** el worker pipe ya expone el getter `'get_goals'` (`envs.py:71-72` devuelve `env.server.goals`). Hace falta UN getter nuevo `'get_current_goal'` que devuelva `env.server.user_sessions[env.session]['goal']` para acceder al goal del episodio actual (no la lista completa). ~5 LOC agregadas en `envs.py:71-72` (extender el handler) + ~3 LOC en `WebshopMultiProcessEnv` para exponer el método.

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
| **Vía env (Opción A propuesta)**: getter `get_current_goal` en cada step, propagado a `non_tensor_batch["golden_dict"]` en el rollout loop. | Goal siempre sincronizado con el episodio real. D.9 (shuffled) se implementa shuffleando el atributo en el batch. | ~10 LOC en `envs.py` + `env_manager.py` + `rollout_loop.py` para propagar. |
| **Vía dataset offline**: pre-extraer golden por `session_idx` en preprocessing y joinear al batch. | Cero cambios al env durante rollout. | Requiere modificar `examples/data_preprocess/prepare.py` y el dataset parquet. Más fricción, peor para iteración rápida. |

**Recomendado:** Vía env. ~10-15 LOC totales para propagar el golden_dict por episodio hasta el batch.

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

### 5.1 Conteo de LOC por archivo

| Archivo | Cambios | LOC est. | Tipo |
|---|---|---|---|
| `code/examples/istar_trainer/run_webshop.sh` | `update=after` → `update=none`. Opcional: agregar `algorithm.piar.shuffle_golden=False`, `algorithm.piar.golden_template=opsd`. | 2-5 | config |
| `code/verl/trainer/ppo/ray_trainer.py:1242-1268` | Si Opción B (recomendada): reemplazar dispatch por llamada a `compute_piar_step_reward`. | 20-25 net (delta) | reemplazo |
| `code/istar/piar_step_reward.py` (nuevo) | Función `compute_piar_step_reward` + helper `inject_golden_into_prompt`. | 60-90 | nuevo archivo |
| `code/agent_system/environments/env_package/webshop/envs.py:71-72` | Extender handler con `'get_current_goal'`. | 5-8 | extensión |
| `code/agent_system/environments/env_package/webshop/envs.py` (clase `WebshopMultiProcessEnv`) | Método `get_current_goals()` que devuelve list[dict] uno por sub-env. | 5-10 | extensión |
| `code/agent_system/environments/env_manager.py` (`WebshopEnvironmentManager`) | Propagar `goals` desde reset/step a infos o atributo `self.current_goals`. | 5-10 | extensión |
| `code/agent_system/multi_turn_rollout/rollout_loop.py:336-339` | Agregar `batch.non_tensor_batch['golden_dict'] = envs.current_goals` por step (cachear por trayectoria). | 5-8 | extensión |
| `code/istar/rm_fsdp_workers.py` | **Opcional**: guard para skip optimizer cuando `update=none`. **No tocar** en primera iteración. | 0-5 | opcional |
| `code/istar/dp_rm.py` | **Nada** si Opción B. La función entera queda como dead code accesible via `update=after` legacy. | 0 | sin cambios |
| `code/istar/core_istar.py` | **Nada.** | 0 | sin cambios |

**Total estimado: ~100-160 LOC** considerando el archivo nuevo (`piar_step_reward.py` con docstrings y helpers). 

**Si excluimos el nuevo archivo y contamos solo deltas en archivos existentes: ~40-65 LOC**, dentro del rango 30-100 prometido.

**Honestidad sobre el rango 30-100:** se cumple **estrictamente** si contamos solo deltas en archivos existentes (40-65 LOC). El archivo nuevo `piar_step_reward.py` agrega ~60-90 LOC adicionales pero es código aislado, fácil de revisar, no entrelazado con el framework. Si se quisiera meter inline en `ray_trainer.py` se llegaría a 90-110 LOC totales — al límite alto pero defendible. **Recomendación: archivo nuevo separado, por trazabilidad.**

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

### 5.3 Riesgo residual de la estimación

Si al implementar aparece:
- **Mismatch de logprobs** entre actor.compute_log_prob y reward_module (por kernel fused vs non-fused, o ulysses sharding distinto): puede sumar 20-40 LOC para forzar consistencia. Total subiría a ~100-200 LOC. **Mitigación:** usar SIEMPRE `actor_rollout_wg.compute_log_prob` para los dos términos (Opción B 2.2.1).
- **El env worker pipe no expone bien el goal** (es Flask subyacente, puede haber estado mutable raro entre steps): puede sumar 10-15 LOC de defensive coding. Total ~120-180 LOC.
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
