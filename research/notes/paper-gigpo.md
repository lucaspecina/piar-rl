# GiGPO — Group-in-Group Policy Optimization for LLM Agent Training

> **Issue:** epic Pivot 2026-06 (ver Project v2) · **arxiv:** [2505.10978](https://arxiv.org/abs/2505.10978) · **Código:** [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent) (veRL-based) · **Autores:** Lang Feng, Zhenghai Xue, Tingcong Liu, Bo An (afiliación NTU Singapore ⚠️ no verificada en el fetch). v1 mayo 2025, v3 octubre 2025. **NeurIPS 2025.**
> **Rol en PIAR:** **baseline obligatorio B2 (decisión N.10 del pivot 2026-06)** — SOTA critic-free en ALFWorld/WebShop. Es el representante de la familia "estadística entre rollouts": señal densa por paso derivada SOLO de agrupar acciones tomadas desde el mismo estado entre rollouts, sin juez, sin información privilegiada, sin modelos extra. Si PIAR no supera a GiGPO, la PI no aporta nada sobre estadística pura. Bonus operativo: el fork vendoreado en `code/` ya trae un trainer GiGPO ejecutable.

## 1. Setup MDP y problema que ataca

Multi-turn agentic RL: el agente LLM interactúa con un environment durante $T$ pasos (ALFWorld hasta 50, WebShop hasta 15), reward mayormente sparse al final. El problema es el clásico del pivot: GRPO/RLOO dan **un solo advantage por trayectoria** — todas las acciones de un episodio exitoso reciben el mismo crédito, incluyendo las inútiles o dañinas. PPO con critic da crédito por paso pero el value model en agentic long-horizon es inestable y caro.

La apuesta de GiGPO: obtener advantage **por paso** sin critic, sin reward model, sin rollouts extra — explotando que cuando $N$ rollouts parten de las mismas condiciones, **los mismos estados del environment reaparecen entre trayectorias**, y las acciones tomadas desde ese estado compartido se pueden comparar entre sí como un grupo GRPO "local".

## 2. Mecanismo: advantage jerárquico de dos niveles

### 2.1 Nivel episodio (macro)

Estándar GRPO sobre el grupo de $N$ trayectorias generadas desde condiciones idénticas (Eq. 3):

$$A^E(\tau_i) = \frac{R(\tau_i) - \text{mean}(\{R(\tau_j)\}_{j=1}^N)}{F_{\text{norm}}(\{R(\tau_j)\}_{j=1}^N)}$$

con $R(\tau_i)$ = retorno total de la trayectoria y $F_{\text{norm}} \in \{\text{std}, 1\}$ (la variante $F_{\text{norm}}=1$, "GiGPO w/o std", es estimador insesgado tipo leave-one-out y es la que gana en las tablas principales).

### 2.2 Nivel paso (micro): anchor-state grouping

La novedad. Después de generar los $N$ rollouts, se identifican retroactivamente los **estados únicos** $\mathcal{U} = \{\tilde{s}_1, \dots, \tilde{s}_U\}$ que aparecen repetidos en el grupo. Para cada anchor state $\tilde{s}$ se construye el grupo step-level (Eq. 6):

$$G^S(\tilde{s}) = \{(a_t^{(i)}, R_t^{(i)}) \mid s_t^{(i)} = \tilde{s},\ 1 \le i \le N,\ 1 \le t \le T\}$$

es decir: **todas las acciones tomadas desde ese mismo estado, en cualquier trayectoria del grupo y en cualquier timestep**, cada una con su return descontado (Eq. 5):

$$R_t^{(i)} = \sum_{k=t}^{T} \gamma^{k-t} r_k^{(i)}, \qquad \gamma = 0.95$$

El advantage step-level es GRPO dentro de ese grupo local (Eq. 7):

$$A^S(a_t^{(i)}) = \frac{R_t^{(i)} - \text{mean}(\{R_t^{(j)} \mid (a_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{s})\})}{F_{\text{norm}}(\cdot)}$$

**Lectura clave:** en environments de reward sparse (WebShop: reward solo al final), $R_t^{(i)} = \gamma^{T-t} r_T^{(i)}$ — o sea que el step-level advantage compara **cómo terminaron las trayectorias que pasaron por el mismo estado**, ponderado por cuánto tardaron en terminar. Es un estimador Monte Carlo del Q-value relativo de cada acción, gratis, sin ningún juez. El $\gamma < 1$ además penaliza implícitamente dar vueltas (dos trayectorias que pasan por el mismo estado y llegan al mismo outcome, la más corta tiene mayor $R_t$).

### 2.3 Combinación (Eq. 8)

$$A(a_t^{(i)}) = A^E(\tau_i) + \omega \cdot A^S(a_t^{(i)})$$

con $\omega = 1$ sin tuning en los experimentos principales. El objetivo final (Eq. 9) es el clipped surrogate estándar de PPO con este advantage por token (todos los tokens de la acción $a_t$ comparten $A(a_t^{(i)})$).

**Estructura idéntica al hook de PIAR:** suma de un término episode-level y un término step-level ponderado, exactamente la forma $A = A_{\text{outcome}} + \beta \cdot \text{steps}$ donde iStar (y PIAR) inyectan su step reward. La diferencia es de dónde sale el término step: estadística de estados repetidos (GiGPO) vs juez con privilegio en pesos (iStar) vs log-ratio/belief con PI en el prompt del scorer (PIAR).

## 3. Identificación de estados idénticos

- **Implementación base: matching exacto vía hashmap** — el anchor state es la key, las (acción, return) se agregan a una hash table. Costo reportado: 0.01s el grouping + 0.53s el cómputo de advantages por iteración, < 0.002% del tiempo total. Completamente offline post-rollout, cero overhead de GPU.
- **Variante por similitud** (apéndice E.1): agrupar si la similitud por longest matching subsequence supera umbral 0.9 — para environments con ruido superficial en las observaciones.
- Qué es "el estado": la observación del environment en ese turno (no el historial completo). ⚠️ El paper no especifica con precisión qué componentes de la observación entran a la key; el código vendoreado (ver §8) hashea `anchor_obs` con matching exacto.

## 4. La limitación estructural: estados que no se repiten

Este es el punto que define el rol de GiGPO como baseline (y su tabla de la §2 del pivot doc):

- Si un anchor state aparece **una sola vez** en el grupo, $|G^S(\tilde{s})| = 1$ y el advantage step-level de esa acción es trivial (no hay pares contra quién comparar).
- Cita textual del paper (limitations): *"In highly complex environments, identical states may be hard to detect due to noise or subtle differences. Despite this, GiGPO still retains a strong performance lower bound: in the extreme case where no states are repeated across trajectories (i.e., $A^S = 0$), it naturally degrades to GRPO."*
- O sea: **degradación elegante pero degradación al fin** — en el caso extremo GiGPO = GRPO, toda la señal densa desaparece.
- Empíricamente les funciona porque ALFWorld/WebShop tienen espacios de observación discretos y chicos: en ALFWorld (Figura 5) <35% de los estados quedan como singletons durante el training; >20% de los grupos tienen $|G^S| \ge 10$ en iteraciones tempranas, concentrándose en tamaños 6–8 hacia la iteración 140.
- Para estados continuos/estocásticos el paper solo ofrece la variante por similitud y deja "embedding-based representations or domain-specific structural equivalence" como trabajo futuro.

**Consecuencia conceptual:** la densidad de la señal de GiGPO depende de una propiedad del *environment* (tasa de revisita de estados entre rollouts), no del método. La señal de PIAR (log-ratio / belief sobre PI) no depende de revisitas — funciona igual en el primer y único rollout que pasa por un estado. Es la dimensión exacta en la que los dos enfoques se separan.

## 5. Setup experimental y resultados

**Modelos base:** Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct (principales); Qwen2.5-3B-Instruct para QA con búsqueda; Qwen2.5-VL-3B-Instruct para visión.

**Benchmarks:** ALFWorld (3.827 tareas, 6 categorías household, success rate), WebShop (1.1M productos, 12k instrucciones; score promedio + success rate), QA con search (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle), y VL (Sokoban 6×6, EZPoints).

**Tabla principal (GiGPO w/o std vs baselines):**

| Método | ALFWorld 1.5B | WebShop 1.5B (score / succ%) | ALFWorld 7B | WebShop 7B (score / succ%) |
|---|---|---|---|---|
| Prompting (Qwen2.5) | 4.1 | 23.1 / 5.2 | 14.8 | 26.4 / 7.8 |
| ReAct | 12.8 | 40.1 / 11.3 | 31.2 | 46.2 / 19.5 |
| Reflexion | 21.8 | 55.8 / 21.9 | 42.7 | 58.1 / 28.8 |
| PPO | 54.4±3.1 | 73.8±3.0 / 51.5±2.9 | 80.4±2.7 | 81.4±3.1 / 68.7±5.1 |
| RLOO | 69.7±2.5 | 73.9±5.6 / 52.1±6.7 | 75.5±4.6 | 80.3±3.2 / 65.7±4.0 |
| GRPO | 72.8±3.6 | 75.8±3.5 / 56.8±3.8 | 77.6±5.2 | 79.3±2.8 / 66.1±3.7 |
| **GiGPO w/o std** | **86.1±4.7** | **83.5±1.8 / 67.4±4.5** | **90.2±2.3** | **86.2±2.6 / 75.2±3.8** |

**Δ vs GRPO:** +13.3 (ALFWorld 1.5B), +12.6 (ALFWorld 7B), +10.6 / +9.1 puntos de success en WebShop. Consistente con el claim del abstract (">12% ALFWorld, >9% WebShop"). Notable: GiGPO 1.5B (86.1) supera a PPO 7B (80.4) en ALFWorld.

**QA con search (accuracy promedio 7 datasets):** GiGPO 3B = 42.1 (vs Search-R1 3B = 32.5, ZeroSearch 3B = 31.7); GiGPO 7B = 47.2. **VL:** Sokoban 81.0 vs GRPO 67.1; EZPoints 100.0 vs 86.9.

## 6. Hyperparams clave (paper, apéndice)

- **Group size $N = 8$** (ALFWorld/WebShop), 5 (QA). **128 environments paralelos** = 16 grupos × 8.
- Batch size: 256 (ALFWorld), 64 (WebShop), 512 (QA). LR actor = 1e-6 (critic 1e-5 en el baseline PPO).
- $\gamma = 0.95$ en todo; $\omega = 1$ sin tuning.
- Temperatura: 1.0 rollout / 0.4 validación (ALFWorld/WebShop).
- Max steps: 50 (ALFWorld), 15 (WebShop). Max prompt 2048/4096, max response 512.
- KL coefficient: 0.01 (ALFWorld/WebShop), 0.001 (QA) — ⚠️ contrasta con la línea implicit-PRM (Yuan/iStar/PRIME, KL=0); y el trainer vendoreado en `code/` lo corre con KL apagado (ver §8).
- Historial multi-turn: **truncado a las últimas 2 observaciones** en ALFWorld/WebShop; historial completo en QA.
- Hardware: 2×H100 (1.5B), 4×H100 (7B) — **el run de 7B del paper cabe holgado en nuestra VM 2×H100 NVL solo para 1.5B; el 7B habrá que ajustarlo** (gradient checkpointing + offload ya vienen activados en el script vendoreado).
- 150 iteraciones de training (ALFWorld/WebShop). Framework: veRL (su fork `verl-agent`).

## 7. Ablations

- **$\omega$ (Tabla 5, WebShop):** 0.0 → 76.2 score / 56.6 succ; **0.8 → 84.9 / 68.3 (óptimo)**; 1.0 → 83.5 / 67.4; 1.4 → 77.0 / 56.3. Robusto en [0.4, 1.2], colapsa afuera. $\omega = 0$ ≈ GRPO confirma que toda la ganancia viene del término step-level.
- **Estructura (Figura 4):** sacar $A^E$ → caída severa en todas las tareas; sacar $A^S$ → caídas pronunciadas sobre todo en tareas difíciles (Cool, Pick2, WebShop). Los dos niveles son necesarios.
- **Normalización:** w/ std vs w/o std es task-dependent, sin ganador universal; las tablas principales reportan w/o std.
- Sin ablation de $\gamma$ (fijo 0.95).

## 8. El baseline B2 ya está vendoreado en `code/`

El fork de CharacterRL-iStar en `code/` **ya incluye GiGPO ejecutable** — el brazo B2 viene casi gratis en el stack actual (verificado 2026-06-12, sin modificar nada):

- **`code/examples/gigpo_trainer/run_webshop.sh`** y **`run_sokoban.sh`** — scripts completos: `algorithm.adv_estimator=gigpo`, `group_size=8`, `gamma=0.95`, `step_advantage_w=1.0` ($\omega$), `mode=mean_norm` (= w/o std, la variante ganadora del paper; alternativa `mean_std_norm`), lr 1e-6, Qwen2.5-7B-Instruct, temperatura validación 0.4 — todo consistente con el paper.
- **`code/gigpo/core_gigpo.py`** — el algoritmo: `compute_step_discounted_returns` (Eq. 5), `build_step_group` (anchor grouping por matching **exacto** vía `to_hashable` sobre `anchor_obs` + hashmap; la variante por similitud 0.9 del apéndice no está en este fork ⚠️), `compute_gigpo_outcome_advantage` (Eqs. 3+7+8).
- **`code/verl/trainer/ppo/ray_trainer.py`** — integración: `AdvantageEstimator.GiGPO`, `anchor_obs` viaja en el `non_tensor_batch`, y el advantage combinado se computa en el mismo `compute_advantage` donde iStar inyecta el suyo.
- Diferencias del script vendoreado vs paper a tener en cuenta al replicar: `use_kl_loss=False` y `use_kl_in_reward=False` (paper reporta KL 0.01), `env.max_steps=10` en WebShop (paper: 15), `invalid_action_penalty_coef=0.1` (penalty por acción inválida que el paper no destaca), batch de train 16 prompts × 8 rollouts. Documentar la config exacta al correr B2.

## 9. Conexiones con otros papers

- **GRPO:** GiGPO es estrictamente GRPO + término step-level; con $\omega=0$ o sin estados repetidos colapsa a GRPO. Por eso es el baseline "estadística pura" perfecto: misma infraestructura, solo cambia el origen de la señal densa.
- **iStar (B1):** ambos producen un advantage = episodio + paso. iStar saca el término paso de un juez DPO con privilegio en los *pesos*; GiGPO lo saca de estadística de revisitas, **sin ningún privilegio**. Juntos cubren los dos extremos contra los que PIAR se mide.
- **PRIME (#6):** PRIME también combina outcome + step a nivel advantage (Eq. 5 de PRIME, LOO baseline), pero su step reward viene de un PRM entrenado online. GiGPO no entrena nada — comparte con PIAR el "cero parámetros extra", no comparte la fuente de señal.
- **Math-Shepherd / value MC:** el $R_t$ de GiGPO es un estimador Monte Carlo de Q sin rollouts extra — los "rollouts de continuación" son las otras trayectorias del grupo que pasaron por el mismo estado. Misma amortización que el backward de PIAR logra vía belief, pero GiGPO la consigue solo donde hay revisitas.
- **POAD, SLEA-RL:** misma familia (estadística entre rollouts) según el survey de credit assignment; GiGPO es el de mejores números publicados en nuestros benchmarks target.

## 10. Lo que esto significa PARA PIAR (post-pivot)

### 10.1 Por qué es baseline obligatorio (brazo B2, decisión N.10)

GiGPO responde la pregunta **"¿cuánto crédito por paso se puede extraer GRATIS, sin información privilegiada?"** — y la respuesta es: mucho (+9 a +13 puntos sobre GRPO en nuestros dos benchmarks target, con Qwen2.5-7B-Instruct, nuestro modelo base exacto). Eso fija la vara real:

- Si PIAR (dual ± residual) no supera a GiGPO en WebShop/ALFWorld, la conclusión honesta es que **la PI no aporta nada sobre estadística pura entre rollouts** — el costo de serializar golden/componentes g_i y los forward passes extra no se justifica. Brazo B2 es el test existencial del valor de la PI, así como C1 (shuffled-golden) es el test existencial del contenido de la PI.
- El piso A0 (GRPO outcome-only) ya no alcanza como única referencia: GiGPO demuestra que el gap entre outcome-only y "señal densa bien hecha" es grande, y un método nuevo tiene que ganarle al mejor extractor de señal densa sin privilegio, no al piso.

### 10.2 Complementariedad con iStar (B1) — los dos ejes del diseño

B1 y B2 aíslan variables distintas y juntos triangulan el claim:

| Baseline | Fuente de la señal densa | Qué privilegio usa | Qué prueba si PIAR le gana |
|---|---|---|---|
| B1 iStar | Juez entrenado (DPO trayectorial) | En los **pesos** del juez | La PI en el prompt ≥ PI destilada en pesos (sin entrenar nada) |
| B2 GiGPO | Estadística de estados repetidos | **Ninguno** | La PI aporta información que la estadística entre rollouts no puede ver |

Ganarle solo a B1 dejaría abierta la crítica "la estadística pura ya lo lograba"; ganarle solo a B2 dejaría abierta "un juez entrenado lo haría mejor". Los dos juntos cierran la pinza.

### 10.3 Donde PIAR predice separarse de GiGPO

La limitación de §4 es la predicción direccional: la señal de GiGPO **escala con la tasa de revisita de estados entre rollouts**. WebShop/ALFWorld (observaciones discretas, espacios chicos, N=8 desde el mismo seed) son su mejor escenario — y aun ahí <35%–65% de cobertura según la fase del training. La señal de PIAR es por-acción incondicional: no necesita que otro rollout haya pasado por el mismo estado. Si en algún brazo/environment con menos revisitas (instrucciones más diversas, horizontes más largos, estados más ricos) PIAR mantiene la ventaja y GiGPO converge a GRPO, eso es directamente parte del mapa "dónde/por qué" (arista 3 de LA PREGUNTA nueva). Vale loggear la distribución de $|G^S|$ (el código vendoreado ya trae `summarize_group_size`) como variable explicativa.

### 10.4 Ventaja operativa

B2 cuesta casi cero implementación: trainer, scripts y core del algoritmo ya están en `code/` (§8), sobre el mismo veRL fork donde corre iStar y donde se va a implementar PIAR. Mismo modelo base, mismos environments, mismo hook de advantage → comparación limpia por construcción. Lo único a decidir al correrlo: reconciliar las diferencias script-vs-paper (KL, max_steps de WebShop, invalid action penalty) y pre-registrar esa config.

### 10.5 Préstamos técnicos sueltos

- $\gamma = 0.95$ con returns descontados como anti-loitering implícito: penaliza trayectorias largas que llegan al mismo outcome. Relevante para el riesgo #2 del pivot (hacking del backward por loitering) — un $\gamma < 1$ en la parte outcome es un contrapeso barato adicional.
- La discusión $F_{\text{norm}}$ std vs 1 (task-dependent) alimenta directamente N.7 (normalización separada vs conjunta de los dos scores de PIAR): probar ambas, no asumir.
- $\omega$ robusto en [0.4, 1.2] pero colapsando afuera es un dato para el grid de λ (N.5): grids chicos centrados en 1 con un punto bajo y uno alto, no log-scale agresivo.

## 11. Lo más importante para retener

1. **GiGPO = GRPO + advantage step-level gratis**, agrupando acciones tomadas desde el mismo estado entre los N rollouts del grupo (anchor-state grouping, hashmap, <0.002% overhead). $A = A^E + \omega A^S$, $\omega \approx 1$.
2. **Es el SOTA critic-free en nuestros dos benchmarks target con nuestro modelo base exacto**: 90.2% ALFWorld y 75.2% success WebShop con Qwen2.5-7B-Instruct (+12.6 / +9.1 sobre GRPO). Esa es la vara de B2, no el piso A0.
3. **Su talón de Aquiles es estructural y medible**: sin estados repetidos entre rollouts, $A^S = 0$ y degrada a GRPO (cita textual en §4). La señal de PIAR no depende de revisitas — es la dimensión donde los métodos se separan y parte del mapa dónde/por qué.
4. **El brazo B2 viene casi gratis**: `code/examples/gigpo_trainer/` + `code/gigpo/core_gigpo.py` ya están vendoreados en el fork de iStar, con config consistente con el paper (salvo KL apagado, max_steps=10 en WebShop e invalid-action penalty — documentar al replicar).
5. **B1 (iStar) + B2 (GiGPO) forman la pinza del diseño experimental**: privilegio-en-pesos vs cero-privilegio. PIAR tiene que ganarle a los dos para que el claim "la PI en el canal del reward aporta" quede en pie.
