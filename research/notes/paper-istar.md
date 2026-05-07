# iStar — Agentic RL with Implicit Step Rewards

> **Issue:** [#4](https://github.com/lucaspecina/piar-rl/issues/4) · **arxiv:** [2509.19199](https://arxiv.org/abs/2509.19199) · **Código:** [`Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar) · **Autores:** Liu, Wang, Wu, Huang, Li, Zhang, Jiao (Alibaba). Submitted Sep 2025, v3 Sep 28. **Aceptado en ICLR 2026.**
> **Rol en PIAR:** **vecino matemático directo.** Es la formulación más cercana en la literatura a lo que PIAR propone, y la diferencia con PIAR es exactamente la tesis no-negociable (invariante 4): iStar entrena un PRM separado vía DPO trayectorial; PIAR usa el mismo modelo con golden en contexto.

## 1. Contexto y motivación

iStar ataca el credit assignment en agentes RL multi-turn. Critica trabajos previos que intentan usar process supervision pero sufren de:
- Sesgo en step labels.
- Reward hacking.
- Varianza alta cuando los rewards son demasiado granulares.
- Fallos cuando hay poco overlap entre estados.

La propuesta: process rewards **implícitos** que se integran a algoritmos RL estándar **sin rollouts adicionales ni step labels explícitos**.

## 2. Step reward implícito — la fórmula

$$r_\phi(o_{1:t}, a_t) = \beta \log \frac{\pi_\phi(a_t \mid o_{1:t}, x)}{\pi_{\theta_{\text{old}}}(a_t \mid o_{1:t}, x)}$$

Donde:
- $\pi_\phi$: PRM aprendido (modelo separado, distinto del policy actual).
- $\pi_{\theta_{\text{old}}}$: snapshot del policy de la iteración anterior — **no** una referencia frozen, sino una referencia móvil que se actualiza después de cada update (co-evolución).
- $\beta = 0.05$ (mismo valor que Yuan).
- $a_t$: acción ReAct completa (CoT + acción ejecutable). **Action-level**, NO token-level.

Es decir: $\pi_\phi$ y $\pi_{\theta_{\text{old}}}$ son **dos redes con pesos distintos**. El log-ratio mide cuánto el PRM aprendido difiere del policy actual sobre cada acción.

## 3. Cómo se entrena el PRM

**Objetivo DPO trayectorial:**

$$\mathcal{J}_{\text{PRM}}(\phi) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\phi(\tau^+ \mid x)}{\pi_{\theta_{\text{old}}}(\tau^+ \mid x)} - \beta \log \frac{\pi_\phi(\tau^- \mid x)}{\pi_{\theta_{\text{old}}}(\tau^- \mid x)}\right)\right]$$

donde $\pi_\phi(\tau \mid x) = \prod_t \pi_\phi(a_t \mid o_{1:t}, x)$ es **producto de probabilidades de acciones**, no la likelihood completa de la trayectoria. Tokens generados por el ambiente (observaciones) no entran al producto — iStar es explícito sobre esto.

**Construcción de pares $\tau^+, \tau^-$:**
- Rankear los rollouts de la iteración con un outcome verifier.
- Threshold: success rate > 0 (WebShop / VisualSokoban) o goal score > 6 (SOTOPIA).
- $\tau^+$ supera el threshold; $\tau^-$ no.

**Por qué decompone en step-rewards:** la factorización autoregresiva del likelihood de la trayectoria descompone la preferencia trayectorial en una suma de preferencias por acción → de ahí sale la equivalencia con un PRM step-wise (extiende un teorema de Zhong et al. 2025).

**Frecuencia:** PRM se actualiza una vez por iteración, alternado con el policy.

## 4. Combinación step + trajectory advantage

**Episode advantage** (outcome reward, normalizado intra-batch):
$$A^E(\tau_i) = \frac{r_o(\tau_i) - \text{mean}(R_o)}{\text{std}(R_o)}$$

**Step advantage** (implicit PRM, normalizado intra-batch):
$$A^S(a_t^i) = \frac{r_\phi(a_t^i) - \text{mean}(R_s)}{\text{std}(R_s)}$$

**Combinado:**
$$A(a_t^i) = A^E(\tau_i) + \alpha A^S(a_t^i), \quad \alpha = 1$$

Sin discounting, sin pesos extra. La normalización intra-batch es lo que pone ambos en escalas comparables.

**Policy loss (GRPO):**
$$\mathcal{J}_{\text{policy}} = \mathbb{E}\left[\frac{1}{NT}\sum_i \sum_t \min\left(\rho_\theta(a_t^i) A(a_t^i),\, \text{clip}(\rho_\theta(a_t^i), 1 \pm \epsilon) A(a_t^i)\right)\right]$$

donde $\rho_\theta = \pi_\theta / \pi_{\theta_{\text{old}}}$ es el ratio de importancia **a nivel acción**, NO token. **Sin KL penalty.**

## 5. Algoritmo (resumen del loop)

Por iteración:
1. Samplear N trayectorias (8 por prompt) con el policy actual.
2. Rankearlas con el outcome verifier; formar pares preferenciales.
3. Update del PRM con DPO trayectorial.
4. Computar $A^E$ y $A^S$ con el PRM ya updateado.
5. Update del policy con surrogate GRPO usando el advantage combinado.
6. Snapshot $\theta_{\text{old}} \leftarrow \theta$.

## 6. Setup experimental

**Benchmarks y modelos base:**
- **WebShop** — Qwen2.5-7B-Instruct.
- **VisualSokoban** — Qwen2.5-VL-7B-Instruct (policy) + Qwen2.5-7B (PRM).
- **SOTOPIA** — Qwen2.5-7B o Llama3.1-8B; GPT-4o juzga goal completion 0-10.

**Hyperparameters críticos:**
- $\beta = 0.05$, $\alpha = 1$.
- LR: $5 \times 10^{-7}$ (policy), $10^{-6}$ (PRM).
- Batch 64, micro-batch 8.
- Rollouts: 8 por prompt; WebShop 16 grupos (128 envs), Sokoban 32 grupos (256 envs).
- Pasos: 200 (WebShop/Sokoban), 800 (SOTOPIA).
- Penalty acciones inválidas: −0.1.
- Max length: 512 response, 4096 prompt (WebShop).
- **Sin KL penalty.**

**Hardware:** 8×A100. Match con lo que tendríamos en Azure ML Y-TEC.

**Framework:** **veRL** (Alibaba). NO prime-rl.

## 7. Resultados — números concretos

| Benchmark | Métrica | Baseline RLOO | iStar | Δ |
|---|---|---|---|---|
| WebShop | Success | 77.4 | **86.5** | +9.1 |
| WebShop | Score | 87.6 | **93.6** | +6.0 |
| WebShop (vs GiGPO) | Success | 84.1 (GiGPO) | **86.5** | +2.4 |
| VisualSokoban | Success | 86.3 | **91.7** | +5.4 |
| SOTOPIA Hard self-chat | Goal score (0-10) | 7.92 (GRPO) | **8.06** | +0.14 pts |
| SOTOPIA Hard vs GPT-4o | Goal score (0-10) | 6.68 (GRPO) | **7.16** | +0.48 pts |

## 8. Ablations clave (Tabla 3)

| Variante | WebShop Score | VisualSokoban Success |
|---|---|---|
| RLOO baseline | 84.2 | 85.9 |
| + ground-truth process rewards | — | 87.5 |
| + merged rewards (suma directa) | 90.7 | 88.3 |
| + token-level process rewards | 90.0 | 89.1 |
| **iStar (action-level + advantage-level)** | **94.7** | **93.0** |

**Tres insights enormes:**

1. **Action-level supera token-level** (94.7 > 90.0 en WebShop; 93.0 > 89.1 en Sokoban). Para PIAR esto pesa: el span natural es la acción ReAct, no el token. La fórmula de PIAR debería sumarse sobre el span de la acción, no usarse token-wise.

2. **Environmental/raw process rewards no ganan tanto** (+1.6% en Sokoban) — confirma el insight de Yuan: process labels (sea de MCTS o reward shaping del entorno) son ruidosos o mal calibrados, los implicit rewards captan estructura por step gratis y mejor.

3. **Advantage-level normalization gana sobre suma directa** (94.7 vs 90.7). Importante para PIAR: no sumar $r^{step}$ y $r^{trajectory}$ directo — normalizar primero a $A^S$ y $A^E$ con stats de batch, después sumar.

> "We should not only reward intermediate actions but also gate credit by final task success to prevent speculative reward exploitation."

Esto va al corazón del leakage problem que identificamos en Yuan: necesitás que la señal trayectorial gate la señal step para evitar que el modelo explote rewards intermedios espurios.

## 9. Lo que NO mencionan: privileged-context teachers

**No hay ninguna mención** de pasar la golden answer al PRM en contexto. Todo el credit assignment viene de learned preferences ranked por outcome verifier. Ni descartan ni evalúan la idea — simplemente no la consideran.

**Esto confirma el posicionamiento de PIAR:** la idea de "teacher = mismos pesos del student + golden en contexto" no está cubierta por iStar — ni la consideran ni la descartan. iStar es la formulación más cercana matemáticamente, pero el régimen es diferente: dos redes vs una red + dos contextos.

## 10. Lo que esto significa PARA PIAR

### 10.1 Decisiones que iStar fija o sugiere

1. **Action-level, no token-level.** Sumar el log-ratio de PIAR sobre el span de la acción ReAct completa (CoT + acción). El ablation 2 lo justifica empíricamente.

2. **Combinar $A^S + A^E$, no $r^{step} + r^{trajectory}$.** Normalizar ambos con stats de batch antes de combinar. $\alpha = 1$ como punto de partida.

3. **β = 0.05** consistente con Yuan. Mantener.

4. **Importance ratio agregado a nivel acción en GRPO.** Internamente se usan logprobs token-level (por la naturaleza autoregresiva del modelo), pero se agregan al nivel de la acción ReAct y se excluyen los tokens generados por el ambiente. El score de cada acción es la suma de logprobs token-level de los tokens producidos por el modelo dentro del span de esa acción. Cambia cómo se computa $\rho_\theta$ vs GRPO standard token-level.

5. **Sin KL penalty.** iStar lo omite y funciona; posible default razonable para PIAR.

6. **Hardware target: 8×A100** confirma que Azure ML Y-TEC con A100/H100 es adecuado para replicar.

### 10.2 La sub-decisión abierta del invariante 4 (PROJECT.md)

PROJECT.md deja abierta dentro del invariante 4: ¿teacher **frozen al checkpoint inicial del student** (estilo OPSD) o teacher **co-evolutivo con el student** (estilo Skill-SD)? Ambas respetan el invariante porque ambas usan los pesos del student (la diferencia es si esos pesos están congelados al snapshot inicial o si se mueven junto al student). La asimetría sigue viviendo solo en el contexto.

**Caveat sobre cómo iStar informa esta decisión.** $\pi_{\theta_{\text{old}}}$ en iStar **no es un teacher privilegiado** — es el snapshot móvil del policy, sin golden context. iStar no entrena un teacher con golden context bajo ningún esquema. Por lo tanto iStar **no es evidencia directa** sobre cuál de las dos opciones de PIAR funciona mejor; a lo sumo sugiere que esquemas con referencias móviles son entrenables sin colapso, pero no más que eso. La decisión real entre frozen-checkpoint vs co-evolución se va a decidir leyendo OPSD (#5), donde sí hay teacher con privileged context con esquema frozen, y eventualmente experimentalmente.

**Las dos opciones bajo invariante 4:**
- (a) **Co-evolución (estilo Skill-SD)**: teacher = student actual + golden en contexto. Un solo conjunto de pesos en GPU. Cuando el student updatea, el teacher se mueve.
- (b) **Frozen al checkpoint inicial (estilo OPSD)**: teacher = snapshot del student al t=0 + golden en contexto. Dos snapshots en GPU. Más estable; mismo set de pesos al inicio, pero teacher congelado mientras student avanza.

**Esta sub-decisión hay que resolverla tras leer OPSD (#5)** y posiblemente comparando ambas empíricamente.

### 10.3 Estimación líneas a modificar

iStar usa **veRL** (Alibaba), no prime-rl. **El código ya está liberado** en [`Tongyi-ConvAI/Qwen-Character/CharacterRL-iStar`](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar) (post-aceptación ICLR 2026). El repo contiene: `verl/` (fork del framework), `istar/` (lógica del método), `gigpo/` (baseline), `agent_system/`, `recipe/`, `scripts/`, etc. Esto da dos caminos reales:

- **Plan A** — implementar PIAR sobre **prime-rl**. **Líneas estimadas: 80–250.** Portar la lógica core de step-advantage de iStar (~30-50 líneas) + teacher-context-injection (golden en prompt sí/no por trayectoria, ~50-80 líneas) + advantage combination (~30-50 líneas). prime-rl ya soporta multi-turn nativo + verifiers como rubric.
- **Plan B** — implementar PIAR sobre **veRL fork de iStar** (ahora viable porque el código está). **Líneas estimadas: 30–100.** Aprovechás el codebase exacto del paper más cercano; la modificación es máximamente quirúrgica (básicamente cambiar el "teacher" de "PRM aprendido vía DPO trayectorial" a "mismo modelo + golden en contexto" — borrar más que agregar).

**Recomendación operativa para cuando arranque la fase de código:** mantener Plan A (prime-rl) por la decisión de issue #11, **pero leer el código de iStar como referencia** antes de empezar a portar. Hay que confirmar que las primitivas de iStar (action-level importance ratio, advantage-level combination, step-level GRPO) son razonables de reproducir en prime-rl. Si el porteo resulta caro, **reabrir Plan B** como alternativa concreta — ya no depende de espera, el código está.

Acción concreta cuando arranque fase de código: clonar [`CharacterRL-iStar`](https://github.com/Tongyi-ConvAI/Qwen-Character/tree/main/CharacterRL-iStar), correr setup, replicar baseline WebShop. Eso dispara también la fase 3 del roadmap (Replicación de baseline).

## 11. Limitaciones declaradas por iStar

1. PRM separado del policy en training — sugieren unificarlos para eficiencia de memoria (paradójicamente, esa es exactamente la dirección de PIAR).
2. SOTOPIA PRM solo entrenado en goal-completion preference — se podría hacer multi-objetivo.
3. (No declaradas pero notables): no evalúan robustez a verifier ruidoso, no hacen ablation sobre $\alpha$, no exploran $\beta$ alternativos.

## 12. Decisiones / preguntas que esto dispara

- [ ] **Confirmar action-level vs token-level para PIAR.** Decisión: action-level (ablation iStar lo justifica).
- [ ] **Confirmar advantage-level combination (no reward-level).** Decisión: advantage-level con $A^E + \alpha A^S$, $\alpha = 1$ inicial.
- [ ] **Frozen vs sincronizado** — pendiente OPSD (#5).
- [ ] **Plan A (prime-rl) vs Plan B (veRL)** — decidido Plan A en issue #11. Re-confirmar tras leer OPSD.
- [ ] **Replicar baseline RLOO de iStar en WebShop antes de tocar PIAR** — esto es la fase 3 del roadmap. Hyperparameters ya conocidos.

## 13. Conexiones con otros papers

- **Yuan 2024 (#3)**: iStar usa la misma fórmula log-ratio + β, pero con dos redes distintas y entrenamiento DPO trayectorial en vez de outcome cross-entropy. La descomposición autoregresiva → step rewards extiende Zhong et al. 2025 en lugar de seguir directamente a Yuan.
- **OPSD (#5)**: la otra opción del invariante 4 (frozen teacher con privileged context). Próximo paper a leer.
- **Math-Shepherd (#8)**: iStar lo critica explícitamente como ejemplo de PRM con step labels ruidosos. Conviene leerlo después para entender qué problema concreto está evitando iStar.
- **GiGPO**: baseline directo en WebShop (84.1% vs 86.5% iStar). Vale la pena entender qué hace si vamos a comparar con él.

## 14. Lo más importante para retener

1. **Action-level + advantage-level normalization es el setup ganador.** Heredar para PIAR.
2. **β = 0.05** consistente cross-paper.
3. **iStar es lo más cercano a PIAR pero no considera privileged context** → PIAR está en territorio que iStar no exploró.
4. **Co-evolución (teacher = student actual) es el default forzado en PIAR** por invariante 4, no una elección como en iStar.
5. **Replicar RLOO + iStar en WebShop antes de tocar PIAR** define el baseline reproducible (fase 3 del roadmap).
