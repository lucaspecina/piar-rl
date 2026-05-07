# Math-Shepherd — predecesor histórico explícito

> **Issue:** [#8](https://github.com/lucaspecina/piar-rl/issues/8) · **arxiv:** [2312.08935](https://arxiv.org/abs/2312.08935) · **Venue:** [ACL 2024 Long](https://aclanthology.org/2024.acl-long.510) · **Autores:** Wang, Li, Shao, Xu, Dai, Y. Li, Chen, Y. Wu, Sui (DeepSeek + Peking). Submitted Dec 2023, v3 Feb 2024.
> **Rol en PIAR:** **contexto histórico, lectura liviana.** Es la línea "explicit and expensive" que Yuan/iStar/PRIME mejoran con implicit PRM. Saber qué resuelve Math-Shepherd y por qué pierde frente a implicit PRM aclara qué problema están atacando los papers que sí impactan a PIAR.
>
> **Caveat de fuente:** PDF de arxiv y ACL no parsearon a texto en esta sesión; este resumen viene del abstract verificado, los números del ACL, y la descripción del método citada en otros papers que ya leímos (Yuan §1, §5; PRIME §1; SWEET-RL related work). Las fórmulas exactas merecen verificación con el PDF si en algún momento se necesita más profundidad.

## 1. Problema que resuelve

PRMs anteriores requerían **step labels manuales** (estilo "Let's Verify Step-by-Step" de OpenAI 2023) — costoso, no escala. Math-Shepherd propone **construir step labels automáticamente** vía Monte Carlo rollouts, eliminando la dependencia humana.

## 2. Método central — Monte Carlo step labeling

**Idea.** Para cada prefijo parcial $\mathbf{y}_{\leq t}$ de una solución, samplear $N$ continuaciones del mismo policy model y ver cuántas llegan a la respuesta correcta.

**Dos variantes de label:**

- **HE (Hard Estimation):**
$$y^{HE}_t = \mathbb{1}\left[\exists i \in \{1, \dots, N\}: \text{answer}(\text{rollout}_i) = y^*\right]$$
  ("alguno llegó a la respuesta correcta")

- **SE (Soft Estimation):**
$$y^{SE}_t = \frac{|\{i: \text{answer}(\text{rollout}_i) = y^*\}|}{N}$$
  (proporción de rollouts correctos)

**Hyperparams del paper:** $N = 8$ rollouts por step. El completer (modelo que genera los rollouts) es **LLemma-7B**; los modelos usados como PRM, policy, etc., son distintos según el experimento (no necesariamente el mismo que el completer). Trabajos posteriores exploran rangos similares de $N$.

## 3. Loss y entrenamiento del PRM

**Loss** (cross-entropy sobre los step labels):
$$\mathcal{L}_{\text{PRM}} = -\mathbb{E}_t \left[ y_t \log p_\phi(y_t \mid \mathbf{y}_{\leq t}) + (1 - y_t) \log(1 - p_\phi(y_t \mid \mathbf{y}_{\leq t})) \right]$$

**PRM model:** suele ser el mismo backbone que el policy (Mistral-7B, DeepSeekMath), con head de clasificación (sigmoid sobre el step). Fine-tune supervisado.

**Datasets:** GSM8K (~7K problems), MATH (~12K problems). Multiplicado por number of steps × N rollouts → varios cientos de miles de generaciones para construir el dataset de step labels.

## 4. Dos modos de uso

**(a) Verification (best-of-N):** dado $N$ candidatos, computar step scores con el PRM y agregarlos por trayectoria con **minimum score across all steps** como método principal reportado en el paper. Combinable con self-consistency por grupos de respuestas. (Otras agregaciones — mean, product — son posibles pero no son lo principal.)

**(b) RL (PPO reinforcement):** el PRM emite un step reward que entra al advantage de PPO. Step rewards densos vs outcome-only.

## 5. Resultados (Mistral-7B base)

Reportados en el abstract / ACL:

| Configuración | GSM8K | MATH |
|---|---|---|
| SFT base | 77.9 | 28.6 |
| RL with Math-Shepherd | 84.1 | 33.0 |
| RL + Math-Shepherd verification | 89.1 | 43.5 |

**Δ vs SFT:** +6.2 GSM8K (RL only), +4.4 MATH (RL only). Stack completo (RL + verification): +11.2 / +14.9.

## 6. Costo computacional — el punto débil

El cuello de botella es el labeling MC: $N$ rollouts × $T$ steps × $|D|$ problems. Para $N = 8$, $T \approx 5$ steps por solución, $|D| \approx 10K$ problemas → **~400K generaciones** mínimas para construir el step-labeled dataset (más para $N$ mayor o $T$ más largo).

**Comparación con implicit PRM (Yuan 2024):**
- Yuan iguala o supera Math-Shepherd usando **~1/38 del compute** — la cifra exacta del paper de Yuan (abstract / Figure 2 sobre overhead) es que CE implicit PRM es **38.6× a 38.8× más eficiente** que Math-Shepherd en FLOPs/overhead.
- La fuente del speedup: implicit PRM no necesita generar rollouts MC desde cada step intermedio. Solo necesita outcome labels (1 forward pass por response completa) y la parametrización log-ratio se encarga de extraer step rewards "gratis" via factorización autoregresiva.

**Por qué implicit PRM gana conceptualmente, no solo en compute** (insight de Yuan):
- **HE sobreestima Q**: bastando que UN rollout llegue a correcto, el step se etiqueta como "valor 1", aunque el camino sea de baja probabilidad. Es un estimador del max sobre rollouts, no del valor esperado.
- **SE subestima en problemas difíciles**: si el problema es difícil, la mayoría de los $N$ rollouts fallan, dando label cercano a 0 incluso cuando el step es razonable; muchos falsos negativos.
- **Implicit PRM** estima Q vía log-ratio sin rollouts MC. La Proposition 3.2 de Yuan (que ya consolidamos en `paper-yuan-implicit-prm.md` §6.4) muestra que el implicit Q queda entre los bounds soft/hard MCTS — un estimador intermedio robusto que evita ambos sesgos.

## 7. Limitaciones declaradas

- **Ruido en HE/SE:** los labels MC son aproximaciones a "value true" del state, dependientes de cuántos rollouts se hicieron. Con $N$ chico → labels ruidosos.
- **Modelo policy debe ser razonable** para que los rollouts logren llegar a respuestas correctas con frecuencia. Si el problema es muy difícil → la mayoría de rollouts fallan → SE colapsa cerca de 0 o varianza alta.
- **Costo compute** es el más declarado.

## 8. Código

**Sin training code oficial abierto** (los autores indican que el código interno de step-wise PPO no se libera). Pero **sí hay artefactos oficiales** en HuggingFace bajo `peiyi9979/Math-Shepherd`: dataset de step labels + checkpoints (PRM, SFT, RL). Suficiente para usar el PRM como baseline off-the-shelf en evaluation, no suficiente para reproducir el training pipeline completo. Yuan y PRIME re-implementan el método en sus propios repos.

## 9. Por qué Yuan lo elige como baseline

Yuan (#3) compara contra Math-Shepherd porque:
1. Es el baseline más fuerte de "automated step labels en math reasoning" al momento de su publicación.
2. Comparar implicit PRM (sin step labels) vs Math-Shepherd (con step labels MC) aísla la contribución de la formulación implícita: si la implícita iguala con menos compute, demuestra que los step labels MC eran innecesarios.
3. **Math-Shepherd es uno de varios baselines** que Yuan usa (también AutoPSV y varios ORMs/PRMs abiertos), no el único. Pero **es el baseline headline** porque hace el punto del overhead más concreto: Yuan iguala/supera con ~1/38 de compute.

## 10. Lo que esto significa PARA PIAR

### 10.1 No genera decisiones nuevas para PIAR
Math-Shepherd no aporta primitivas nuevas — ni log-ratio (no usa), ni privileged context (no usa), ni multi-turn (single-turn math only). Es referencia de "lo que NO hacemos" en PIAR.

### 10.2 Refuerza la motivación general de implicit PRM
PIAR hereda la idea de Yuan de que **step labels explícitos son innecesarios** si la parametrización es la correcta (log-ratio). Math-Shepherd cuantifica el ahorro: implícito = ~1/38 del costo de explícito MC. Si en algún momento alguien sugiere "hagamos step labels MC además del log-ratio context-induced", la evidencia de Yuan / iStar (Tabla 3 ablation: GT process labels NO ayudan) y de Math-Shepherd vs implicit PRM dice que es deuda muerta.

### 10.3 Tabla mental "Math-Shepherd → Yuan → PIAR"

| | Math-Shepherd | Yuan (Implicit PRM) | PIAR |
|---|---|---|---|
| Cómo se obtiene la señal por step | $N$ rollouts MC desde cada prefijo + voting | Factorización autoregresiva del log-ratio entre policy y ref | Factorización del log-ratio entre teacher (con golden) y student (sin) |
| Step labels manuales? | No (auto via MC) | No | No |
| Step labels via MC? | **Sí** | No | No |
| Modelo entrenado? | PRM separado, CE sobre step labels | Policy entrenado con outcome labels, parametrización log-ratio | **Nada se entrena adicionalmente** — teacher es el modelo inicial con golden context |
| Compute relativo | 1× (línea base) | ~1/38× | similar a Yuan, sin el costo del PRM training (un par de forward passes adicionales para obtener logits del teacher) |
| Multi-turn agentic? | No | No | **Sí** |
| Privileged context? | No | No | **Sí (golden answer en prompt del teacher)** |

PIAR es la tercera columna: hereda "no step labels" de los dos anteriores, hereda "no PRM training" del segundo, agrega "multi-turn" y "privileged context" como contribuciones propias.

## 11. Lo más importante para retener

1. **Math-Shepherd es el ancestro a vencer** que Yuan derrota con implicit PRM. No tiene primitivas reusables para PIAR.
2. **El ratio "1/38× compute" de Yuan vs Math-Shepherd es la motivación operativa más concreta** del approach implicit. Si un reviewer pregunta "¿por qué no usar Math-Shepherd directamente?" la respuesta es: 38× más caro y no le gana al implicit en accuracy (Yuan Tabla 1).
3. **Step labels MC añadidos no ayudan a un implicit PRM** (Yuan Tabla 2 ablation, ya consolidado). Math-Shepherd refuerza el contexto de esa conclusión.
4. **Sin código oficial canónico** — el método se reimplementa en Yuan / PRIME para tener un baseline reproducible.
