# Yuan et al. 2024 — Free Process Rewards without Process Labels

> **Issue:** [#3](https://github.com/lucaspecina/piar-rl/issues/3) · **arxiv:** [2412.01981](https://arxiv.org/abs/2412.01981) · **Código:** [lifan-yuan/ImplicitPRM](https://github.com/lifan-yuan/ImplicitPRM)
> **Rol en PIAR:** base teórica del log-ratio como step reward sin step labels. Provee el lenguaje matemático y los supuestos. **No** prueba lo que PIAR necesita — cambiamos el régimen del teorema (ver §6).

## 1. Problema y motivación

PRMs (process reward models) dan reward denso por step pero requieren step labels manuales o vía MCTS — caros y ruidosos. ORMs (outcome reward models) entrenan con etiqueta solo a nivel respuesta completa, baratos pero la señal es esparsa. Yuan demuestra que se puede tener lo mejor de ambos: entrenar con outcome labels pero recuperar reward por step "gratis", parametrizando el reward de cierta forma.

## 2. Resultado central — Proposition 3.1

Definir el outcome reward como log-ratio escalado:

$$r_\theta(\mathbf{y}) := \beta \log \frac{\pi_\theta(\mathbf{y})}{\pi_{\text{ref}}(\mathbf{y})}$$

Acumular hasta el step $t$:

$$q_\theta^t(\mathbf{y}_{<t}, y_t) := \sum_{i=1}^{t} \beta \log \frac{\pi_\theta(y_i \mid \mathbf{y}_{<i})}{\pi_{\text{ref}}(y_i \mid \mathbf{y}_{<i})}$$

Yuan demuestra:

$$q_\theta^t = \beta \log \mathbb{E}_{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{y}_{\leq t})} e^{\frac{1}{\beta} r_\theta(\mathbf{y})}$$

Es decir, $q_\theta^t$ es la **soft Q-value** del proceso bajo el ORM en KL-regularized RL — sin haber entrenado un PRM ni usado step labels.

**Step reward implícito:**

$$r_\theta^t := q_\theta^t - q_\theta^{t-1} = \beta \log \frac{\pi_\theta(y_t \mid \mathbf{y}_{<t})}{\pi_{\text{ref}}(y_t \mid \mathbf{y}_{<t})}$$

Calculable con dos forward passes (π_θ y π_ref). Sin entrenar nada nuevo.

## 3. Derivación — qué hace funcionar el teorema

La descomposición aditiva del log-ratio sobre tokens es álgebra elemental:

$$\log \frac{\pi_\theta(\mathbf{y})}{\pi_{\text{ref}}(\mathbf{y})} = \sum_{i=1}^{n} \log \frac{\pi_\theta(y_i \mid \mathbf{y}_{<i})}{\pi_{\text{ref}}(y_i \mid \mathbf{y}_{<i})}$$

Pero la **interpretación como Q-function válida del PRM** se apoya en KL-regularized RL (Rafailov et al. 2024 — DPO):

> "DPO training enables the model to learn the Q function implicitly, but our insights subsume their conclusion since this property is not limited to the DPO algorithm" (p. 6)

Yuan generaliza el insight de DPO: cualquier loss que parametrice $r_\theta = \beta \log \pi_\theta / \pi_{\text{ref}}$ y entrene con outcome labels da automáticamente un PRM válido. Las dos piezas que necesita la prueba:

1. **Parametrización**: el reward está expresado como log-ratio entre dos LMs.
2. **Régimen de entrenamiento**: π_θ se obtiene optimizando un objetivo con outcome labels que respeta la estructura KL-regularized (DPO, KTO, NCA, o CE sobre el reward escalar).

## 4. Supuestos — distinguir dos niveles

Hay que separar dos claims distintos en el paper, que en una primera lectura se confunden:

**(a) Identidad algebraica (siempre vale).** Proposition 3.1 — la igualdad
$$\sum_{i=1}^{t} \beta \log \frac{\pi_\theta(y_i \mid \mathbf{y}_{<i})}{\pi_{\text{ref}}(y_i \mid \mathbf{y}_{<i})} = \beta \log \mathbb{E}_{\pi_{\text{ref}}(\mathbf{y} \mid \mathbf{y}_{\leq t})} e^{\frac{1}{\beta} r_\theta(\mathbf{y})}$$
es una identidad de soft Q-value bajo KL-regularized RL para **cualquier** par de distribuciones autoregresivas normalizadas $(\pi_\theta, \pi_{\text{ref}})$ con un reward $r_\theta := \beta \log \pi_\theta / \pi_{\text{ref}}$. **No requiere que π_θ haya sido entrenado de ninguna forma específica.** Es álgebra + propiedades del soft-max.

**(b) Semántica de PRM (requiere training).** Que el step reward $r_\theta^t$ se interprete como **progreso hacia la respuesta correcta** (un PRM útil) sí requiere que el outcome reward $r_\theta$ tenga semántica de "correctness". Eso se logra entrenando π_θ con outcome labels usando un loss compatible (DPO, KTO, NCA, CE). El entrenamiento es lo que **alinea el reward parametrizado con la noción de respuesta correcta**.

Lo que el teorema (a) **no** requiere:
- Loss específica.
- Que π_θ haya sido entrenado en absoluto.

Lo que la **utilidad como PRM** (b) sí requiere:
- π_θ entrenado con outcome labels en la parametrización del log-ratio.
- π_ref como referencia frozen para que la KL constraint tenga significado.

## 5. Setup experimental y resultados

**Datos:** UltraInteract (33K instrucciones de math) × 8 rollouts = 264K respuestas. Outcome labels = correctness vs ground truth.

**Modelo base PRM:** Llama-3.1-8B-Instruct.

**Benchmark:** MATH-500 con generadores Mistral-7B / Llama-3.1-8B / Llama-3.1-70B.

**Headline (Tabla 1, p. 15):**

| Método | Avg Acc | Overhead vs Math-Shepherd |
|---|---|---|
| Implicit PRM (DPO) | 50.4% | 1× |
| Implicit PRM (CE) | 48.4% | 1× |
| Math-Shepherd | 47.8% | 38.8× (1/38 menos data en implicit) |
| AutoPSV | 45.7% | ~20× |

**Selection en best-of-N (inferencia):** $\text{score}(\mathbf{y}) = \min_t r_\theta^t$ (mínimo step reward por respuesta — heurística de robustez).

**Hyperparámetro clave:** $\beta = 0.05$.

## 6. Ablations relevantes para PIAR

1. **Step labels de Math-Shepherd añadidos al training NO ayudan** (+0.0% a −0.1%, Tabla 2 p. 21). Interpretación: el implicit PRM ya capta la estructura por step; añadir labels ruidosos no aporta y empeora.

2. **Loss functions** — todas las parametrizaciones razonables del log-ratio funcionan: DPO 50.4 / NCA 49.4 / CE 48.4 / KTO 45.7. **CE funciona con una sola respuesta por instrucción** (caso extremo unpaired) — democratiza muchísimo la recolección de datos.

3. **Reference model en inferencia puede omitirse** si el implicit PRM viene de un modelo fuerte que ya pasó por preference learning (Llama-3.1-Instruct). Implica overhead reducible en producción.

4. **Soft vs hard Q-bound (Proposition 3.2, p. 8):**
   $$q_{\theta_s}^t \leq q_\theta^t \leq q_{\theta_h}^t$$
   El implicit PRM cae entre el MCTS soft (subestima) y hard (sobreestima); es un punto intermedio robusto, lo cual la lógica empírica que vimos en (1) refuerza.

5. **PRM ≠ buen policy.** Un implicit PRM bueno como evaluador no es necesariamente un buen policy: DPO/NCA/CE empeoran accuracy directo en MATH (25–36% vs baseline 45%); solo KTO mejora marginal (+1.4%, Tabla 3 p. 22). Para PIAR esto es importante: no confundir "el teacher es un buen evaluador" con "el teacher es un buen actor".

## 7. Limitaciones declaradas

- **Overhead del ref model en inferencia:** 22% (Llama-70B) a 200% (Mistral-7B) — depende del tamaño relativo PRM/generador.
- **Domain mismatch:** instrucciones off-task (coding, general QA) bajan accuracy. El implicit PRM es de dominio.
- **Noise en step labels MCTS** (resultado empírico, no de prueba): MCTS introduce ruido que el implicit PRM evita por construcción.

## 8. Lo que esto significa PARA PIAR — el delta crítico

Con la distinción (a) algebraica vs (b) semántica de §4, podemos ser precisos sobre qué hereda PIAR.

### Lo que sí se conserva: la igualdad soft-Q (nivel a)

Definamos:
$$r_{\text{PIAR}}(\mathbf{y}) := \beta \log \frac{\pi_{\text{teacher}}(\mathbf{y} \mid x, \text{golden})}{\pi_{\text{student}}(\mathbf{y} \mid x)}$$

con teacher y student el mismo modelo evaluado en contextos distintos, y β escalar (mismo rol que en Yuan). Bajo condiciones razonables (mismo tokenizer, mismo prefijo generado, soporte compartido), la identidad de Proposition 3.1 vale: la suma token-wise dentro del span de una acción ReAct es exactamente la soft Q-value de **un** reward — concretamente, el reward "cuánto racionaliza el contexto golden la trayectoria $\mathbf{y}$".

Es decir: $r_{\text{PIAR}}^t$ **es** una soft-Q exacta. Eso no es un agujero teórico.

### Lo que no se conserva: la semántica del reward (nivel b)

El agujero está en (b). El reward implícito de Yuan tiene semántica de "correctness" porque π_θ fue entrenado con outcome labels. El reward de PIAR tiene semántica de "leakage de contexto privilegiado":

> "cuánta probabilidad extra le asigna el modelo a esta trayectoria cuando le metés la golden answer en el prompt, vs cuando no"

Esa cantidad **no es necesariamente** una medida de progreso causal hacia la respuesta correcta. Puede serlo en gran parte de los casos (si la golden induce al modelo a internamente reconstruir el camino correcto y eso se refleja en mayor probabilidad token a token de las acciones que efectivamente avanzan), pero también puede ser:

- **Compatibilidad textual con la golden** (el modelo le sube probabilidad a tokens que coinciden con strings de la golden, sin que eso refleje progreso útil).
- **Leakage trivial** (si la golden contiene la respuesta directa, cualquier acción que la repita gana score sin haber "razonado").

### La pregunta abierta de PIAR — reformulada

> El reward $r_{\text{PIAR}}$ es una soft-Q válida en el sentido algebraico. La pregunta es **semántica**: ¿correlaciona con progreso causal hacia la respuesta correcta, o solo con compatibilidad textual / leakage del contexto privilegiado? ¿Cuándo sí y cuándo no? ¿Qué construcciones del golden minimizan el leakage trivial y preservan la señal de progreso?

Esta es la pregunta empíricamente verificable, y es la que define el éxito o fracaso experimental de PIAR.

### Lo que Yuan sí da, listo para usar

- **Vocabulario y formulación:** PIAR puede expresarse en el mismo lenguaje (log-ratio, β, suma sobre span).
- **β = 0.05** como punto de partida razonable. Mantener el escalar β explícito en la formulación de PIAR — sin él, la escala experimental de Yuan no se importa.
- **Heurística de selección $\min_t r_\theta^t$** para best-of-N.
- **Advertencia (ablation 5):** un buen reward no implica buen policy. Aplicable a PIAR si se intenta usar el teacher como actor.
- **Evidencia empírica (ablation 1) de que step labels de MCTS no ayudan al implicit PRM en el setup de Yuan.** Esto **no** se generaliza directamente a PIAR — es evidencia sugestiva, no predictiva. El paper mismo advierte no extender la conclusión.

### Decisiones que dispara para PIAR

1. **El primer ablation crítico de PIAR debe medir leakage vs progreso causal.** Comparar $r_{\text{PIAR}}^t$ con un PRM entrenado o con anotaciones humanas en una muestra. Si correlacionan en trayectorias correctas pero también suben en trayectorias que repiten texto del golden sin razonar, hay leakage y la métrica falla en su rol semántico.
2. **Diseño del golden en función del leakage.** Si pasamos el answer literal hay máximo leakage; si pasamos un SCM o un esquema de razonamiento, el modelo todavía gana información pero el reward se acerca a "progreso" en lugar de "compatibilidad textual".
3. **Mantener β = 0.05 explícito** como hiperparámetro inicial. Sin β explícito el reward se desnormaliza y los rangos experimentales de Yuan no son referencia.
4. **No reentrenar el teacher.** Romper invariante 4 nos lleva a SWEET-RL / distillation guiada — otra línea distinta.

### Notación: token vs step

Yuan define $r_\theta^t = q_\theta^t - q_\theta^{t-1} = \beta \log \pi_\theta(y_t \mid \mathbf{y}_{<t}) / \pi_{\text{ref}}(y_t \mid \mathbf{y}_{<t})$ a nivel **token**. Para PIAR la unidad natural es el **span de una acción ReAct** $\mathbf{y}_{[a]} = (y_{i}, \dots, y_{j})$. El step reward de PIAR para esa acción es la suma token-wise dentro del span:

$$r_{\text{PIAR}}^{[a]} := \sum_{i \in \text{span}(a)} \beta \log \frac{\pi_{\text{teacher}}(y_i \mid \mathbf{y}_{<i}, x, \text{golden})}{\pi_{\text{student}}(y_i \mid \mathbf{y}_{<i}, x)}$$

Mantener clara la distinción token-index vs step-index para evitar ambigüedad cuando se compare con Yuan o con iStar.

## 9. Citas y conexiones

- **Rafailov et al. 2024** (DPO + Q-function implícita): el caso particular del que Yuan generaliza.
- **Math-Shepherd** (Wang et al. 2024): baseline más fuerte que Yuan supera. Próximo paper a leer (issue #8).
- **MCTS-based PRMs** (varios): la línea con step labels que Yuan compara.
- **PRIME framework** (issue #6): aplica implicit PRM al loop completo de RL.

## 10. Decisiones que faltan resolver

- [ ] ¿En PIAR usamos **el mismo β = 0.05** o lo dejamos como hyperparámetro a tunear?
- [ ] ¿Aplicamos $\min_t$ de Yuan para inferencia, o $\sum_t$ por advantage en GRPO?
- [ ] ¿Hace falta una condición de KL constraint análoga en PIAR para que el teacher no se aleje del student de manera incoherente?
