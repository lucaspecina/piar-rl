# TAMTRL — Teacher-Aligned Reward Reshaping for Multi-Turn RL in Long-Context Compression

> **Issue:** epic Pivot 2026-06 (ver Project v2) · **arxiv:** [2603.21663](https://arxiv.org/abs/2603.21663) (v1, 23 Mar 2026) · **Código:** ⚠️ no verificado — la página de arXiv y el HTML no linkean repo · **Autores:** Li Wang, Yandong Wang, Xin Yu, Kui Zhang, Tianhao Peng, Wenjun Wu (afiliaciones no listadas en la página de abs).
> **Rol en PIAR:** **el vecino que mata al forward standalone** (pivot 2026-06, §2-§3). Publica el mecanismo "mismo modelo, dos contextos, probabilidad del scorer privilegiado como step reward multi-turn" — la mitad pragmática del PIAR original, en variante. Pasa de amenaza a related work que **valida el componente forward** y obliga a las 4 diferenciaciones de pivot §10.6. NO toca la descomposición dual, ni la PI residual, ni agentes de decisión.

## 1. Setup: memory agent multi-turn para long-context

La tarea es **compresión/lectura long-context por chunks**, no agentes de decisión:

- Documento largo $D$ que excede la context window se parte en $n$ chunks de longitud predefinida.
- En cada turno $t$ el modelo recibe $[q, D_t, M_t]$ (query, chunk crudo, memoria) y genera la memoria actualizada $M_{t+1}$ (longitud máxima fija).
- Tras procesar todos los chunks, la respuesta final se genera desde $[q, M_n]$.
- Supervisión natural = solo outcome final (Exact Match) → temporal credit assignment sobre los updates de memoria. Ese es el problema que TAMTRL ataca.

Lo formalizan como POMDP y lo enmarcan en **CTDE** (centralized training, decentralized execution):

> "Inspired by the centralized training with decentralized execution (CTDE) paradigm in RL, TAMTRL employs a teacher model with a more global perspective to provide centralized supervision during training."

En ejecución el student opera solo con observaciones locales (chunk crudo); el privilegio existe únicamente en training. Esto es estructuralmente el invariante 3 de PIAR con otro nombre.

## 2. El mecanismo same-model dos-contextos

**Teacher = exactamente el mismo modelo $\pi_\theta$ que el student, con otro input.** No hay segundo modelo, ni pesos extra, ni juez entrenado:

- **Student**: ve $[q, D_t, M_t]$ — chunk crudo, sin filtrar.
- **Teacher**: ve $[q, C_t, M_t]$ — donde $C_t$ es el **chunk filtrado**: "we first remove irrelevant content based on the ground-truth document annotations, yielding a filtered chunk $C_t$ that contains only relevant information" (§4.3).

La información privilegiada es **la anotación de qué pasaje del documento es relevante** (proveniente de HotpotQA, con síntesis estilo RULER/MemAgent), **no la golden answer**. El teacher no sabe la respuesta; sabe dónde mirar. El paper no detalla algorítmicamente el filtrado más allá de "remove irrelevant content based on the ground-truth document annotations".

Detalle operativo importante: el teacher se computa con la **policy actual en entrenamiento** $\pi_\theta$ — no hay snapshot congelado ni $\pi_{old}$ separada (a diferencia de PIAR, que usa $\pi_{old}$ como scorer):

> "we feed the student model with $[q, D_t, M_t]$ to obtain the updated memory $M_{t+1}$. The teacher-aligned reward for this memory update is computed as the average token-wise probability assigned by the teacher model $\pi_\theta$"

## 3. El reward: probabilidad del teacher A SECAS — verificado, NO es ratio

Confirmado contra el texto: **TAMTRL usa la probabilidad del teacher directamente, no un log-ratio teacher/student**. Y va más lejos: ni siquiera usa log-probs — promedia **probabilidades crudas** token a token (Eq. 4):

$$p_t = \frac{1}{|M_{t+1}|} \sum_i \pi_\theta\!\left(m_{t+1}^{(i)} \,\middle|\, [q, C_t, M_t]\right)$$

Es decir: "¿qué probabilidad le asigna el modelo-con-contexto-filtrado a la memoria que el student escribió viendo el chunk crudo?". El contexto del student ($[q, D_t, M_t]$) **no aparece en ninguna parte del reward** — no hay denominador, no hay resta de baselines de logprob del student. La fórmula PIAR-forward $\log \pi(a|h,g) - \log \pi(a|h)$ no tiene contraparte acá.

El $\frac{1}{|M_{t+1}|}$ es length-norm explícita (ablation: sacarla cuesta −1.72 puntos en 0.6B — "previene preferencia por outputs más largos").

## 4. Min-max normalization + gating multiplicativo por outcome

### 4.1 Min-max (Eq. 5)

Las probabilidades crudas son inconsistentes en magnitud entre prompts/turnos, así que normalizan min-max **globalmente sobre el batch de entrenamiento** — todas las queries, todos los turnos, todos los rollouts del grupo:

$$\hat{p}_{t,ji} = \frac{p_{t,ji} - \min\{p^1_{11}, \ldots, p^s_{nG}\}}{\max\{p^1_{11}, \ldots, p^s_{nG}\} - \min\{p^1_{11}, \ldots, p^s_{nG}\}}$$

con $s$ = queries del batch, $n$ = turnos, $G$ = rollouts por query (G=8). ⚠️ Matiz de notación no 100% verificado: la extracción del HTML sugiere que el min/max corre sobre el conjunto completo del batch (no por grupo de rollouts ni por turno aislado); una fuente del fetch lo describió como "across the dataset divided by turn". A revisar contra el PDF si se cita en el paper.

**La ablation más violenta del paper**: sin min-max el training **colapsa a 0%** ("Falla completa"). La normalización no es un detalle — es condición de existencia del método. Razón plausible: probabilidades crudas promediadas viven en rangos minúsculos y dispares; sin reescalar, el advantage queda dominado por artefactos de escala.

### 4.2 Gating multiplicativo por outcome (Eq. 6)

El reward por turno NO es la probabilidad sola — se **multiplica por el outcome reward binario** $r_{ji} \in \{0,1\}$ (Exact Match de la respuesta final):

$$R_{t,ji} = \hat{p}_{t,ji} \cdot r_{ji}$$

> "Turn-level rewards are assigned by modulating the normalized teacher probability score with the outcome reward […] This design ensures that the correctness of the final answer governs the overall supervision signal, thereby avoiding conflicts between optimization objectives."

Consecuencia estructural: **las trayectorias fallidas reciben reward 0 en todos los turnos**. El shaping de TAMTRL solo redistribuye crédito *dentro* de trayectorias exitosas; no da señal de "ibas bien hasta acá" en trayectorias que fallan. La ablation `plus-reward` (sumar $\hat{p} + r$ en vez de multiplicar) destruye el método: **−20.12 puntos** — el gating multiplicativo es tan crítico como el min-max.

### 4.3 Integración RL (Eq. 7)

Algoritmo base: **DAPO** (clipping desacoplado $\varepsilon_{low}/\varepsilon_{high}$), no PPO ni GRPO vanilla. El advantage por turno se normaliza dentro de la query:

$$A_{t,ji} = \frac{R_{t,ji} - \text{mean}(\{R^i_{11}, \ldots, R^i_{nG}\})}{\text{std}(\{R^i_{11}, \ldots, R^i_{nG}\})}$$

Los $R_{t,ji}$ **reemplazan** al outcome reward único de DAPO (no se suman al advantage de outcome como en iStar/PRIME — el outcome ya entra vía el gating multiplicativo).

Hiperparams: G=8 rollouts/query, LR $10^{-6}$, 800 steps, rollout batch 32, **KL coef $10^{-3}$** (nota: a diferencia de la línea Yuan/iStar/PRIME/OPSD que pone KL=0, TAMTRL sí usa KL penalty), contexto de training 8K (1024 query + 5000 chunk + 1024 memoria).

## 5. Teoría: Teorema 1 (descomposición information-theoretic del objetivo)

El objetivo de TAMTRL se descompone exactamente en tres términos:

$$J(\theta) = P(r_i{=}1|S_t)\,\mathcal{L}_{succ}(\theta) + P(r_i{=}0|S_t)\,\mathcal{L}_{fail}(\theta) + \beta\, I_{\pi_\theta}(M_{t+1}; r_i \mid S_t)$$

Lectura: (a) en trayectorias exitosas, alinearse con el teacher bajo restricción KL; (b) en fallidas, mantener KL sin castigo; (c) un término de **información mutua entre la memoria y el outcome** — maximizar cuánto predice la memoria el éxito/fallo.

Nuance relevante para PIAR: el término de MI tiene sabor epistémico ("la memoria debe informar sobre el resultado"), pero es un **artefacto del análisis del objetivo**, no un componente de reward separado y medible por turno. TAMTRL implementa **un solo término de reward** ($\hat{p} \cdot r$); no hay descomposición pragmático/epistémico operativa, ni belief tracking, ni nada parecido al backward de PIAR.

## 6. Setup experimental y resultados

**Modelos base:** Qwen3-0.6B y Qwen3-1.7B (thinking mode off). Modelos **chicos** — un orden de magnitud abajo del Qwen2.5-7B de iStar/PIAR.

**Benchmarks (7, todos QA long-context):** HotpotQA (in-domain, de ahí salen las anotaciones), RULER-QA, Multi-keys NIAH, 2WikiMultihopQA, MuSiQue, NarrativeQA, Qasper.

**Resultados principales (Tabla 1, promedios):**

| Método | 0.6B | 1.7B |
|---|---|---|
| Base | 1.94 | 16.85 |
| SFT | 35.29 | 39.86 |
| MemAgent (DAPO outcome-only) | 37.26 | 40.23 |
| LLM-judge | 36.72 | 39.46 |
| **TAMTRL** | **39.29** | **43.21** |

Baselines completos: SFT, STaR, Vanilla-KD, MemAgent+DAPO (outcome-only), LLM-as-judge, PRM. Mejora sobre el mejor baseline: **~+2 puntos** (modesta pero consistente en 7 benchmarks y 2 escalas).

**Costo:** "only a single forward pass" extra. Tabla A2: TAMTRL 33.79h vs LLM-judge 48.41h vs PRM 81.77h (y hasta menos que MemAgent 37.75h, por respuestas más cortas). Valida el argumento de eficiencia que PIAR comparte: el scoring same-model es ~gratis frente a jueces externos.

**Ablations adicionales:** `global-reward` (alinear contra el contexto completo en vez del chunk filtrado del turno): −1.25 — alinear contra todo mete ruido; el privilegio *localizado por turno* importa. Chunk size óptimo ~5000 tokens (Fig. 6). Robustez moderada a densidad de información (Fig. 5).

## 7. Related work del paper (lo que NO cita)

Verificado: TAMTRL **no cita** la línea de privileged information, ni information gain (IGPO), ni on-policy distillation con contexto privilegiado (OPSD/π-Distill/OPCD), ni iStar/implicit PRM. Sus vecinos declarados: knowledge distillation (Vanilla-KD), process supervision (LLM-as-judge, PRMs), temporal credit assignment (Monte Carlo, critic-based). Llegaron al mecanismo desde MARL/CTDE y memory agents, no desde la línea implicit-PRM/privileged-teacher. Dos comunidades convergiendo al mismo truco independientemente — evidencia adicional de que la ventana del forward standalone estaba cerrada.

Sin sección explícita de limitaciones.

## 8. Las 4 diferencias vs PIAR (mandato pivot §10.6, con evidencia textual)

| # | Dimensión | TAMTRL | PIAR (post-pivot) | Evidencia |
|---|---|---|---|---|
| (i) | **Forma del score** | Probabilidad del teacher a secas — probs crudas promediadas por token, sin log, sin denominador: $p_t = \frac{1}{\|M_{t+1}\|}\sum_i \pi_\theta(m^{(i)}_{t+1}\|[q,C_t,M_t])$. El contexto del student no entra al reward | **Log-ratio teacher/student**: $\log\pi_{old}(tok\|h,R(t)) - \log\pi_{old}(tok\|h)$ — mide el *delta inducido por la PI*, controla por lo que el student ya hace probable | Eq. 4 + confirmación textual: "Turn-level rewards are assigned by modulating the normalized teacher probability score with the outcome reward". No hay ratio ni diferencia teacher−student en el paper |
| (ii) | **Qué es la PI** | Documento filtrado: anotaciones ground-truth de relevancia de pasajes (HotpotQA) — "remove irrelevant content based on the ground-truth document annotations". El teacher sabe *dónde mirar*, no la respuesta | Golden **estructurada en componentes fácticos** $g=(g_1,\ldots,g_K)$ (producto/atributos/precio; goal + estado oculto) y **residual**: el scorer ve solo los $g_i$ que el student aún no sabe (gating por belief $b_i(t{-}1)<\tau$) | §4.3 del paper vs pivot §4.1/4.3. TAMTRL no descompone la PI ni la dosifica — el filtrado es fijo por turno, dado por el dataset |
| (iii) | **Dominio** | Compresión/memoria long-context: QA sobre documentos por chunks, acción = escribir memoria. Título mismo: "…in Long-Context Compression". 7 benchmarks, todos QA | **Agentes de decisión** multi-turn (WebShop, ALFWorld): acciones que cambian el estado de un environment, trade-off explorar/ejecutar | Lista de benchmarks §6; el paper no menciona WebShop/ALFWorld/tool-use/search agents. Crucial: en su dominio cada turno "informa" por construcción (leer chunk → memoria), así que el hindsight bias contra la exploración casi no existe — en agentes de decisión sí, y es lo que motiva el backward |
| (iv) | **Descomposición dual** | Un solo término de reward, multiplicativo: $R = \hat{p} \cdot r$, cero en trayectorias fallidas. El término de MI del Teorema 1 es análisis, no componente operativo | $r_t = \hat{z}(r_{fwd}) + \lambda\,\hat{z}(r_{bwd})$: pragmático + epistémico como componentes medibles y ablacionables por separado, con el mapa de cuándo domina cada uno | §5: "un único reward multiplicativo (p̂·r), sin componentes pragmático/epistémico separadas". Además TAMTRL no da señal alguna en episodios fallidos; el dual de PIAR da señal densa independiente del outcome |

Diferencias adicionales menores (útiles para review pero no centrales): (v) scorer = policy actual $\pi_\theta$ en entrenamiento vs $\pi_{old}$ snapshot en PIAR; (vi) reemplaza el outcome reward vs se suma al advantage de outcome (estilo iStar); (vii) KL coef $10^{-3}$ vs la línea KL=0; (viii) modelos 0.6B/1.7B vs 7B.

## 9. Lo que esto significa PARA PIAR (post-pivot)

### 9.1 Por qué mata al forward standalone

El claim original de PIAR — "señal densa por turno desde un teacher same-model con contexto privilegiado, sin reward model ni step labels, en RL multi-turn" — está **publicado en variante** desde marzo 2026. TAMTRL tiene cada ingrediente del esqueleto: mismo modelo dos contextos, privilegio derivado de ground truth, score por turno, single forward pass, dentro de RL multi-turn, con la misma narrativa anti-PRM/anti-judge ("substantial computational overhead… estimation noise") y con teoría. Un paper PIAR-viejo no podría diferenciarse más que en detalles de forma (ratio vs prob, golden vs documento) — insuficiente como contribución standalone, exactamente como concluye pivot §3. Que además TAMTRL llegue desde CTDE sin conocer la línea OPSD/iStar confirma que la idea estaba "in the air".

### 9.2 Por qué valida el mecanismo como componente

- **El canal funciona**: probabilidades del same-model con contexto privilegiado llevan señal de crédito por turno suficiente para ganarle a outcome-only, LLM-judge y PRM en 7 benchmarks y 2 escalas, a una fracción del costo (33.8h vs 81.8h del PRM). Es evidencia empírica independiente a favor del forward de PIAR.
- **CTDE = invariante 3 con otro vocabulario**: privilegio solo en training, ejecución a ciegas. La estructura que PIAR asume queda validada y con frame teórico citable.
- **Lecciones de ingeniería directas**: (a) la normalización del score es existencial (sin min-max → 0%); refuerza que la normalización por grupo $\hat{z}$ del reward combinado de PIAR (pivot §4.4, N.7) no es opcional; (b) length-norm importa (ablation D.5 de PIAR ya la tenía como candidata); (c) alinear privilegio *localizado* por turno > privilegio global (`global-reward` −1.25) — rima con el residual: menos PI mejor dirigida > toda la PI siempre.
- **Lo que TAMTRL NO puede hacer y PIAR sí**: cero señal en trayectorias fallidas (gating multiplicativo) — en environments difíciles donde casi todo rollout temprano falla, TAMTRL degenera a outcome-only; el dual de PIAR da señal siempre. Y en agentes de decisión su score puro-forward heredaría el hindsight bias contra la exploración (pivot §4.3) que su dominio le ocultó: escribir memoria siempre "parece" ejecutar; buscar en WebShop no.

### 9.3 Posicionamiento en related work (obligatorio, pivot §10.6)

TAMTRL entra al paper de PIAR como el vecino forward más cercano, con las 4 diferencias de §8 explícitas. Frase candidata: "TAMTRL valida que las probabilidades de un teacher same-model con contexto privilegiado dan crédito por turno barato y útil — en compresión long-context, con la PI como documento filtrado y score absoluto gateado por outcome. PIAR generaliza el mecanismo a agentes de decisión (log-ratio que aísla el delta inducido por la PI, PI fáctica estructurada y dosificada por belief) y lo complementa con la dirección epistémica que el dominio de TAMTRL no necesita pero los agentes sí."

## 10. Lo más importante para retener

1. **Verificado el claim central del pivot doc**: TAMTRL usa la **probabilidad del teacher a secas** (probs crudas promediadas por token, Eq. 4) — NO un log-ratio teacher/student. El contexto del student no aparece en el reward. La diferencia (i) de §10.6 es sólida y citable.
2. **El reward es $\hat{p} \cdot r$ — gateado multiplicativamente por el outcome binario** (detalle que el pivot doc no menciona): trayectorias fallidas reciben 0 en todos los turnos, y la versión aditiva pierde 20 puntos. TAMTRL solo redistribuye crédito dentro de éxitos; PIAR da señal densa independiente del outcome. Es una quinta diferencia explotable en related work.
3. **Min-max sobre el batch es existencial** (sin ella: colapso a 0%). Confirma que la normalización del score privilegiado no es un hiperparámetro menor — presupuesta la decisión N.7 de PIAR.
4. **Dominio y escala acotados**: compresión long-context QA (HotpotQA-family), Qwen3-0.6B/1.7B, +~2 puntos sobre el mejor baseline. Nada de agentes de decisión, nada de exploración, nada de descomposición dual ni dosificación. El espacio (a)+(b)+(c) del pivot sigue libre.
5. **No cita a OPSD/iStar/IGPO/privileged-info** — llegó desde CTDE/memory agents. Dos líneas independientes convergieron al same-model-dos-contextos en ~12 meses: la velocidad del nicho que pivot §10.5 advierte es real, y la diferenciación de PIAR tiene que vivir en la descomposición y la dosificación, no en el mecanismo base.
