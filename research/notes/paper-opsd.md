# OPSD — On-Policy Self-Distillation

> **Issue:** [#5](https://github.com/lucaspecina/piar-rl/issues/5) · **arxiv:** [2601.18734](https://arxiv.org/abs/2601.18734) · **Código:** [`siyan-zhao/OPSD`](https://github.com/siyan-zhao/OPSD) · **Autores:** Zhao, Xie, Liu, Huang, Pang, Chen, Grover (UCLA / Meta). Submitted Jan 2026, v3 Mar 2026.
> **Rol en PIAR:** **el truco central de PIAR (privileged-context teacher) viene de acá.** Resuelve la sub-decisión frozen vs co-evolución del invariante 4: OPSD usa **frozen al checkpoint inicial** y lo justifica como regularización implícita.

## 1. Setup y notación

Dos políticas instanciadas del **mismo modelo** $p_\theta$ con contextos distintos:
- **Student:** $p_s(\cdot \mid x)$ — condicionada solo en el problema $x$.
- **Teacher:** $p_t(\cdot \mid x, y^*)$ — condicionada en problema + solución ground-truth $y^*$.

Misma arquitectura y mismo checkpoint inicial. **En step 0 son exactamente los mismos pesos**; durante el training el teacher se queda frozen (ver §3) mientras el student avanza, así que en cualquier $t > 0$ son `teacher θ_0` vs `student θ_t`. La diferencia entre ambos sigue viviendo en el contexto + el delta de training, pero el invariante de "no hay teacher con arquitectura distinta o entrenamiento separado" se mantiene. Esa es la primitiva que PIAR hereda.

## 2. Formato del prompt del teacher

OPSD da el template concreto (Figura 2):

```
Student prompt:
  Problem: [x]
  Answer:

Teacher prompt:
  Problem: [x]
  Here is a reference solution:
  [y*]   ← chain-of-thought completo
  After understanding the reference solution,
  please try to solve this problem using your own approach below:
  Answer:
```

Detalles clave:
- $y^*$ es el "reference solution" del dataset (CoT con pasos intermedios). El paper no detalla truncado más allá de los max length de generación de cada modo.
- El teacher **no genera tokens** — solo se usa como prefilling para obtener logits sobre los tokens del student.
- **Sin ofuscación, sin noising, sin re-paraphrasing.** El paper no discute leakage.

## 3. Régimen del teacher — frozen al checkpoint inicial

> "We fix the teacher policy to be the initial policy, rather than the currently updating learning policy, as we find this helps stabilize training and implicitly acts as regularization to prevent excessive deviation from the initial policy."

Es decir:
- Al inicio del training, snapshot del modelo → ese es el teacher para todo el run.
- El student avanza con gradientes; el teacher se queda fijo.
- Dos snapshots en GPU (teacher frozen + student que se mueve).

Esto **resuelve la sub-decisión abierta del invariante 4 de PIAR**: OPSD, que es la línea de la que sale la idea de privileged-context, usa **frozen** y lo justifica empíricamente como regularización. Es un dato fuerte para que PIAR adopte frozen como default.

Caveat para PIAR: OPSD es single-turn math reasoning con KL token-level. PIAR es multi-turn agentic con log-ratio span-level. La justificación "estabiliza training" probablemente transfiere, pero merece verificación empírica.

## 4. Loss — forward KL token-level

**Loss principal (full-vocabulary logit distillation):**

$$\mathcal{L}_{\text{OPSD}}(\theta) = \mathbb{E}_{(x, y^*) \sim S} \left[ \mathbb{E}_{\hat{y} \sim p_s(\cdot \mid x)} \left[ \frac{1}{|\hat{y}|} \sum_{n=1}^{|\hat{y}|} D\left( p_t(\cdot \mid x, y^*, \hat{y}_{<n}) \,\|\, p_s(\cdot \mid x, \hat{y}_{<n}) \right) \right] \right]$$

con $D = \text{forward KL}$ por defecto:

$$\text{KL}(p_t \,\|\, p_s) = \sum_v p_t(v) \log \frac{p_t(v)}{p_s(v)}$$

**Es token-level, full-vocabulary, on-policy** ($\hat{y}$ se samplea del student).

**Ablation forward KL vs reverse KL vs JSD (Tabla 3):**
| Divergencia | Acc step 100 |
|---|---|
| Forward KL | **41.1** |
| Reverse KL | 35.0 |
| JSD (β=0.5) | 39.0 |

Forward KL gana claramente. El issue body de #5 decía "reverse KL token-level" pero en realidad **el setup final del paper usa forward KL** — corrección importante.

**No comparan contra log-ratio directo (estilo Yuan).** Es un gap de la literatura: OPSD podría haber probado el log-ratio como Yuan lo formula, pero eligieron KL.

## 5. Alternativa: sampled-token policy gradient (Eq. 9)

Forma equivalente a un advantage por token:

$$\mathcal{L}(\theta) = -\mathbb{E}\left[\frac{1}{|\hat{y}|} \sum_n a_n(x, \hat{y}) \log p_s(\hat{y}_n \mid x, \hat{y}_{<n})\right], \quad a_n(x, \hat{y}) = \log p_t(\hat{y}_n \mid x, y^*, \hat{y}_{<n}) - \log p_s(\hat{y}_n \mid x, \hat{y}_{<n})$$

**El $a_n$ es la misma cantidad local que la primitiva de PIAR**: log-ratio token-wise en el token sampleado por el student. Pero el uso es distinto — en OPSD entra como advantage de policy gradient (con $\log p_t$ detached, sin gradiente al teacher), mientras PIAR lo va a sumar como step reward sobre el span de la acción ReAct y meter en advantage de GRPO. Misma cantidad local, distinto estimador y distinto uso.

OPSD reporta que full-vocab logit (forward KL) supera sampled-token PG por ~2% (Tabla 4). **Para PIAR**: el sampled-token PG de OPSD es la formulación más cercana a lo que PIAR quiere hacer; vale leer su implementación.

## 6. Trick crítico: pointwise KL clipping

Sin clipping el loss colapsa porque los tokens stylistic ("hmm", "maybe", "wait") dominan el gradiente — tienen alto KL pero baja relevancia para razonamiento. Con clipping pointwise sobre $\ell_{n,v}$ (cap por entry de vocab y posición), el training se estabiliza.

Esto es **muy relevante para PIAR**: en el span de una acción ReAct, los tokens stylistic (header del action ReAct, tokens delimitadores) pueden dominar el log-ratio. Si pasamos a action-level summing del log-ratio, la suma probablemente atenúa el problema, pero conviene loggear contribuciones por token al menos en debug.

## 7. Setup experimental

- **Datasets:** OpenThoughts (math, 30K problemas con CoT). Eval: AIME 2024, AIME 2025, HMMT 2025.
- **Modelos base:** Qwen3-1.7B / 4B / 8B (instruct), LoRA rank 64.
- **Baselines:** SFT, GRPO (8 rollouts × 16k tokens), off-policy distillation (mencionado, no implementado).
- **Hyperparams:**
  - LR: $5 \times 10^{-6}$ (igual GRPO).
  - Batch efectivo: 32.
  - OPSD: max 1024 tokens, 1 generación / prompt, 100 steps.
  - GRPO: max 16k tokens, 8 generaciones / prompt, 500 steps.
  - Temperature: OPSD 1.1, GRPO 1.2.

## 8. Resultados — números concretos

**Tabla 2, Qwen3-1.7B (la más dramática):**

| Método | AIME24 | AIME25 | HMMT25 | Avg |
|---|---|---|---|---|
| Base | 51.5 | 36.7 | 23.1 | 37.1 |
| SFT | 48.4 | 36.3 | 22.7 | 35.8 |
| GRPO | 51.1 | 38.3 | 23.7 | 37.7 |
| **OPSD** | **57.2** | **43.9** | **29.2** | **43.4** |

**Δ vs GRPO:** +5.7% en promedio. **SFT pierde contra base** — interesante, sugiere que distillation off-policy de y* no es suficiente.

**Qwen3-8B:** OPSD 64.8% vs GRPO 64.0% (+0.8%, mucho menor). El método pierde fuelle a más escala.

## 9. Eficiencia de tokens — corregir la cifra del issue

El issue body de #5 menciona "4-8x más token-efficient que GRPO". Los números del paper sugieren un ratio mucho mayor — Figura 3 + Tabla 6 implican (no como claim textual, sino como ratio computable):

- **OPSD por update:** 1 generación × 1024 tokens = ~1K tokens / problema.
- **GRPO por update:** 8 generaciones × 16k tokens = ~128K tokens / problema.
- **Ratio de presupuesto de generación: ~128×.**

Multiplicado por batch 32 ambos tienen la misma escala lineal; el ratio no cambia.

La razón física es importante:
> "More than half of [GRPO's] batches have zero reward standard deviation within 100 steps, yielding no gradient signal."

Cuando todas las trayectorias de un batch tienen el mismo outcome reward (binario), GRPO tiene gradiente cero. OPSD con KL token-level tiene gradiente denso siempre. **Ese es el argumento más fuerte de OPSD para PIAR**: con señal densa por step, no necesitás batches grandes para tener varianza no-trivial.

**Caveat:** la comparación es en su setup específico de single-turn math; en multi-turn la outcome reward es típicamente menos binaria (parcial credit), así que GRPO no se queda tan seca. La ventaja de PIAR sobre GRPO en multi-turn será probablemente menor que 128×, pero el argumento direccional vale.

## 10. Ablations relevantes para PIAR

1. **Forward KL > reverse KL > JSD** (Tabla 3). Esto es para distillation; para PIAR seguimos con log-ratio (heredado de Yuan/iStar) — pero la elección no está sin trade-off.

2. **Pointwise clipping crítico** (Figura 4). Sin él, loss colapsa.

3. **Generation length: 1024 vs 4096 no mejora** (Figura 5). "Early tokens más críticos." Para PIAR multi-turn esto se traduce a "tokens iniciales de la acción ReAct importan más que la cola del CoT". Vale revisar si en iStar el span de acción es típicamente largo o corto.

4. **Full-vocab logit ~2% mejor que sampled-token** (Tabla 4). Costo de memoria mayor.

5. **NO hay análisis de leakage.** Es el gap más grande del paper. Para PIAR esto es **oportunidad y obligación**: es un gap real en la literatura. Si PIAR no demuestra que su reward correlaciona con progreso causal y no con leakage textual, hereda la misma debilidad.

## 11. Multi-turn: no exploran

OPSD es 100% single-turn math reasoning. **No mencionan multi-turn ni agentic**. La extensión a multi-turn es genuinamente terreno de PIAR.

## 12. Limitaciones declaradas

1. Scale: solo hasta 8B. No verifican si escala a modelos más grandes.
2. Sin verificación explícita del output: no integran outcome reward signal.
3. Problem difficulty: si el problema supera la capacidad del modelo, el teacher no puede ayudar.
4. Curriculum: sugerido como future work.

## 13. Lo que esto significa PARA PIAR

### 13.1 Decisiones que OPSD fija o sugiere

1. **Teacher frozen al checkpoint inicial: default informado para PIAR.** OPSD usa frozen y lo justifica empíricamente (estabiliza training + regulariza implícitamente). Eso inclina la sub-decisión del invariante 4 hacia frozen como primer default — pero **no la cierra teóricamente**. OPSD valida frozen en single-turn distillation con KL forward; PIAR opera en multi-turn con log-ratio span-level. La justificación probablemente transfiere, pero la decisión definitiva merece una validación empírica en el setup de PIAR (puede ser tan simple como un ablation frozen-vs-coevol en la fase de experimentos).

2. **Formato del prompt del teacher reusable casi tal cual.** El template "Here is a reference solution: [y*]. After understanding the reference solution, please try to solve this problem using your own approach below: Answer:" se puede adaptar al setup multi-turn agentic de PIAR. La forma del prompt importa y OPSD ya iteró.

3. **El log-ratio token-wise (su sampled-token PG, Eq. 9) es lo más cercano a la primitiva de PIAR.** Su implementación es referencia útil cuando arranque el código.

4. **Teacher no genera, solo prefilling.** Eso es eficiente: un forward pass del teacher sobre los tokens del student. Mismo costo que iStar para computar logprobs sobre la trayectoria.

### 13.2 Lo que NO hereda PIAR

1. **OPSD usa forward KL, no log-ratio.** PIAR sigue con log-ratio (estilo Yuan/iStar). El argumento es que log-ratio hereda la igualdad soft-Q de Yuan y la integración con GRPO de iStar; KL no.

2. **OPSD es token-level, PIAR es action-level.** iStar mostró que action-level supera token-level en multi-turn. Mantener action-level para PIAR.

3. **OPSD ignora leakage.** PIAR no puede ignorarlo: el primer ablation crítico debe medir leakage vs progreso causal (ya identificado en notas de Yuan).

### 13.3 Riesgo identificado: stylistic tokens dominan

El trick de pointwise KL clipping de OPSD señala un riesgo real: en distribuciones de logprobs token-wise, ciertos tokens "stylistic" pueden dominar el log-ratio sin reflejar progreso útil. En PIAR multi-turn esto puede manifestarse como:

- Headers de la acción ReAct ("Thought:", "Action:") con log-ratio alto trivial.
- Tokens de boilerplate que el teacher predice mejor solo porque "vio el formato esperado en la golden".

**Mitigación a considerar para PIAR:** loggear distribución de contribuciones al log-ratio por tipo de token (header vs contenido vs delimitadores). Si headers/delimitadores dominan, aplicar clipping pointwise estilo OPSD o filtrado de tokens por rol.

### 13.4 Diseño del privileged context — variantes a explorar en PIAR

OPSD pasa la solución completa con CoT. Para PIAR, qué pasa al "teacher" puede variar y cada opción tiene distinto perfil de leakage:

- **Answer literal** (la respuesta final, sin reasoning): máximo leakage, mínima información de progreso.
- **CoT completo estilo OPSD:** balance OPSD adoptó. Funciona para math pero no probado en agentic.
- **SCM o esquema de razonamiento:** el modelo gana información estructural sin texto literal del answer. Menor leakage trivial.
- **Trace exitoso de un agente entrenado** (otra línea, más cara).

**Acción para PIAR:** la elección del privileged context es un hyperparameter de diseño que merece su propia ablation. No asumir que "lo de OPSD" es óptimo para multi-turn.

## 14. Conexiones con otros papers

- **Yuan 2024 (#3):** OPSD usa forward KL; Yuan formaliza log-ratio. Son hermanos: ambos derivan de KL-regularized RL, pero OPSD se queda en el lado distillation y Yuan se queda en el lado RL. PIAR está más en el lado Yuan.
- **iStar (#4):** ambos usan "two policies" pero iStar entrena el segundo (PRM via DPO trayectorial), OPSD usa el mismo modelo con contexto distinto. PIAR combina las dos primitivas: same-model + log-ratio.
- **Skill-SD / Co-evolución (mencionado en PROJECT.md):** la otra opción frozen-vs-coevol que OPSD descarta (con justificación empírica). Si experimentamos con co-evolución más adelante, esto es contexto.

## 15. Decisiones / preguntas que esto dispara

- [x] **Sub-decisión invariante 4: teacher frozen al checkpoint inicial como primer default.** (Inclinada tras leer OPSD; cierre definitivo merece validación empírica en setup PIAR.)
- [ ] **Diseño del privileged context para PIAR (multi-turn agentic):** ¿pasamos answer + trace exitoso? ¿solo SCM? ¿solo respuesta? Ablation de PIAR a hacer en fase 5.
- [ ] **¿Aplicar clipping pointwise estilo OPSD?** Sospecho que sí; loggear primero distribución de contribuciones por token.
- [ ] **¿Token-level o action-level?** Action-level por iStar; reconfirmado.

## 16. Lo más importante para retener

1. **Frozen al checkpoint inicial es el primer default informado para el teacher de PIAR.** Una sub-decisión inclinada (no cerrada): falta validar empíricamente en el régimen multi-turn log-ratio.
2. **Template del prompt del teacher** ya iterado por OPSD; reusar.
3. **Log-ratio token-wise (Eq. 9) es la primitiva común de PIAR.** Implementación de OPSD vale como referencia.
4. **OPSD no analiza leakage.** PIAR debe; es gap real de la literatura, no oversight de PIAR.
5. **Eficiencia 128× vs GRPO en su setup específico.** No transfiere literal a multi-turn pero el argumento direccional (señal densa por step → menos colapso de varianza) es la motivación más fuerte para PIAR.
