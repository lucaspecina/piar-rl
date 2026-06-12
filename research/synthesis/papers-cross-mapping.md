# PIAR — Síntesis cruzada de papers vecinos

> **Cierre del epic de research** (#2). Consolida los 7 papers analizados en
> `research/notes/paper-*.md` y aísla **el delta de PIAR**: qué hereda de cada
> vecino, qué cambia, y dónde vive PIAR en el mapa de la literatura.
>
> Para decisiones de diseño específicas (β, α, snapshot del teacher, etc.) ir
> a [`design-decisions.md`](design-decisions.md). Este doc es el **story
> cross-paper** + el delta explícito, no el índice de decisiones.
>
> ⚠️ **Actualización 2026-06-12**: el delta original ("celda vacía"
> multi-turn × PI-en-prompt × frozen) **fue ocupado** por la literatura de
> feb–jun 2026 (TAMTRL en variante; IGPO tomó el espejo). El mapa de las
> secciones siguientes sigue siendo válido como historia de los 7 vecinos
> originales, pero el posicionamiento vigente de PIAR está en la sección
> **"Pivot 2026-06"** al final de este doc y en
> [`pivot-2026-06.md`](pivot-2026-06.md) (fuente de verdad del pivot).

## TL;DR — dónde vive PIAR en la literatura

PIAR ocupa una **celda vacía** en el cruce de tres líneas existentes:

| Eje | Lo que PIAR hereda | De quién |
|---|---|---|
| Mecanismo | log-ratio de logprobs como step reward (soft-Q implícito) | Yuan 2024 |
| Setting | multi-turn agentic RL con GRPO + action-level + advantage-level | iStar (ICLR 2026) |
| Teacher | mismos pesos del student + golden answer en contexto | OPSD (Zhao et al. 2026) |

Y agrega **una cosa nueva**: combinar los tres ejes en un solo método sin entrenar nada extra.

### El delta de PIAR en una frase

> *Reemplazar el PRM aprendido de iStar (información privilegiada codificada en
> pesos vía DPO sobre outcomes) por el mismo modelo con golden en el prompt
> (información privilegiada codificada en el contexto). Cero entrenamiento de
> un juez separado.*

## Mapa cruzado paper × PIAR

### Yuan 2024 — Implicit PRM
[notas](../notes/paper-yuan-implicit-prm.md) · [#3](https://github.com/lucaspecina/piar-rl/issues/3) (cerrado)

- **Hereda**: el teorema soft-Q (Prop 3.1) — cualquier log-ratio de dos
  distribuciones autoregresivas es interpretable como función-Q implícita de
  un reward dado. Esa es la base matemática que justifica que el reward de
  PIAR es algebraicamente válido como soft-Q sin entrenar nada.
- **Cambia**: Yuan asume que el PRM se entrena (CE o DPO sobre outcomes).
  PIAR no entrena: usa el log-ratio entre dos prompts del mismo modelo.
- **Delta**: PIAR es Yuan con la asimetría en el contexto en vez de en
  pesos entrenados.
- **Caveat**: Yuan demuestra la igualdad **algebraica**; el gap entre "valor
  Q de un reward" y "reward semánticamente útil" sigue siendo abierto y es
  el riesgo central de PIAR — leakage textual vs progreso causal (D.1, D.9).

### iStar — Agentic RL with Implicit Step Rewards (ICLR 2026)
[notas](../notes/paper-istar.md) · [#4](https://github.com/lucaspecina/piar-rl/issues/4) (cerrado)

**El vecino matemático directo.** La modificación de PIAR es quirúrgica sobre
este paper.

- **Hereda**:
  - Action-level + advantage-level normalization (94.7 vs 90.0 token-level en WebShop).
  - β = 0.05, α = 1, sin KL penalty.
  - GRPO con importance ratio agregado a nivel acción.
  - Setup completo de training: Qwen2.5-7B-Instruct + WebShop + 8 GPUs.
  - Pipeline reference de combinación A^E (outcome) + α·A^S (step).
- **Cambia**: iStar entrena un PRM separado (π_φ) vía DPO trayectorial sobre
  outcome rankings. PIAR no — usa el mismo modelo con golden en prompt como
  término "privilegiado" del log-ratio.
- **Delta**: **PIAR = iStar sin entrenar el juez.** La asimetría del
  log-ratio se mueve de los pesos del PRM al prompt del mismo modelo. iStar
  mismo declara en sus limitaciones que la unificación PRM+policy es una
  dirección abierta — PIAR la cierra desde una óptica más radical (no solo
  unificar pesos, sino reemplazar el aprendizaje por información en el
  contexto).
- **Implicación operacional**: forkear `CharacterRL-iStar` y cambiar
  **~150 líneas** (estimación inicial 30-100 era optimista, revisada
  2026-05-12 post-review Codex — ver [`piar-implementation-points.md`](piar-implementation-points.md) §6.4). Es Plan B (cerrado en A.2 de `design-decisions.md`).

### OPSD — Self-Distilled Reasoner (Zhao et al. 2026)
[notas](../notes/paper-opsd.md) · [#5](https://github.com/lucaspecina/piar-rl/issues/5) (cerrado)

**El origen conceptual del teacher con privileged context.**

- **Hereda**:
  - La idea central: mismos pesos del student + golden en contexto como
    "teacher". Lo que PIAR llama "invariante 4" sale de acá.
  - Teacher como prefilling (no genera tokens nuevos, solo logits sobre la
    trayectoria del student) — 1 forward pass extra.
  - Template del prompt del teacher: "Here is a reference solution: [y*].
    After understanding it, please try to solve this problem using your own
    approach below."
- **Cambia**: OPSD vive en **single-turn math** con **forward KL token-level**
  entre teacher y student. PIAR vive en **multi-turn agentic** con
  **log-ratio action-level** como step reward de GRPO.
- **Delta**: PIAR es OPSD trasladado al régimen agentic, con log-ratio en
  vez de KL como mecanismo de propagación de la señal.
- **Caveat resuelto 2026-05-11**: OPSD usa frozen θ₀. PIAR primario usa
  `π_old` (snapshot reciente igual al denominador del log-ratio en iStar),
  porque frozen θ₀ rompe "mismos pesos del student" después del primer
  update y mezcla efecto-contexto con weight-drift. Frozen θ₀ queda como
  ablation legítima de estabilidad.

### PRIME — RL with Implicit PRM (Cui et al. 2025)
[notas](../notes/paper-prime.md) · [#6](https://github.com/lucaspecina/piar-rl/issues/6) (cerrado)

- **Hereda**: el patrón de combinar step rewards implícitos con outcome
  rewards vía LOO baseline (PRIME lo aplica a math; PIAR adopta el patrón
  para agentic).
- **Cambia**: PRIME entrena el PRM online (CE sobre outcomes). PIAR mantiene
  frozen el "teacher" (sin entrenamiento alguno del término privilegiado).
- **Delta**: PIAR es PRIME sin el online update del PRM.
- **Caveat útil**: PRIME ablation muestra que el accuracy filtering en
  rollout selection mejora ~35% la estabilidad cuando todos los rollouts del
  batch comparten outcome. PIAR debe verificar si necesita filtering análogo
  cuando la varianza del log-ratio es baja (D.6 candidata).

### SWEET-RL — Asymmetric Critic for Agentic RL (Zhou et al. 2025)
[notas](../notes/paper-sweet-rl.md) · [#7](https://github.com/lucaspecina/piar-rl/issues/7) (cerrado)

**La línea de la que PIAR explícitamente se diferencia.**

- **Hereda**: evidencia empírica directa de que "privileged info at training
  time mejora credit assignment" (+9.2 puntos atribuibles a PI en su
  ablation on/off, 40.4 → 31.2). Validación direccional clave para PIAR.
- **Cambia**: SWEET-RL pone la PI en un crítico LLM separado entrenado con
  Bradley-Terry. PIAR la pone en el prompt del mismo modelo. Sin critic
  entrenado, sin pipeline two-stage offline.
- **Delta**: **PIAR es SWEET-RL minimalista** — misma intuición (PI at
  training time), implementación radicalmente más simple (prompt vs critic
  separado).
- **Test científico vs SWEET-RL**: si log-ratio context-induced captura
  suficiente señal, PIAR gana en simplicidad sin perder en credit
  assignment. Si pierde, hay que pivot.
- **Caveat operacional**: SWEET-RL ablation muestra que sin $1/L$
  normalization su método colapsa de 40.4% → 3.6%. PIAR debe loggear
  distribución de longitudes de spans ReAct y decidir empíricamente
  (D.5 inclinada).

### Math-Shepherd — Process Reward Models (Wang et al. 2024)
[notas](../notes/paper-math-shepherd.md) · [#8](https://github.com/lucaspecina/piar-rl/issues/8) (cerrado)

**Contexto histórico, lectura liviana.**

- **Hereda**: el problema que ataca (credit assignment per-paso sin step
  labels manuales).
- **Cambia**: Math-Shepherd resuelve con MC step labels (rollout de N
  completions desde cada prefix). ~38× más caro que implicit PRM
  (Yuan/iStar/PIAR).
- **Delta**: PIAR resuelve el mismo problema con costo computacional
  dramáticamente menor (1 forward pass extra por step).
- **Por qué importa**: define la línea "explicit and expensive" que
  Yuan/iStar/PRIME/PIAR superan. Útil como contexto histórico para
  argumentar **por qué** el implicit PRM (y por tanto PIAR) es la dirección
  correcta. **No aporta primitivas** a PIAR.

### π-Distill — Privileged Information Distillation (Penaloza et al. 2026)
[notas](../notes/paper-pi-distill.md) · [#10](https://github.com/lucaspecina/piar-rl/issues/10) (cerrado)

**El vecino conceptual más cercano matemáticamente y operativamente.** PIAR
sin entrenar al teacher.

- **Hereda**: el setting más cercano matemáticamente — PI teacher + agentic
  multi-turn. Mismo spirit que PIAR.
- **Cambia**: π-Distill entrena al teacher (α > 0 sobre el teacher, en su
  lenguaje) y usa KL distillation forward como loss. PIAR mantiene al
  teacher frozen y usa log-ratio como step reward de GRPO.
- **Delta**: PIAR es π-Distill **sin entrenar al teacher**, con log-ratio
  reward en vez de KL distillation.
- **Caveats útiles**:
  - π-Distill provee taxonomía de PI types (Tool Calls & Args / Tool Calls
    Only / Self-Generated Hints). PIAR adopta PI denso (spec estructurada
    en WebShop) por la misma evidencia que π-Distill encontró: PI menos
    denso requiere entrenar el teacher para funcionar.
  - π-Distill observa empíricamente leakage textual sobre tokens
    específicos — refuerza D.1 + D.9 + D.7 candidata en PIAR.
- **El vecino más peligroso**: si π-Distill libera código (placeholder al
  2026-05, sin training code todavía), conviene comparar directo.

## La celda vacía — mapa visual

```
                        Multi-turn agentic
                              │
                              │  iStar       π-Distill       PIAR ◄─── celda vacía
                              │  (log-ratio  (KL distill +    (log-ratio
                              │   sin PI)     trained PI)      frozen + PI)
                              │
              ────────────────┼────────────────────────────────────────
                              │
                       OPSD   │
                       (KL +  │
                       frozen │
                       PI)    │
                              │
                  Yuan        │
                (math base)   │
                              │
                        Single-turn math
```

**Ejes del mapa**:
- **Vertical**: setting (math single-turn ↔ agentic multi-turn).
- **Horizontal implícito**: tipo de privilegio (sin PI / PI en pesos
  entrenados / PI en prompt frozen).

PIAR ocupa: **multi-turn agentic × PI en prompt × frozen**. Antes de PIAR esa
celda no tiene paper.

## Lo que la literatura NO dice sobre PIAR

Tres preguntas que **ningún paper vecino responde**, y que el experimento de
PIAR debe decidir empíricamente:

1. **¿El log-ratio context-induced en multi-turn agentic correlaciona con
   progreso causal o solo con compatibilidad textual?** OPSD validó frozen+PI
   en single-turn math; iStar validó log-ratio en multi-turn pero sin PI.
   Nadie hizo el cross. → resuelve con D.1 + D.9.
2. **¿La info privilegiada en prompt iguala o supera al PRM entrenado con
   outcomes?** La apuesta central de PIAR vs iStar. → resuelve con el
   experimento primario PIAR vs iStar.
3. **¿`π_old` vs frozen θ₀ cambia algo en multi-turn?** OPSD validó frozen
   en single-turn KL; PIAR primario usa `π_old` por argumento de pureza de
   invariante 4. → resuelve con ablation directo (E.3).

## Riesgos científicos heredados de la literatura

Cuatro riesgos que la literatura ya identificó y que PIAR no puede ignorar:

1. **Leakage textual** (Yuan §4, §8 / π-Distill §9.1): el log-ratio puede
   capturar compatibilidad textual con el golden, no progreso causal. →
   controles D.1 (muestreo pareado) + D.9 (shuffled-spec).
2. **Dominio de tokens stylistic** (OPSD §6, §13.3): sin pointwise clipping,
   tokens como headers/delimitadores dominan el reward. → D.2 (loggear
   distribución por tipo de token).
3. **Varianza alta por longitud de span** (SWEET-RL §6, §10.1): sin $1/L$
   normalization, acciones largas dominan trivialmente. → D.5 (loggear
   longitudes; decidir data-driven).
4. **PRM evaluador ≠ buen actor** (Yuan §6): un teacher que puntúa bien no
   necesariamente sirve para mejorar el policy. → E.5 (diagnóstico cuando
   arranquen experimentos).

## Cierre — el plan después de la síntesis

Con esta síntesis cerrada, las próximas fases (ver PROJECT.md "Roadmap
conceptual"):

- **Fase 2**: setup compute Azure ML Y-TEC.
- **Fase 3** (#16): replicar baseline iStar en WebShop con `CharacterRL-iStar`.
- **Fase 4**: implementar PIAR sobre Plan B (#14 cerrado en favor de Plan B).
- **Fase 5**: comparación + ablations.
  - Críticas: D.1 + D.9 (leakage), D.5 (length norm), C.5 ablation A/B/trayectorias humanas.
  - Secundarias: E.3 (`π_old` vs frozen θ₀), D.2 (tipos de token).
- **Fase 6**: 2do benchmark (Sokoban primero, después candidatos externos) + draft del paper.

**LA PREGUNTA, ahora completamente operacional** (PROJECT.md "Reformulación
operativa contra iStar"):

> *iStar codifica información privilegiada (outcome rankings) en los pesos de
> un juez separado vía DPO. ¿Esa info puede reemplazarse por información
> privilegiada en el contexto del mismo modelo (golden answer en el prompt),
> produciendo igual o mejor señal per-paso con un setup más simple (sin
> entrenar juez)?*

---

## Pivot 2026-06 — mapa actualizado y veredicto de novedad

> Agregado 2026-06-12. Fuente de verdad: [`pivot-2026-06.md`](pivot-2026-06.md).
> El ciclo de research del 2026-06-12 reveló que la fase 1 tenía un punto
> ciego (la línea information-gain y los desarrollos feb–jun 2026). Esta
> sección reemplaza el posicionamiento de "La celda vacía" de arriba.
> Paper notes nuevos: [`paper-igpo.md`](../notes/paper-igpo.md),
> [`paper-tamtrl.md`](../notes/paper-tamtrl.md),
> [`paper-survey-ca.md`](../notes/paper-survey-ca.md),
> [`paper-gigpo.md`](../notes/paper-gigpo.md),
> [`paper-knowrl.md`](../notes/paper-knowrl.md),
> [`paper-menores-pivot-2026-06.md`](../notes/paper-menores-pivot-2026-06.md).

### Mapa de literatura actualizado (vecinos nuevos)

| Familia | Métodos clave | Mecanismo | Relación con PIAR |
|---|---|---|---|
| Juez entrenado | AgentPRM, SWEET-RL, iStar, StepAgent | Segundo modelo (o pesos extra) puntúa pasos | Baseline (iStar). SWEET-RL = privilegio en critic entrenado |
| Juez prompteado / hindsight | HCAPO¹, C3, CCPO, CriticSearch | LLM congelado evalúa post-hoc y emite señal por turno | CriticSearch usa PI (gold answer) en el prompt del crítico → vecino del forward, pero generativo/opinado |
| Estadística entre rollouts | **GiGPO** (NeurIPS 2025), POAD, SLEA-RL | Agrupa acciones desde el mismo estado entre rollouts → advantage por paso, sin juez | **Baseline obligatorio nuevo (N.10, brazo B2).** Limitación: degrada a GRPO sin estados repetidos entre rollouts — la dimensión exacta donde PIAR predice separarse |
| Information-theoretic | **IGPO** (ICLR 2026), IG semántico (2602.00845) | Δ log π(golden \| historial) como reward por turno | = nuestro backward, validado solo en search/QA. IGPO v2 renombró a "Search Agents" → ejecución quedó fuera de su claim (brazo A2 libre) |
| Privileged context como loss | OPSD, π-Distill, **OPCD/OEL** (2026) | Teacher = mismo modelo + contexto; supervisa vía KL | Misma asimetría que el forward, canalizada como distillation, no como reward en el advantage |
| Teacher same-model como reward | **TAMTRL** (2603.21663) | Mismo modelo, dos contextos (chunk filtrado por GT); prob del teacher por turno, min-max normalizada, gateada por outcome binario², como reward shaping multi-turn | **El vecino que mata al forward standalone.** Difiere en: prob del teacher a secas (no ratio teacher/student), PI = documento filtrado (no golden estructurada), dominio = compresión long-context (no agentes de decisión), sin descomposición dual |
| Hints al student | QuestA, KnowRL³, BREAD, Prefix-RFT, UFT, Guide, Scaf-GRPO, AHRL (ERNIE 5.0) | Solución parcial en el prompt DEL STUDENT durante rollout | Familia estructuralmente opuesta al residual: contaminan la política. KnowRL es el vecino conceptual más cercano (guía mínima-suficiente, descomposición atómica) pero del lado del student, en math single-turn |

Correcciones post-verificación de los papers (2026-06-12), donde la nota
difiere de la tabla original del pivot doc:

1. **HCAPO** no es un juez generativo que "razona y opina": computa
   **logprobs mecánicos del mismo modelo bajo contexto de hindsight**
   (verificado dos veces, 2026-06-12 — la segunda lectura a pedido de review
   externo corrigió además la primera: la implementación real, Eq. 7, usa
   **un solo contexto** normalizado contra la media intra-trayectoria, no el
   ratio dos-contextos de su forma teórica Eq. 5). Es el **vecino mecánico
   #1 del forward** — por encima de TAMTRL en prioridad de related work —
   pero el delta de PIAR es nítido: PI fáctica del dataset vs outcome
   auto-generado en hindsight; contraste explícito con/sin-PI vs score de un
   contexto; aditivo en advantage vs multiplicativo en return; dual+residual
   vs nada epistémico. Sus números: ALFWorld 91.4±2.3 / WebShop 73.8±2.8 con
   Qwen2.5-7B — pero **vs GiGPO la ganancia es +0.6 con barras solapadas**
   (el +13.8 es vs GRPO vanilla), el split seen/unseen no está desambiguado,
   y es preprint sin código. ALFWorld >90% en esa config → alerta de
   saturación (N.13). Ver
   [`paper-menores-pivot-2026-06.md`](../notes/paper-menores-pivot-2026-06.md) §1.
2. **TAMTRL** además gatea el reward multiplicativamente por el outcome
   binario (trayectorias fallidas reciben 0 en todos los turnos) — quinta
   diferencia explotable: PIAR da señal densa independiente del outcome. Ver
   [`paper-tamtrl.md`](../notes/paper-tamtrl.md).
3. **KnowRL no tiene retirada por schedule** — fija subsets mínimos offline
   (minimalidad estática). La caracterización precisa de la familia hints:
   manejan el train/test mismatch *por schedule (QuestA-style) o por
   minimalidad estática (KnowRL-style)*; ninguna lo elimina, porque la info
   entra a la política. Ver [`paper-knowrl.md`](../notes/paper-knowrl.md) §7.4.

**Distinción estructural que define al residual** (related work + invariante
11 propuesto): en toda la familia de hints, la información entra al *prompt
del student* y por lo tanto a la política → hay train/test mismatch del input
y hace falta annealing programado (o minimalidad estática). En PIAR la PI
entra **solo al canal del reward** (prompts del scorer): el student rollea
siempre a ciegas, no hay nada que retirar del input, y lo que se apaga (solo)
es la señal.

### Veredicto de novedad (2026-06-12)

| Pieza | Estado | Evidencia |
|---|---|---|
| Forward solo (golden completa en prompt del scorer, log-ratio como step reward) | **Muerto como contribución** | TAMTRL + OPCD + CriticSearch (+ HCAPO con outcome en hindsight) |
| Backward solo (Δ belief hacia golden) | **Tomado** | IGPO (+ sucesor IG semántico 2602.00845) |
| Dual (forward + backward como descomposición pragmático/epistémico) | **Libre** | Ausente en survey 2604.09459 (41 core + 6 enablers) y búsquedas dirigidas 2026-06-12. El propio survey declara el gap en su §4.2: "most current CA methods do not explicitly address this distinction" entre decision errors / information gaps / exploratory actions |
| PI residual (gating por belief, lado scorer) | **Libre** | Familia hints es lado-student (schedule o minimalidad estática); PI-dropout robótica es random/schedule sobre critic entrenado |
| Mapa dónde/por qué (por environment) | **Libre** | Gap §4.2 del survey + "CA meets exploration" (§9.2) como frontera adyacente |
| Análisis de leakage de scores privilegiados (D.1/D.9) | Sigue libre y ahora aplica a *dos* direcciones | Ningún vecino lo hizo a fondo |

Implicancia de posicionamiento: TAMTRL, OPCD, CriticSearch, HCAPO e IGPO
pasan de "amenazas" a **related work que valida los componentes**. El claim
del paper ya no es ningún componente: es la descomposición, la dosificación
y el mapa.

Caveat de vigilancia (riesgo de scoop, pivot §10.5): **TAMTRL, la familia
hints y OPSD/OPCD no aparecen ni en el survey ni en el repo
Awesome-Credit-Assignment** → el repo Awesome solo NO alcanza como mecanismo
de re-check mensual; hace falta búsqueda directa + alertas de citas a IGPO y
TAMTRL.
