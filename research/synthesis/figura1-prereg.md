# Pre-registro — Figura 1 (go/no-go del método dual, sin RL)

> **CONGELADO 2026-06-12 — [#23](https://github.com/lucaspecina/piar-rl/issues/23) cerrado.**
> Umbrales fijados con los defaults del review externo (2026-06-12), bajo
> delegación explícita de Lucas ("son defaults razonables... ajustalos a tu
> gusto, pero fijalos antes de mirar un solo rollout" + aprobación general).
> Cualquier cambio requiere editar este doc **antes de generar el primer
> rollout**, en commit propio y explícito. Una vez corrido el experimento,
> este doc NO se edita: los desvíos se reportan contra esta versión (ese es
> el punto del pre-registro).
>
> **Refs:** pivot §8 (predicciones), §4 (formalización de los scores),
> [`pi-webshop.md`](pi-webshop.md) (componentes g_i y wrappers — dependencia
> dura: sin #17 cerrado no se pueden computar los b_i).

---

## 1. Objetivo

Verificar, **sin entrenar nada**, que la descomposición
pragmático/epistémico existe en WebShop: que el forward y el backward miden
cosas distintas y complementarias sobre trayectorias congeladas, y que el
backward trackea valor real. Es el gate de la fase de implementación: si la
descomposición no aparece acá, no hay método que entrenar.

## 2. Setup

| Ítem | Valor congelado |
|---|---|
| Modelo | Qwen2.5-7B-Instruct (shakedown previo del pipeline con Qwen2.5-1.5B-Instruct; los números que cuentan son los del 7B) |
| Environment | WebShop, mismo config que `code/examples/istar_trainer/run_webshop.sh` (max_steps incluido — reconciliar el 10 vs 15 señalado en [`paper-gigpo.md`](../notes/paper-gigpo.md) §8 ANTES de rollear) |
| Rollouts | ~200 episodios del modelo base, **temperatura, N por tarea y set de tareas fijados con seed ANTES de rollear** (default: temperatura del script de iStar; set = sample uniforme seeded del split de training). **Cero filtrado manual de episodios.** Congelados a disco antes de computar ningún score |
| Outcome | **Score continuo de WebShop** (no éxito binario) — los rollouts del modelo base pueden tener poca varianza binaria |
| Scorer | El mismo checkpoint base (π_old = θ_base — no hay training, así que el invariante 4 es trivial acá) |
| PI | Componentes g_i + wrappers de [`pi-webshop.md`](pi-webshop.md) §2–§4 |
| Seeds | Fijadas y loggeadas en `experiments/E001/manifest.yaml` |

Las tres lecturas a computar por turno sobre cada trayectoria congelada:

1. **r_fwd full-golden** (pivot §4.3 con R(t) = g completa — versión brazo A1).
2. **r_bwd** (pivot §4.2): Σ_i [b_i(t) − b_i(t−1)].
3. **r_fwd residual** (pivot §4.3 con gating τ): para la lectura del canal.
   **τ = percentil empírico por tipo de componente** (no umbral universal —
   mitiga el mass-splitting de [`pi-webshop.md`](pi-webshop.md) §4.1),
   **condicionado a que P5 pase**; percentil fijado: **p50 por tipo**.

## 3. Clasificación de acciones

Cada acción se clasifica con reglas determinísticas sobre el texto de la
acción (sin LLM judge):

| Clase | Regla |
|---|---|
| `search` | la acción es `search[...]` |
| `click-nav` | `click[...]` sobre links de navegación/paginación/producto (ni opción ni buy) |
| `click-opcion-buy` | `click[...]` sobre una opción del producto o `click[buy now]` |

Sub-clasificaciones para las confirmatorias within-type (igual de
determinísticas, **reglas congeladas acá, antes de mirar un solo rollout**):

| Sub-clase | Regla |
|---|---|
| click **correcto** / **incorrecto** | click de producto: el ASIN del link == `goal['asin']` o no. Click de opción: el valor normalizado (lowercase, `normalize_color`) ∈ valores de `goal['goal_options']` o no |
| acción **informativa** / **no informativa** | la observación resultante contiene la forma canónica (matching textual case-insensitive) de ≥1 componente g_i que NO había aparecido en observaciones previas del episodio / no agrega ninguno |

## 4. Predicciones pre-registradas

**v2 (2026-06-12, post-review externo):** las comparaciones *between-type*
originales (P1/P2 del pivot §8) tienen un confound sintáctico — un `search`
es texto libre largo, un `click` es formato fijo corto; las medianas pueden
separarse por construcción del span y no por semántica. Las
**confirmatorias** pasan a ser contrastes *within-type* (mismo formato, pura
semántica); las between-type quedan como **descriptivas** (se reportan, no
gatean).

### 4.1 Confirmatorias (gatean; Mann-Whitney, α = 0.05 con Bonferroni sobre {P1', P2', P3})

| # | Predicción | Criterio cuantitativo | Qué valida |
|---|---|---|---|
| P1' | El forward distingue semántica dentro del mismo formato | mediana r_fwd(click sobre el producto target, ASIN del goal) > mediana r_fwd(click sobre otro producto); ídem opciones: r_fwd(click de opción ∈ goal_options) > r_fwd(click de opción ∉) | El forward mide corrección de la decisión, no longitud/sintaxis del span |
| P2' | El backward premia las acciones que revelan información | mediana r_bwd(acciones cuya observación resultante contiene la forma canónica de algún g_i todavía no aparecido) > mediana r_bwd(acciones cuya observación no agrega ningún g_i) — clasificación determinística por matching textual sobre la observación, regla escrita antes de mirar datos | La direccionalidad epistémica existe a igualdad de sintaxis |
| P3 | El belief amortizado trackea valor | **Spearman parcial** (Σ_t r_bwd vs score continuo final, **controlando longitud de trayectoria**) **ρ > 0.3**. **Robustness obligatoria**: la correlación **excluyendo el último turno** mantiene signo y significancia (anti-circularidad: si compraste el item correcto, la página final lo contenía y el belief final es alto por definición) | b_i no es ruido; el gating tiene base (N.9 parcial) |
| P5 | El belief separa lo sabido de lo no sabido | Para cada tipo de componente: **AUC > 0.8** separando b_i de componentes **ya aparecidos verbatim (forma canónica) en observaciones del episodio** vs no aparecidos. **Se corre para las dos variantes de inyección del wrapper** (assistant-prefill y user-message, `pi-webshop.md` §4); el gate pasa si **al menos una** supera el umbral, y esa variante queda fijada para training (regla de selección pre-registrada — agregada 2026-06-12 en commit propio, antes de cualquier rollout) | La premisa del gating. Si falla en ambas variantes, el gating por belief se rediseña ANTES de gastar GPU (fallback: gating observacional, `pi-webshop.md` §4.2) |

### 4.2 Control existencial

| # | Predicción | Criterio | Qué valida |
|---|---|---|---|
| P4 | Nada de esto es leakage textual | Con shuffled-golden (g de otro episodio, mismo formato): para cada confirmatoria, el **efecto shuffled es < 1/3 del efecto real Y no significativo (α = 0.05)** | D.9 aplicado a ambas direcciones |

### 4.3 Descriptivas y exploratorias (se reportan, no gatean)

- P1/P2 between-type originales del pivot §8 (search vs click-opción/buy).
- Distribución de b_i(0) y b_i(T) por tipo de componente (calibra el
  percentil de τ).
- Cuasi-ortogonalidad de los g_i (correlación entre Δb_i de componentes
  distintos — caveat "pruning interaction paradox" de KnowRL).
- Fracción de turnos con R(t) = ∅ al final de episodios de score alto (¿el
  auto-annealing llegaría a activarse?).
- **Predicciones derivadas de los datos reales del dataset**
  (`pi-webshop.md` §8, agregadas 2026-06-12 pre-rollouts): b_attr/b_opt
  altos ya en t=0 (los attrs/options son visibles en la instrucción del
  student); el grueso de Σr_bwd viene de Δb_prod (la info genuinamente
  oculta). Si esto sale, cuantifica el mapa dónde/por qué para WebShop.
- **N.14 (ventana de historial)**: en el scoring offline el historial
  completo está disponible — computar b_i con (a) la ventana-2 que la
  política realmente ve en training (`env_manager.py:392`) y (b) historial
  completo. Reportar cuántas caídas de b_i se explican por
  salida-de-ventana vs navegación real. Decide N.14 con datos.

### 4.4 Alcance (honestidad pre-registrada)

**La Figura 1 valida que las señales discriminan sobre trayectorias
congeladas; NO valida que mejoren el RL.** Es condición necesaria, no
suficiente — el efecto en training se decide en los brazos A1–A4. Queda
dicho acá antes de que lo diga un reviewer.

## 5. Árbol de decisión (cada rama tiene salida — pivot §8, v2)

- **P1', P2', P3, P5 pasan y P4 colapsa como debe** → Figura 1 del paper +
  luz verde a la implementación (fase 5 del roadmap propuesto).
- **P4 falla** (los efectos sobreviven al shuffle) → parar todo: el método
  mide afinidad textual condicionada a la golden, no causalidad. Resultado
  negativo limpio (D.9).
- **P5 falla** → la premisa del gating por belief no se sostiene (el nivel
  absoluto de b_i no separa sabido/no-sabido — mass-splitting u otra causa).
  NO se gasta GPU en el residual por belief: se rediseña el gate. Fallback
  pre-registrado y determinístico: **gating observacional**
  ([`pi-webshop.md`](pi-webshop.md) §4.2) — el componente sale del residual
  cuando su forma canónica apareció en una observación. El dual (A3) sigue
  vivo independiente de esto.
- **P3 falla** (o pasa solo gracias al último turno) → el belief amortizado
  no trackea valor. NO entrenar. Abrir la investigación de calibración N.9
  completa (correlacionar b_i contra values Monte Carlo estilo Math-Shepherd
  en subset chico) antes de cualquier otra cosa.
- **P1' o P2' fallan** → la descomposición no existe en WebShop a nivel
  semántico. Dos lecturas a desambiguar con los logs: (a) WebShop tiene poca
  información oculta (consistente con el mapa dónde/por qué — probar
  ALFWorld antes de descartar), (b) los wrappers/g_i están mal diseñados
  (revisar pi-webshop.md §4.1). Ninguna habilita implementación todavía.

## 6. Qué NO se decide con la Figura 1

- λ (N.5), normalización separada/conjunta (N.7), gating duro/soft (N.4) —
  son decisiones de training, van al grid pre-registrado de fase de
  comparación.
- La forma final de τ sí se **calibra** acá (§2) pero su validación es en
  training.

## 7. Requisitos previos

1. #17 cerrado (g_i + extractor adaptado al formato por componentes).
2. Compute: alcanza una GPU (scoring de prefills, sin training) — la VM spot
   sobra; checkpointing del scoring por lotes igual (spot).
3. `experiments/E001/manifest.yaml` con commit SHA, seeds, config del env,
   y los artefactos g_i por episodio (invariante 5).
