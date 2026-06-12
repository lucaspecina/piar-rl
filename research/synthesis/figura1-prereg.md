# Pre-registro — Figura 1 (go/no-go del método dual, sin RL)

> **DRAFT para review de Lucas — [#23](https://github.com/lucaspecina/piar-rl/issues/23).**
> Los umbrales marcados `[LUCAS]` los fija él; el resto queda congelado al
> mergear este doc. Una vez corrido el experimento, este doc NO se edita:
> los desvíos se reportan contra esta versión (ese es el punto del pre-registro).
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
| Rollouts | ~200 episodios del modelo base, greedy-off (temperature del script de iStar), **congelados a disco antes de computar ningún score** |
| Scorer | El mismo checkpoint base (π_old = θ_base — no hay training, así que el invariante 4 es trivial acá) |
| PI | Componentes g_i + wrappers de [`pi-webshop.md`](pi-webshop.md) §2–§4 |
| Seeds | Fijadas y loggeadas en `experiments/E001/manifest.yaml` |

Las tres lecturas a computar por turno sobre cada trayectoria congelada:

1. **r_fwd full-golden** (pivot §4.3 con R(t) = g completa — versión brazo A1).
2. **r_bwd** (pivot §4.2): Σ_i [b_i(t) − b_i(t−1)].
3. **r_fwd residual** (pivot §4.3 con gating τ): para la lectura del canal;
   τ inicial = mediana de b_i(0) sobre los 200 episodios `[LUCAS: confirmar
   esta regla de calibración o fijar τ a mano]`.

## 3. Clasificación de acciones

Cada acción se clasifica con reglas determinísticas sobre el texto de la
acción (sin LLM judge):

| Clase | Regla |
|---|---|
| `search` | la acción es `search[...]` |
| `click-nav` | `click[...]` sobre links de navegación/paginación/producto (ni opción ni buy) |
| `click-opcion-buy` | `click[...]` sobre una opción del producto o `click[buy now]` |

## 4. Predicciones pre-registradas

| # | Predicción | Criterio cuantitativo | Qué valida |
|---|---|---|---|
| P1 | El forward castiga informarse y premia ejecutar | mediana r_fwd(search) < mediana r_fwd(click-opcion-buy), Mann-Whitney p < 0.05 `[LUCAS: confirmar test y α]` | El hindsight bias del forward existe → el backward no es opcional |
| P2 | El backward premia informarse y es ciego a ejecutar | mediana r_bwd(search) > mediana r_bwd(click-opcion-buy) ≈ 0, mismo test | La direccionalidad epistémica existe |
| P3 | El belief amortizado trackea valor | Spearman(Σ_t r_bwd, score final del episodio) ρ > 0.3 `[LUCAS: fijar el número final — 0.3 es el propuesto del pivot §8]` | b_i no es ruido; el gating tiene base (N.9 parcial) |
| P4 | Nada de esto es leakage textual | Con shuffled-golden (g de otro episodio, mismo formato), P1–P3 colapsan: efectos < 50% del tamaño original y/o pierden significancia `[LUCAS: confirmar criterio de colapso]` | D.9 aplicado a ambas direcciones |

Métricas secundarias (se loggean, no gatean): distribución de b_i(0) y
b_i(T) por tipo de componente (τ por tipo, `pi-webshop.md` §4.1);
cuasi-ortogonalidad de los g_i (correlación entre Δb_i de componentes
distintos — caveat KnowRL); fracción de turnos con R(t) = ∅ al final de los
episodios exitosos (¿el auto-annealing llegaría a activarse?).

## 5. Árbol de decisión (cada rama tiene salida — pivot §8)

- **P1–P3 pasan** → Figura 1 del paper + luz verde a la implementación
  (fase 5 del roadmap propuesto). P4 debe pasar también; si P4 falla, parar
  todo: el método mide afinidad textual (resultado negativo limpio, D.9).
- **P3 falla** → el belief amortizado no trackea valor. NO entrenar. Abrir
  la investigación de calibración N.9 completa (correlacionar b_i contra
  values Monte Carlo estilo Math-Shepherd en subset chico) antes de cualquier
  otra cosa.
- **P1 o P2 fallan** → la descomposición no existe en WebShop. Dos lecturas
  posibles a desambiguar con los logs por tipo de acción: (a) WebShop tiene
  poca información oculta (consistente con el mapa dónde/por qué — probar
  ALFWorld antes de descartar), (b) los wrappers/g_i están mal diseñados
  (revisar §4.1 de pi-webshop.md). Ninguna habilita implementación todavía.

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
