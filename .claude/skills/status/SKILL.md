---
name: status
description: Show a quick overview of the project state — current phase, active issues, recent changes, and what's next. Use when the user asks "¿en qué estamos?", "¿qué hicimos?", "status", or wants a quick situational awareness check.
---

# Project status overview

Da un overview rápido del estado actual de PIAR. Pensado para responder
"¿en qué estamos?" sin tener que abrir 5 docs.

## Pasos

1. **Fase actual.** Leer `CURRENT_STATE.md` y resumir en 2-3 líneas qué existe hoy.
2. **Activos.** Listar los issues con `status: active` (o `open` con trabajo
   reciente) en `issues/` y mostrar el Status header de cada uno.
3. **NOW / NEXT.** Mostrar las secciones NOW y NEXT de `TODO.md`.
4. **Últimos cambios.** Mostrar el `git log --oneline -5` del repo.
5. **Pregunta abierta.** Si hay un issue activo, indicar el "Próximo paso" de
   su Status header.

## Output esperado (formato)

```
## PIAR — Status

**Fase actual:** [research / implementación / experimentación]
[2-3 líneas resumen de CURRENT_STATE]

**Issues activos:**
- I-NNN — [título] — [próximo paso del Status]

**NOW (desde TODO.md):**
- [item 1]
- [item 2]

**NEXT (desde TODO.md):**
- [item 1]

**Últimos commits:**
- [SHA] — [mensaje]
- ...

**Próximo paso sugerido:** [recomendación basada en lo de arriba]
```

## Notas

- Output siempre en español.
- Si hay autoresearch ON, mencionarlo prominentemente arriba.
- No abrir issues cerrados a menos que el usuario pregunte explícitamente por
  historia.
