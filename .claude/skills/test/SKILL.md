---
name: test
description: Run the project's test suite. Use when the user wants to validate code changes or verify nothing is broken. Accepts an optional filter argument (e.g., "/test test_logratio").
disable-model-invocation: true
---

# Run tests

## Status actual del proyecto

PIAR está en fase research/papers — **todavía no hay código ni tests**. Cuando
arranque la fase de implementación, este skill ejecutará el test suite.

## Plan tentativo

Cuando exista código:

1. Detectar el framework (probable: pytest sobre `tests/`).
2. Si hay `$ARGUMENTS`, pasarlos como filtro (`pytest -k "$ARGUMENTS"`).
3. Sin argumentos: correr todo el suite.
4. Reportar resultados al usuario en español: pasados / fallados / errores.

## Comando placeholder

```bash
# Cuando exista pyproject.toml + tests:
# pytest tests/ $ARGUMENTS
echo "PIAR está en fase research. No hay tests todavía."
echo "Ver TODO.md para el plan de implementación."
```

## Cuando se concrete el environment

Actualizar este skill con:
- Comando exacto (pytest, uv run pytest, conda run pytest, etc.).
- Path al directorio de tests.
- Marker conventions (slow, gpu, integration, etc.).
- Cómo correr smoke tests del pipeline real (Level 2 QA).
