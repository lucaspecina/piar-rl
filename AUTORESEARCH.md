# Autoresearch

## Status: OFF

Cuando se active, llenar acá:

```
## Run actual
- **Issues / Epic:** #N (epic) y/o #N1, #N2, #N3 (sub-issues concretos)
- **Objetivo:** [descripción]
- **Branch base:** [branch desde donde se crea autoresearch/...]
- **Max experimentos:** N (si aplica)
- **Stop conditions:**
  - [condición 1]
  - [condición 2]
  - [condición 3]
- **Política de commits:** [frecuencia de commit/push]
- **Paths permitidos:** [paths que se pueden modificar]
```

Ver user-level `dev-workflow/autoresearch.md` para el protocolo completo.

## Notas

- Mientras el proyecto esté en fase research/papers, el uso natural de
  autoresearch es dejar a Claude leyendo y sintetizando varios papers en una
  sesión larga. Cada paper = un sub-issue del epic de research. Stop condition
  típica: "todos los sub-issues del scope cerrados con synthesis en
  `research/synthesis/`".
- Cuando arranque la fase de código, autoresearch se vuelve más interesante
  para loops de experiment → analyze → refinar hipótesis → siguiente
  experimento, con commits/pushes según política.
- En autoresearch los issues del scope se mueven a `In Progress` al empezar y
  se cierran con `gh issue close --reason completed --comment "..."` al
  terminar. El comentario sobrevive compactación de contexto.
