# Research

Centraliza el conocimiento que vamos acumulando sobre PIAR y sus papers vecinos.

## Estructura

```
research/
├── notes/         # Dumps de lectura, debates, exploración (efímeros, pueden borrarse)
├── synthesis/     # Conclusiones consolidadas (canon, perdura)
├── examples/      # Ejemplos canónicos worked-out (referencia activa)
└── archive/       # Notas obsoletas o superadas
```

## Reglas de promoción

```
exploración/lectura      diseño/síntesis        decisión de proyecto
research/notes/      ->   research/synthesis/ ->  PROJECT.md (Invariantes / Roadmap)
                                              \
                                               -> ejemplos worked-out -> research/examples/
notes obsoletos                              ->  research/archive/
```

## Convenciones de naming

- `notes/`: `paper-<short-name>.md` (ej. `paper-yuan-implicit-prm.md`),
  `topic-<tema>.md` para temas no atados a un paper, `debate-<tema>.md` para debates.
- `synthesis/`: `<tema>-conclusion.md` o `<tema>-synthesis.md`. Ej: `papers-cross-mapping.md`.
- `examples/`: `<concepto>-example.md`. Pensado como referencia worked-out concreto.
- `archive/`: nombre original con sufijo de fecha si ayuda.

## Cross-linking con GitHub Issues

- Body de issue → linkea a `research/synthesis/<doc>.md` cuando la justificación vive ahí.
- Comentario de issue → linkea a `research/notes/<scratch>.md` si hubo debate.
- Docs de synthesis → citan los issues que motivaron / cerraron.

## Estado actual

Vacío. Las notas empiezan a aparecer cuando se ejecuten los issues del epic
"Research — síntesis de papers vecinos" (ver [Project v2 #5](https://github.com/users/lucaspecina/projects/5)).
