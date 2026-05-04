# Experiments

Cada experimento significativo vive en `experiments/ENNN-slug/` con un
`manifest.yaml` obligatorio que garantiza reproducibilidad (hipótesis,
parámetros, commit SHA, seeds, métricas, conclusión, ref a issue).

Ver `dev-workflow/experiment-manifest.md` (skill `dev-workflow`) para el template.

## Estado actual

Vacío. Todavía no hay experimentos formales. El primero será cuando se replique
el baseline de iStar en WebShop (ver TODO.md `LATER`).

## Numeración

`ENNN` secuencial global. Ejemplo: `E001-istar-webshop-baseline`,
`E002-piar-webshop-v0`, etc.
