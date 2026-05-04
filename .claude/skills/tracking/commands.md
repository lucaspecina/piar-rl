# Tracking commands — recipes por situación

Cargar IDs desde `reference.md` antes de correr cualquier comando que toque el Project.

> **Shell asumido: Git Bash en Windows** (lo que viene con Git for Windows). Si
> usás PowerShell, adaptar: `$env:VAR` en vez de `VAR=`, `Set-Content` en vez
> de heredocs, etc. Los comandos `gh`, `gh api` y `gh project` funcionan igual
> en ambos. El gotcha del leading slash en endpoints (`repos/...` en vez de
> `/repos/...`) es específico de Git Bash; en PowerShell no aplica.

## 1. Crear issue nueva (flujo completo)

```bash
# (1) Escribir el body con el template de 3 secciones
cat > /tmp/body.md <<'EOF'
## Contexto (para humanos)
<...>

## Detalle técnico (para Claude / sesiones)
<...>

## Criterio de cierre
<...>
EOF

# (2) Crear la issue (devuelve URL). Labels opcionales.
URL=$(gh issue create --title "..." --body-file /tmp/body.md --label research | tail -1)
N=$(echo "$URL" | grep -oE '[0-9]+$')

# (3) Agregar al Project board y capturar item ID
ITEM_ID=$(gh project item-add 5 --owner lucaspecina --url "$URL" --format json --jq '.id')

# (4) Status queda en `Todo` por default. Si querés moverlo de una a `In Progress` ver flujo #2.

# (5) Si es sub-issue de un epic: linkear (sub_issue_id es databaseId, NO number)
# OJO: `gh issue view --json id --jq '.databaseId'` NO devuelve el databaseId entero
# (el flag --json id da el GraphQL global ID). Usar GraphQL directo:
CHILD_ID=$(gh api graphql -f query="query { repository(owner:\"lucaspecina\",name:\"piar-rl\") { issue(number:$N) { databaseId } } }" --jq '.data.repository.issue.databaseId')
# OJO también: en Git Bash en Windows, el endpoint sin leading slash evita
# que el shell convierta `/repos/...` en una path local.
gh api -X POST repos/lucaspecina/piar-rl/issues/<EPIC>/sub_issues \
  -F sub_issue_id=$CHILD_ID
```

## 2. Empezar a trabajar un issue (Status -> In Progress)

```bash
# Obtener item_id por número de issue
ITEM_ID=$(gh api graphql -f query='
  query { user(login:"lucaspecina") { projectV2(number:5) {
    items(first:100) { nodes { id content { ... on Issue { number } } } }
  } } }' --jq ".data.user.projectV2.items.nodes[] | select(.content.number==<N>) | .id")

gh project item-edit --project-id $PROJECT_ID --id $ITEM_ID \
  --field-id $STATUS_FIELD_ID --single-select-option-id $STATUS_IN_PROGRESS
```

## 3. Cerrar issue (completar)

```bash
gh issue close <N> --reason completed \
  --comment "Se completó: <qué se hizo>. <link al PR mergeado / link a research/synthesis/...>"
# Project mueve item a Done automáticamente.
```

## 4. Cerrar como "not planned" (descartar)

```bash
gh issue close <N> --reason "not planned" \
  --comment "Scope change: <explicar>"

# Además, remover del Project para que no polucione Done:
gh api graphql -f query="mutation {
  deleteProjectV2Item(input: {projectId: \"$PROJECT_ID\", itemId: \"$ITEM_ID\"}) { deletedItemId }
}"
```

## 5. Reabrir issue (volver a Todo)

```bash
gh issue reopen <N>
gh project item-edit --project-id $PROJECT_ID --id $ITEM_ID \
  --field-id $STATUS_FIELD_ID --single-select-option-id $STATUS_TODO
```

## 6. Promover sub-issue a epic

Usar cuando un sub-issue crece a necesitar 3+ sub-sub-items.

```bash
# (1) Renombrar título al formato Epic
gh issue edit <N> --title "Epic · <meta concreta>"

# (2) Reescribir body para reflejar criterio de cierre del epic
gh issue edit <N> --body-file /tmp/new_body.md

# (3) Unlink del epic padre anterior (si existía)
CHILD_ID=$(gh api graphql -f query="query { repository(owner:\"lucaspecina\",name:\"piar-rl\") { issue(number:<N>) { databaseId } } }" --jq '.data.repository.issue.databaseId')
gh api -X DELETE repos/lucaspecina/piar-rl/issues/<OLD_PARENT>/sub_issue \
  -F sub_issue_id=$CHILD_ID

# (4) Crear los nuevos sub-issues y linkearlos al epic <N>
# (loop usando flujo #1 para cada uno)

# (5) Actualizar la tabla "Epics activos" en CLAUDE.md
```

## 7. Listar sub-issues de un epic

```bash
gh api /repos/lucaspecina/piar-rl/issues/<EPIC>/sub_issues \
  --jq '.[] | "#\(.number) [\(.state)] \(.title)"'
```

## 8. Auditar razones de cierre (busca items mal cerrados)

```bash
gh issue list -R lucaspecina/piar-rl --state closed --limit 50 \
  --json number,title,stateReason | \
  python -c "import json,sys; [print(f\"#{i['number']:3} {i['stateReason']:13} {i['title'][:60]}\") for i in json.load(sys.stdin)]"
```

Si ves `not_planned` en items que deberían ser `completed` (o viceversa), reabrir y cerrar con la razón correcta.

## 9. Limpiar el board de items "not planned"

```bash
# Para cada item cerrado con not_planned que aún aparece en Done:
gh api graphql -f query="mutation {
  deleteProjectV2Item(input: {projectId: \"$PROJECT_ID\", itemId: \"$ITEM_ID\"}) { deletedItemId }
}"
# El issue sigue cerrado; solo se remueve del board.
```

## 10. Agregar campo Worktree (cuando llegue la fase de código + sesiones paralelas)

PIAR no usa el campo `Worktree` todavía. Cuando se necesite paralelización:

**Preferir la UI web** (https://github.com/users/lucaspecina/projects/5/settings/fields) para crear el campo y agregar opciones — la UI evita el bug del API que regenera option IDs.

Si hace falta usar API:

```bash
# (1) Crear el field nuevo
gh api graphql -f query='mutation {
  createProjectV2Field(input: {
    projectId: "PVT_kwHOAiGijs4BWrzK",
    dataType: SINGLE_SELECT,
    name: "Worktree",
    singleSelectOptions: [
      {name: "main", color: GRAY, description: ""},
      {name: "none", color: GRAY, description: ""}
    ]
  }) { projectV2Field { ... on ProjectV2SingleSelectField { id options { id name } } } }
}'

# (2) Capturar el field ID y option IDs, agregarlos a reference.md.

# (3) IMPORTANTE: agregar opciones nuevas SIEMPRE por UI, no por API.
# El bug confirmado: updateProjectV2Field con singleSelectOptions REGENERA todos los option IDs,
# invalidando todas las asignaciones previas.
# Si pasa por accidente: re-aplicar Worktree en cada item afectado con los NUEVOS option IDs.
```

Después de crear el campo:
- Actualizar SKILL.md (volver a 3 conceptos: Epic / Worktree / Issue, con la columna Worktree).
- Actualizar CLAUDE.md: tabla "Epics activos" gana columna Worktree.
- Actualizar reference.md con field ID y option IDs.
- Adaptar este archivo: en flujo #1 paso (4), volver a setear Worktree obligatorio.
