# Tracking reference — Project v2 IDs + GraphQL templates

## IDs estables (al 2026-05-04)

```bash
# Repo
REPO="lucaspecina/piar-rl"
PROJECT_NUM=5
PROJECT_OWNER="lucaspecina"

# Project v2
PROJECT_ID="PVT_kwHOAiGijs4BWrzK"

# Status field (default — viene con todo Project v2)
STATUS_FIELD_ID="PVTSSF_lAHOAiGijs4BWrzKzhR97zQ"
  STATUS_TODO="f75ad846"
  STATUS_IN_PROGRESS="47fc9ee4"
  STATUS_DONE="98236657"
```

> **Worktree field**: PIAR todavía no usa `Worktree`. Cuando se cree (ver
> `commands.md` recipe #10), agregar acá: `WORKTREE_FIELD_ID` + option IDs.

Si algún comando falla con "option not found" o similar, refrescar IDs (pueden
haber cambiado si se agregó/quitó opción del field).

## Query de refresh (obtener todos los IDs actuales)

```bash
gh api graphql -f query='query {
  user(login:"lucaspecina") {
    projectV2(number:5) {
      id
      fields(first:30) {
        nodes {
          ... on ProjectV2Field { id name }
          ... on ProjectV2SingleSelectField {
            id
            name
            options { id name color }
          }
        }
      }
    }
  }
}'
```

## Query del board (Status por item)

```bash
gh api graphql -f query='query {
  user(login:"lucaspecina") {
    projectV2(number:5) {
      items(first:100) {
        nodes {
          id
          content {
            ... on Issue { number title state }
          }
          fieldValues(first:10) {
            nodes {
              ... on ProjectV2ItemFieldSingleSelectValue {
                field { ... on ProjectV2SingleSelectField { name } }
                name
              }
            }
          }
        }
      }
    }
  }
}'
```

Para parsear y ver tabla item/status, pipe a Python:
```bash
gh api graphql -f query='...' | python -c "
import json,sys
d=json.load(sys.stdin)
for n in d['data']['user']['projectV2']['items']['nodes']:
    if not n.get('content'): continue
    num = n['content']['number']
    title = n['content']['title'][:60]
    state = n['content']['state']
    fields = {f['field']['name']: f['name'] for f in n['fieldValues']['nodes'] if f.get('field')}
    status = fields.get('Status', 'MISSING')
    print(f'#{num:3} [{state}] Status={status:12} {title}')
"
```

## Sub-issue API

GitHub nativa — no "Part of #N" en body.

```bash
# OJO Git Bash en Windows: el endpoint sin leading slash evita que el shell
# convierta `/repos/...` en una path local. Usar `repos/...` directo.

# Listar sub-issues de un epic
gh api repos/lucaspecina/piar-rl/issues/<EPIC>/sub_issues \
  --jq '.[] | "#\(.number) [\(.state)] \(.title)"'

# Obtener databaseId entero (NO el GraphQL global ID).
# `gh issue view --json id --jq '.databaseId'` NO funciona — `id` da el GraphQL ID.
# Usar GraphQL directo:
CHILD_ID=$(gh api graphql -f query='query { repository(owner:"lucaspecina",name:"piar-rl") { issue(number:<NNN>) { databaseId } } }' --jq '.data.repository.issue.databaseId')

# Linkear sub-issue
gh api -X POST repos/lucaspecina/piar-rl/issues/<EPIC>/sub_issues \
  -F sub_issue_id=$CHILD_ID

# Unlink (para re-parentear)
gh api -X DELETE repos/lucaspecina/piar-rl/issues/<OLD_EPIC>/sub_issue \
  -F sub_issue_id=$CHILD_ID
```

## Auth

`gh` en PATH. Authenticated as `lucaspecina`. Scopes: `gist, project, read:org, repo`.
Si falla auth: `gh auth refresh -s project,repo`.
