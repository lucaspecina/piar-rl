"""
Extrae las specs estructuradas del dataset de WebShop para uso como privileged
context del teacher en PIAR.

Reproduce la lógica de `get_human_goals` (code/agent_system/environments/
env_package/webshop/webshop/web_agent_site/engine/goal.py:22-65) pero standalone:
no requiere la conda env de webshop, no requiere spaCy, no requiere Java.

Entrada: items_shuffle.json + items_human_ins.json (bajados via setup.sh -d all).
Salida: webshop-specs.json con metadata + análisis estadístico + lista de goals.

Uso:
    python tools/extract_webshop_specs.py \
        --items-shuffle /path/to/data/items_shuffle.json \
        --items-human-ins /path/to/data/items_human_ins.json \
        --out experiments/E000/webshop-specs.json

Para WebShop "small" (1000 productos): usar items_shuffle_1000.json + items_ins_v2_1000.json.

Decisión C.5 (research/synthesis/design-decisions.md): la spec estructurada es el
primary privileged context. Este script extrae los campos relevantes y los serializa
en el formato del prompt teacher.

Post-pivot 2026-06 (N.1/N.8, research/synthesis/pi-webshop.md): además de la spec
monolítica (C.3, brazo A1 full-golden), el script descompone cada goal en
componentes fácticos g_i con su wrapper w_i (para el belief b_i del backward) y
serializa el bloque del residual R(t) (forward residual, brazo A4). Ver
decompose_goal_components / serialize_residual_block.

Refs: #17, design-decisions.md C.3 / C.5 / D.9 / N.1 / N.8, pi-webshop.md §2-§5.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PRICE_RANGE = [10.0 * i for i in range(1, 100)]


# Importante: este loader NO usa los items_shuffle.json en sí mismos para los precios.
# En el WebShop real, product_prices se construye en load_products (engine.py).
# Para extracción de specs no necesitamos precios reales — usamos el price_upper
# que viene en cada instruction, que es lo que el teacher ve. Si querés precios
# reales, pasá --items-with-prices y se cargan; sino se omiten.


def load_products_with_human_instructions(items_human_ins_path: Path) -> list[dict]:
    """Carga items_human_ins.json — productos con instrucciones humanas asociadas.

    Estructura esperada (de items_human_ins.json):
    [
        {
            "asin": "B07XXX",
            "category": "Electronics",
            "query": "...",
            "name": "...",
            "product_category": "Electronics > Audio > Headphones",
            "instructions": [
                {
                    "instruction": "I am looking for a ...",
                    "instruction_attributes": ["brand:Sony", "color:black"],
                    "instruction_options": {"size": "medium"}
                },
                ...
            ]
        },
        ...
    ]
    """
    with items_human_ins_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_goals(
    products_with_instructions: list[dict],
    skip_no_attributes: bool = True,
) -> tuple[list[dict], int]:
    """Reproduce get_human_goals (goal.py:22-65).

    Cada producto puede tener N instrucciones humanas (1 a varias). Cada
    instrucción produce un goal. Goals sin atributos se skipean (mismo
    comportamiento que el original).
    """
    goals = []
    skipped = 0

    for item in products_with_instructions:
        asin = item.get("asin")
        if "instructions" not in item:
            continue
        for product in item["instructions"]:
            attributes = product.get("instruction_attributes", [])
            if skip_no_attributes and len(attributes) == 0:
                skipped += 1
                continue

            # En el WebShop original, price_upper se sortea aleatorio del PRICE_RANGE
            # filtrado por precio del producto. Acá usamos None — el extractor no
            # tiene acceso a product_prices, y para nuestro caso lo que importa es
            # qué información estructurada ve el teacher, no el price_upper exacto.
            # El price_upper real lo computará el env al instanciar.
            goal = {
                "asin": asin,
                "category": item.get("category"),
                "query": item.get("query"),
                "name": item.get("name"),
                "product_category": item.get("product_category"),
                "instruction_text": product.get("instruction", "").strip("."),
                "attributes": attributes,
                "price_upper": None,  # placeholder, ver comentario arriba
                "goal_options": product.get("instruction_options", {}),
            }
            goal["weight"] = 1
            goals.append(goal)

    return goals, skipped


def decompose_goal_components(goal: dict) -> list[dict]:
    """Descompone el goal en componentes fácticos g_i según pi-webshop.md §2 (N.1).

    K = 2 + |attributes| + |goal_options| componentes (g_price se omite si no hay
    restricción de precio). Cada componente lleva:
    - type: prod | attr | opt | price (define wrapper y τ por tipo).
    - wrapper: el prefijo textual w_i para computar el belief b_i (§4).
    - value: el valor canónico VERBATIM del dataset (lo que se puntúa).
    - residual_label: la línea que entra al bloque del residual R(t) (§5).

    Invariante 10: todos son hechos del dataset; `asin`/`query` excluidos (§3).
    Wrappers con canonicalización agresiva (N.8 post-review 2026-06-12):
    restringen el formato de respuesta para colapsar paráfrasis.
    """
    components = []

    name = goal.get("name")
    if name:
        components.append({
            "type": "prod",
            "wrapper": ("Based on the interaction so far, the exact product the "
                        "user is looking for is (answer with the exact product "
                        "title):"),
            "value": str(name),
            "residual_label": f"- Target product: {name}",
        })

    for k, attr in enumerate(goal.get("attributes", []) or [], start=1):
        components.append({
            "type": "attr",
            "wrapper": (f"Based on the interaction so far, required attribute "
                        f"#{k} of the target product is (answer with the exact "
                        f"attribute phrase, lowercase):"),
            "value": str(attr),
            "residual_label": f"- Required attribute: {attr}",
        })

    opts = goal.get("goal_options", {}) or {}
    for key in sorted(opts):
        components.append({
            "type": "opt",
            "wrapper": (f"Based on the interaction so far, the required "
                        f"{key} option of the target product is (answer with "
                        f"the exact option value):"),
            "value": str(opts[key]),
            "residual_label": f"- Required {key}: {opts[key]}",
        })

    price = goal.get("price_upper")
    if price is not None and price < 1_000_000:
        components.append({
            "type": "price",
            "wrapper": ("Based on the interaction so far, the maximum acceptable "
                        "price for the target product is (answer with the exact "
                        "amount in dollars):"),
            "value": f"{price:.2f} dollars",
            "residual_label": f"- Maximum price: {price:.2f} dollars",
        })

    return components


def serialize_residual_block(components: list[dict], known_mask: list[bool] | None = None) -> str:
    """Serializa el residual R(t) al prompt del scorer (pi-webshop.md §5).

    known_mask[i] = True si b_i(t−1) ≥ τ (componente ya sabido → fuera del bloque).
    Orden fijo prod → attr → opt → price independiente de qué entra (el layout no
    debe filtrar cuánto sabe el student). Si R(t) = ∅ devuelve "" (los dos prompts
    del forward quedan idénticos → r_fwd = 0 exacto, auto-annealing).
    """
    if known_mask is None:
        known_mask = [False] * len(components)
    residual = [c for c, known in zip(components, known_mask) if not known]
    if not residual:
        return ""
    order = {"prod": 0, "attr": 1, "opt": 2, "price": 3}
    residual.sort(key=lambda c: order.get(c["type"], 9))
    lines = ["Privileged information (for evaluation only — the shopper cannot "
             "see this):"]
    lines.extend(c["residual_label"] for c in residual)
    return "\n".join(lines)


def serialize_spec_for_teacher_prompt(goal: dict, include_price: bool = True) -> str:
    """Serializa la spec estructurada en el bloque que se inyecta al prompt del teacher.

    Decisión C.3: estilo OPSD. Bloque delimitado para que el chat template
    de Qwen2.5-Instruct lo trate como contexto, no como user message.

    Decisión C.5: NO incluye `asin` ni `name` literal (eso es leakage upper bound,
    reservado para ablations explícitas).

    Output ejemplo:
    [GOLDEN SPEC]
    Target product specification (for the privileged teacher only; the student
    does not see this):
    - Attributes: brand:Sony, color:black, feature:noise-canceling
    - Options: size=medium, color=black
    - Price upper bound: $50.0
    - Category: Electronics > Audio > Headphones
    [/GOLDEN SPEC]
    """
    lines = ["[GOLDEN SPEC]"]
    lines.append(
        "Target product specification (for the privileged teacher only; "
        "the student does not see this):"
    )

    attrs = goal.get("attributes", [])
    if attrs:
        lines.append(f"- Attributes: {', '.join(attrs)}")

    opts = goal.get("goal_options", {})
    if opts:
        opts_str = ", ".join(f"{k}={v}" for k, v in opts.items())
        lines.append(f"- Options: {opts_str}")

    if include_price and goal.get("price_upper") is not None:
        lines.append(f"- Price upper bound: ${goal['price_upper']:.2f}")

    cat = goal.get("product_category")
    if cat:
        lines.append(f"- Category: {cat}")

    lines.append("[/GOLDEN SPEC]")
    return "\n".join(lines)


def estimate_token_length(text: str) -> int:
    """Estimación heurística de longitud en tokens. ~4 chars por token para inglés
    estructurado (suficiente para decidir si excedemos max_prompt_length=4096).
    Para precisión real habría que cargar el tokenizer de Qwen2.5 — overkill
    para esta extracción.
    """
    return len(text) // 4


def analyze_goals(goals: list[dict]) -> dict:
    """Análisis estadístico relevante para decidir el template del teacher y
    estimar bloqueantes (token budget, leakage existencial).
    """
    n = len(goals)
    if n == 0:
        return {"n_goals": 0, "warning": "no goals extracted"}

    # Distribución de # de atributos por goal
    n_attrs = [len(g.get("attributes", [])) for g in goals]
    n_opts = [len(g.get("goal_options", {})) for g in goals]

    # Categorías top
    cats = Counter(g.get("product_category", "unknown") for g in goals)

    # ASINs únicos (para subset analysis)
    unique_asins = len({g.get("asin") for g in goals if g.get("asin")})

    # Longitudes del bloque serializado (token estimado)
    serialized_lens = [estimate_token_length(serialize_spec_for_teacher_prompt(g))
                       for g in goals]

    # Goals con specs "ricas" (>= 2 atributos + >= 1 option) vs "pobres" (< 2 atributos)
    rich_specs = sum(1 for g in goals
                     if len(g.get("attributes", [])) >= 2
                     and len(g.get("goal_options", {})) >= 1)
    poor_specs = sum(1 for g in goals if len(g.get("attributes", [])) < 2)

    # Análisis por componente g_i (post-pivot, pi-webshop.md §2):
    # K por goal + longitud en chars del value por tipo (proxy de |g_i| en tokens,
    # relevante para la normalización 1/L del belief).
    k_per_goal = []
    value_len_by_type: dict[str, list[int]] = {}
    for g in goals:
        comps = decompose_goal_components(g)
        k_per_goal.append(len(comps))
        for c in comps:
            value_len_by_type.setdefault(c["type"], []).append(len(c["value"]))
    components_analysis = {
        "K_per_goal": {
            "min": min(k_per_goal),
            "max": max(k_per_goal),
            "median": statistics.median(k_per_goal),
            "mean": round(statistics.mean(k_per_goal), 2),
            "distribution": dict(sorted(Counter(k_per_goal).items())),
        },
        "value_char_length_by_type": {
            t: {
                "n": len(lens),
                "median": statistics.median(lens),
                "p99": sorted(lens)[int(0.99 * len(lens))] if len(lens) > 100 else max(lens),
            }
            for t, lens in sorted(value_len_by_type.items())
        },
        "note": "price ausente en extracción standalone (price_upper se sortea en el env por episodio)",
    }

    return {
        "components_g_i": components_analysis,
        "n_goals": n,
        "n_unique_asins": unique_asins,
        "attributes_per_goal": {
            "min": min(n_attrs),
            "max": max(n_attrs),
            "median": statistics.median(n_attrs),
            "mean": round(statistics.mean(n_attrs), 2),
            "distribution": dict(sorted(Counter(n_attrs).items())),
        },
        "options_per_goal": {
            "min": min(n_opts),
            "max": max(n_opts),
            "median": statistics.median(n_opts),
            "mean": round(statistics.mean(n_opts), 2),
            "distribution": dict(sorted(Counter(n_opts).items())),
        },
        "top_categories": dict(cats.most_common(15)),
        "serialized_token_length_estimate": {
            "min": min(serialized_lens),
            "max": max(serialized_lens),
            "median": statistics.median(serialized_lens),
            "mean": round(statistics.mean(serialized_lens), 2),
            "p90": sorted(serialized_lens)[int(0.9 * n)] if n > 10 else None,
            "p99": sorted(serialized_lens)[int(0.99 * n)] if n > 100 else None,
        },
        "spec_richness": {
            "rich_specs_count": rich_specs,
            "rich_specs_pct": round(rich_specs / n * 100, 1),
            "poor_specs_count": poor_specs,
            "poor_specs_pct": round(poor_specs / n * 100, 1),
            "definition_rich": ">= 2 attributes AND >= 1 option",
            "definition_poor": "< 2 attributes",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--items-human-ins",
        type=Path,
        required=True,
        help="Path a items_human_ins.json (bajado por setup.sh -d all)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path de salida para el JSON con metadata + analysis + goals",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=10,
        help="Cuántos ejemplos serializados incluir en el output (sample random)",
    )
    parser.add_argument(
        "--include-all-goals",
        action="store_true",
        help="Si se pasa, incluye TODOS los goals en el output. Default: solo metadata + analysis + N ejemplos (más liviano).",
    )
    args = parser.parse_args()

    if not args.items_human_ins.exists():
        print(f"[ERROR] No existe {args.items_human_ins}. Bajalo con setup.sh -d all.")
        sys.exit(1)

    print(f"[INFO] Cargando productos con instrucciones de {args.items_human_ins}...")
    products = load_products_with_human_instructions(args.items_human_ins)
    print(f"[INFO] {len(products)} productos cargados.")

    print(f"[INFO] Extrayendo goals (skip si no tienen attributes)...")
    goals, skipped = extract_goals(products, skip_no_attributes=True)
    print(f"[INFO] {len(goals)} goals extraídos, {skipped} skipeados (sin attributes).")

    print(f"[INFO] Computando análisis estadístico...")
    analysis = analyze_goals(goals)

    # Sample de N ejemplos con serialización completa para inspección manual
    import random
    random.seed(42)
    example_indices = random.sample(range(len(goals)), min(args.n_examples, len(goals)))
    examples = [
        {
            "session_idx_in_sample": i,
            "goal": goals[i],
            "serialized_for_teacher_prompt": serialize_spec_for_teacher_prompt(goals[i]),
            "estimated_token_length": estimate_token_length(serialize_spec_for_teacher_prompt(goals[i])),
            "components_g_i": decompose_goal_components(goals[i]),
            "residual_block_full": serialize_residual_block(decompose_goal_components(goals[i])),
        }
        for i in example_indices
    ]

    output = {
        "metadata": {
            "source_file": str(args.items_human_ins),
            "n_products_loaded": len(products),
            "n_goals_extracted": len(goals),
            "n_goals_skipped_no_attributes": skipped,
            "script": "tools/extract_webshop_specs.py",
            "decisions_applied": ["C.3 (template estilo OPSD)", "C.5 (spec estructurada como primary PI)"],
        },
        "analysis": analysis,
        "examples": examples,
    }

    if args.include_all_goals:
        output["goals"] = goals
        print(f"[INFO] Incluyendo los {len(goals)} goals completos en el output.")
    else:
        print(f"[INFO] Solo metadata + analysis + {len(examples)} ejemplos. Pasá --include-all-goals para todo.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[OK] Escrito {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    print()
    print("=== Resumen ===")
    print(f"  Goals totales: {analysis['n_goals']}")
    print(f"  ASINs únicos: {analysis['n_unique_asins']}")
    print(f"  Attributes/goal — mediana: {analysis['attributes_per_goal']['median']}, max: {analysis['attributes_per_goal']['max']}")
    print(f"  Options/goal — mediana: {analysis['options_per_goal']['median']}, max: {analysis['options_per_goal']['max']}")
    print(f"  Token length del bloque serializado — mediana: {analysis['serialized_token_length_estimate']['median']}, p99: {analysis['serialized_token_length_estimate']['p99']}")
    print(f"  Specs ricas (≥2 attrs + ≥1 opt): {analysis['spec_richness']['rich_specs_pct']}%")
    print(f"  Specs pobres (<2 attrs): {analysis['spec_richness']['poor_specs_pct']}%")
    print(f"  Componentes g_i por goal (K) — mediana: {analysis['components_g_i']['K_per_goal']['median']}, distribución: {analysis['components_g_i']['K_per_goal']['distribution']}")


if __name__ == "__main__":
    main()
