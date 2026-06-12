"""
Harness de scoring de la Figura 1 (pre-registro CONGELADO 2026-06-12, #23).

Implementa, sobre rollouts congelados de WebShop, las tres lecturas del
pre-registro (`research/synthesis/figura1-prereg.md`):

    1. r_fwd full-golden  — log-ratio por accion con el bloque PI completo
                            (pivot 4.3 con R(t) = g completa, formato de bloque
                            por lineas de pi-webshop.md §5).
    2. r_bwd              — Sum_i [b_i(t) - b_i(t-1)], belief por componente
                            con wrapper w_i (pivot 4.2, pi-webshop.md §4).
    3. r_fwd residual     — log-ratio con R(t) = {i : b_i(t-1) < tau_tipo},
                            tau = percentil empirico por tipo (default p50).

y computa las confirmatorias P1'/P2'/P3/P5, el control existencial P4
(`--shuffled` + `--baseline-json`) y las descriptivas de prereg §4.3
(P1/P2 between-type, b(0)/b(T) por tipo, cuasi-ortogonalidad, fraccion R=vacio,
share de Sum r_bwd por tipo, N.14 ventana-2 vs historial completo).

CONTRATO DE INPUT (--rollouts, JSONL; lo produce el generador de rollouts en
la VM — una linea por episodio):

    {
      "goal":  { ... },        # goal dict de WebShop EN RUNTIME (goal.py:48-58):
                               #   name (titulo target — disponible en runtime),
                               #   attributes (list[str]),
                               #   goal_options (dict key->value, o list keyless),
                               #   price_upper (float; >= 1e6 = sin restriccion),
                               #   instruction_text (lo que ve el student),
                               #   asin (NECESARIO para la sub-clase P1' de
                               #         clicks de producto; esta en el goal
                               #         de runtime — loggearlo siempre).
      "turns": [               # turno t (1-based en el reporte)
        {"action": "search[...]" | "click[...]",
         "observation": "..."  # respuesta del env A ESA accion (la obs del
                               # ultimo turno = obs final del episodio; el
                               # generador DEBE incluirla — cubre b_i(T)),
         "clicked_asin": "..." # OPCIONAL: ASIN del link clickeado. Solo hace
                               # falta si el texto del click no es el ASIN
                               # (en WebShop text env los links de producto
                               # son el ASIN, asi que normalmente sobra).
        }, ...
      ],
      "score": 0.75            # score continuo final de WebShop
    }

    NOTA: el contrato NO trae la observacion inicial (pagina de busqueda
    previa a la primera accion) ni las listas de admissible actions. Los
    prompts del harness son una reconstruccion reducida del template real
    (`prompts/webshop.py`): header + transcript de pares (accion j, obs j)
    + tail que pide la proxima accion. b_i(0) se computa con historial =
    solo la instruccion (sin obs inicial). Ventana-2 = ultimos 2 pares
    (accion, obs); el pipeline real (env_manager.py:392) muestra ademas la
    obs ANTERIOR a la accion mas vieja de la ventana — desviacion de media
    observacion, documentada. Las comparaciones del prereg son log-ratios y
    deltas bajo el MISMO prompt base, asi que la reconstruccion es
    auto-consistente.

SCORING (riesgos de piar-implementation-points.md §7.5 cubiertos):
    - Tokenizacion frontera wrapper->valor: SIEMPRE se tokeniza prefijo+valor
      juntos y el span del valor se extrae por offsets de chars (§7.5.5);
      nunca se tokeniza el valor aislado. Asserts en runtime: el span es
      contiguo, termina exactamente al final del texto y solo puede colgar
      whitespace del prefijo en su primer token.
    - Chat template: tokenizer.apply_chat_template(add_generation_prompt=True)
      (§7.5.4). Las DOS variantes de inyeccion del wrapper (pi-webshop §4):
      (a) prefill — wrapper como continuacion del assistant despues de un
      user message que pide una accion; (b) user — la pregunta del wrapper
      como user message final (sin tail de accion). Modelos sin chat template
      (smoke con tiny-gpt2) caen a "user + '\\n\\nAssistant:\\n'".
    - Normalizacion: b_i = media 1/|g_i| sobre los tokens del valor (IGPO
      Eq. 3); r_fwd = SUMA sobre el span de la accion (estilo iStar);
      --length-norm cambia a media (ablation pre-registrada).

DECISIONES DE IMPLEMENTACION donde el prereg era ambiguo (revisar antes de
correr en serio; lista completa en tools/README.md):
    - P4: en el run --shuffled la CLASIFICACION (clicks correctos,
      informatividad) se mantiene contra el goal VERDADERO y los scores se
      computan con el goal rotado (derangement ciclico seeded). El efecto
      shuffled es el mismo estadistico con el canal de PI roto. P4 cubre
      {P1', P2', P3} (la familia Bonferroni); P5-shuffled se reporta
      descriptivo (que el belief siga al contexto bajo shuffle ES el
      mecanismo, no leakage).
    - P1' gatea sobre r_fwd FULL-GOLDEN (el residual depende de tau/P5; se
      reporta como secundario). PASS = ambos contrastes (producto y opcion)
      con direccion correcta y p < alpha/3; clase vacia -> N/A y se gatea
      con el contraste restante.
    - Tests direccionales (one-sided greater) — las predicciones son
      direccionales. Bonferroni alpha = 0.05/3 sobre {P1', P2', P3}.
    - tau_tipo = percentil --tau-percentile (default 50) sobre TODOS los
      b_i(t) pooled (t=0..T, todos los episodios) de ese tipo, del mismo run.
    - Full-golden usa el MISMO formato de bloque por lineas que el residual
      (con los K componentes) — aisla el gating como unica diferencia. El
      template OPSD [GOLDEN SPEC] queda para el brazo A1 de training.
    - P5 incluye filas t=0 (label "no aparecido"). El matching es la forma
      canonica VERBATIM case-insensitive (regla congelada); para price
      ("X.XX dollars") casi nunca matchea una obs ("$X.XX") -> el tipo queda
      sin positivos y se excluye del AUC por falta de datos; se loggea un
      conteo descriptivo con el patron "$X.XX" (NO gatea).
    - "Observaciones previas" excluye la instruccion (no es una observacion).
    - r_fwd puntua la accion pelada como mensaje del assistant (el contrato
      no trae <think>; sin scaffold <action>). Mismo formato en numerador y
      denominador -> el prior de formato se cancela en el ratio.

USO (VM, scoring real — ver tools/README.md):

    PYTHONUTF8=1 uv run --with torch --with transformers --no-project \
        python tools/figura1_scoring.py \
        --rollouts experiments/E001/rollouts.jsonl \
        --model Qwen/Qwen2.5-7B-Instruct --device auto \
        --history both --out experiments/E001/figura1_scores.json

    # P4 (control existencial):
    ... --shuffled --baseline-json experiments/E001/figura1_scores.json \
        --out experiments/E001/figura1_scores_shuffled.json

Refs: #23 (prereg), #17 (g_i), pivot-2026-06.md §4/§8, pi-webshop.md §2-§5/§8,
piar-implementation-points.md §7. Reusa decompose_goal_components y
serialize_residual_block de tools/extract_webshop_specs.py (no duplicar).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_webshop_specs import (  # noqa: E402  (path fix arriba)
    decompose_goal_components,
    serialize_residual_block,
)

ALPHA = 0.05
BONFERRONI_FAMILY = 3  # {P1', P2', P3}
AUC_THRESHOLD = 0.8    # P5
RHO_THRESHOLD = 0.3    # P3

# Targets fijos de navegacion del engine de WebShop (engine.py:32-41).
NAV_TARGETS = {
    "back to search", "next >", "< prev",
    "description", "features", "reviews", "attributes",
}
BUY_TARGET = "buy now"
ASIN_RE = re.compile(r"^[a-z0-9]{10}$", re.IGNORECASE)
SEARCH_RE = re.compile(r"^search\[(.*)\]$", re.DOTALL)
CLICK_RE = re.compile(r"^click\[(.*)\]$", re.DOTALL)

HEADER_TEMPLATE = (
    "You are an expert autonomous agent operating in the WebShop "
    "e-commerce environment.\nYour task is to: {instruction}."
)
ACTION_TAIL = (
    "\n\nNow it's your turn to take one action for the current step. "
    "Reply with exactly one admissible action, e.g. search[<query>] or "
    "click[<target>]."
)


# --------------------------------------------------------------------------
# normalize_color del engine vendoreado (regla congelada del prereg §3).
# Se importa por path para no duplicar COLOR_SET (y no tocar code/).
# --------------------------------------------------------------------------

def load_normalize_color() -> Callable[[str], str]:
    """Carga normalize_color desde el engine vendoreado en code/ (sin deps)."""
    p = (SCRIPT_DIR.parent / "code" / "agent_system" / "environments" /
         "env_package" / "webshop" / "webshop" / "web_agent_site" / "engine" /
         "normalize.py")
    if p.exists():
        spec = importlib.util.spec_from_file_location("webshop_normalize", p)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.normalize_color
    print("[WARN] no se encontro code/.../normalize.py — normalize_color cae a "
          "identidad (la sub-clase de clicks de opcion pierde la normalizacion "
          "de colores del prereg §3).")
    return lambda s: s


# --------------------------------------------------------------------------
# Carga de rollouts + clasificacion determinística de acciones (prereg §3)
# --------------------------------------------------------------------------

@dataclass
class Turn:
    action: str
    observation: str
    clicked_asin: str | None = None


@dataclass
class Episode:
    idx: int
    goal_true: dict
    goal_scoring: dict          # == goal_true salvo --shuffled
    turns: list[Turn]
    score: float
    components: list[dict] | None = None   # del goal_scoring


def load_rollouts(path: Path, limit: int = 0) -> list[Episode]:
    """Lee el JSONL de rollouts congelados (contrato en el docstring)."""
    episodes: list[Episode] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            goal = rec["goal"]
            turns = [Turn(action=str(t["action"]).strip(),
                          observation=str(t["observation"]),
                          clicked_asin=t.get("clicked_asin"))
                     for t in rec["turns"]]
            if not turns:
                print(f"[WARN] linea {line_no}: episodio sin turnos — skipeado.")
                continue
            episodes.append(Episode(
                idx=len(episodes), goal_true=goal, goal_scoring=goal,
                turns=turns, score=float(rec["score"]),
            ))
            if limit and len(episodes) >= limit:
                break
    return episodes


def derange_goals(episodes: list[Episode], seed: int) -> None:
    """P4: rota los goals entre episodios (derangement ciclico de una
    permutacion seeded — ningun episodio conserva su goal). Solo toca
    goal_scoring; la clasificacion sigue usando goal_true."""
    n = len(episodes)
    if n < 2:
        raise SystemExit("[ERROR] --shuffled requiere >= 2 episodios.")
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    goals = [ep.goal_true for ep in episodes]
    for k in range(n):
        episodes[order[k]].goal_scoring = goals[order[(k + 1) % n]]
    for ep in episodes:
        assert ep.goal_scoring is not ep.goal_true, "derangement roto"


def classify_action(action: str, goal: dict,
                    normalize_color: Callable[[str], str],
                    clicked_asin: str | None = None) -> dict:
    """Clasificacion determinística del prereg §3 (reglas congeladas).

    Clases: search / click-nav / click-opcion-buy.
    kind:   search / nav / product / option / buy / invalid.
    correct (solo product y option):
      - product: ASIN del link == goal['asin'] (case-insensitive). El ASIN
        sale del texto del click si matchea ^[A-Za-z0-9]{10}$ (en WebShop
        text env los links de producto SON el ASIN); si no, de clicked_asin
        (loggeado por el generador); si no, fallback por nombre (texto del
        click == goal['name'], lowercase/whitespace-normalizado).
      - option: normalize_color(lowercase) del valor in valores de
        goal['goal_options'] normalizados igual.
    """
    a = action.strip()
    if SEARCH_RE.match(a):
        return {"cls": "search", "kind": "search", "correct": None}
    m = CLICK_RE.match(a)
    if not m:
        return {"cls": "invalid", "kind": "invalid", "correct": None}
    target = m.group(1).strip()
    tl = " ".join(target.lower().split())
    goal_asin = str(goal.get("asin") or "").lower()
    if tl == BUY_TARGET:
        return {"cls": "click-opcion-buy", "kind": "buy", "correct": None}
    if tl in NAV_TARGETS:
        return {"cls": "click-nav", "kind": "nav", "correct": None}
    if ASIN_RE.match(target):
        correct = (target.lower() == goal_asin) if goal_asin else None
        return {"cls": "click-nav", "kind": "product", "correct": correct}
    if clicked_asin and goal_asin:
        return {"cls": "click-nav", "kind": "product",
                "correct": str(clicked_asin).lower() == goal_asin}
    name = " ".join(str(goal.get("name") or "").lower().split())
    if name and tl == name:
        return {"cls": "click-nav", "kind": "product", "correct": True}
    # Residual: cualquier otro clickable de una pagina de producto es una opcion.
    opts = goal.get("goal_options") or {}
    values = list(opts.values()) if isinstance(opts, dict) else list(opts)
    norm = lambda s: normalize_color(" ".join(str(s).lower().split()))  # noqa: E731
    correct = norm(target) in {norm(v) for v in values} if values else False
    return {"cls": "click-opcion-buy", "kind": "option", "correct": correct}


def first_appearance_turns(components: list[dict],
                           turns: list[Turn]) -> list[int | None]:
    """Para cada componente, primer turno t (1-based) cuya observacion
    contiene la forma canonica (substring case-insensitive, regla congelada
    del prereg §3); None si nunca aparece."""
    firsts: list[int | None] = []
    obs_lower = [t.observation.lower() for t in turns]
    for comp in components:
        v = comp["value"].lower()
        first = None
        for t, obs in enumerate(obs_lower, start=1):
            if v in obs:
                first = t
                break
        firsts.append(first)
    return firsts


# --------------------------------------------------------------------------
# Prompts (reconstruccion reducida del template de WebShop — ver docstring)
# --------------------------------------------------------------------------

def build_history_user_text(instruction: str, turns: list[Turn],
                            upto_obs: int, mode: str) -> str:
    """Texto del user message con historial hasta la observacion upto_obs
    inclusive (upto_obs=0 -> solo la instruccion). mode: window2 | full."""
    header = HEADER_TEMPLATE.format(instruction=instruction)
    if upto_obs <= 0:
        return header + "\nYou have not taken any action yet."
    start = max(0, upto_obs - 2) if mode == "window2" else 0
    lines = []
    for j in range(start, upto_obs):
        lines.append(f"[Action {j + 1}: '{turns[j].action}' -> "
                     f"Observation {j + 1}: '{turns[j].observation}']")
    return (header +
            f"\nPrior to this step, you have already taken {upto_obs} step(s). "
            f"Below are the most recent {upto_obs - start} action(s) you took "
            "and the resulting observation(s) (most recent last):\n"
            + "\n".join(lines))


# --------------------------------------------------------------------------
# Motor de scoring (transformers; span por offsets, prefijo+valor JUNTOS)
# --------------------------------------------------------------------------

@dataclass
class SpanJob:
    """Un trabajo de scoring: logprobs del segmento `value` (al final o no)
    bajo el texto completo prefix+value. El span se extrae por offsets."""
    prefix: str
    value: str


class ScoringEngine:
    """Carga tokenizer+modelo y puntua spans por lotes.

    Garantias (asserts en runtime, §7.5.5):
      - prefijo+valor se tokenizan JUNTOS; span = tokens cuyo offset pisa
        los chars del valor;
      - el span es contiguo, su ultimo token termina exactamente donde
        termina el valor, y lo unico del prefijo que puede colgar dentro del
        primer token del span es whitespace (fusion BPE del espacio inicial).
    """

    def __init__(self, model_name: str, device: str, dtype: str,
                 batch_size: int, max_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise SystemExit("[ERROR] se requiere fast tokenizer (offsets).")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype == "auto":
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        print(f"[INFO] cargando modelo {model_name} (device={device}, "
              f"dtype={torch_dtype}) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype)
        self.model.to(device)
        self.model.eval()
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._pad_id = (self.tokenizer.pad_token_id
                        if self.tokenizer.pad_token_id is not None
                        else self.tokenizer.eos_token_id)

    def chat_prefix(self, user_text: str) -> str:
        """Prefijo listo para continuar como assistant (apply_chat_template
        con add_generation_prompt=True; fallback plano sin chat template)."""
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                add_generation_prompt=True, tokenize=False)
        return user_text + "\n\nAssistant:\n"

    def prepare(self, job: SpanJob) -> tuple[list[int], int, int]:
        """Tokeniza prefijo+valor juntos y devuelve (ids, span_start, span_end)
        con el span en indices de token (inclusive). Asserts de frontera."""
        full = job.prefix + job.value
        enc = self.tokenizer(full, add_special_tokens=False,
                             return_offsets_mapping=True)
        ids: list[int] = enc["input_ids"]
        offsets: list[tuple[int, int]] = enc["offset_mapping"]
        v0, v1 = len(job.prefix), len(full)
        span = [i for i, (a, b) in enumerate(offsets) if b > v0 and a < v1 and b > a]
        if not span:
            raise AssertionError(f"span vacio para value={job.value!r}")
        if span != list(range(span[0], span[-1] + 1)):
            raise AssertionError(f"span no contiguo para value={job.value!r}")
        if span[0] < 1:
            raise AssertionError("el span no puede arrancar en la posicion 0 "
                                 "(no habria logprob condicional)")
        cov_start = offsets[span[0]][0]
        cov_end = offsets[span[-1]][1]
        hangover = full[cov_start:v0]
        if hangover.strip() != "":
            raise AssertionError(
                f"frontera prefijo->valor sucia: el primer token del span "
                f"arrastra {hangover!r} (no-whitespace) — value={job.value!r}")
        if cov_end != v1:
            raise AssertionError(
                f"el span no cubre el final del valor (cov_end={cov_end}, "
                f"esperado {v1}) — value={job.value!r}")
        if len(ids) > self.max_tokens:
            raise SystemExit(
                f"[ERROR] prompt de {len(ids)} tokens > --max-tokens "
                f"{self.max_tokens}. Subi el limite o revisa el episodio.")
        return ids, span[0], span[-1]

    def score(self, jobs: dict[str, SpanJob]) -> dict[str, dict]:
        """Puntua todos los jobs. Devuelve key -> {sum, mean, n_tokens}."""
        torch = self.torch
        prepared = []
        for key, job in jobs.items():
            ids, s0, s1 = self.prepare(job)
            prepared.append((key, ids, s0, s1))
        prepared.sort(key=lambda x: len(x[1]))
        results: dict[str, dict] = {}
        t_start = time.time()
        n_total = len(prepared)
        for batch_start in range(0, n_total, self.batch_size):
            batch = prepared[batch_start:batch_start + self.batch_size]
            max_len = max(len(ids) for _, ids, _, _ in batch)
            input_ids = torch.full((len(batch), max_len), self._pad_id,
                                   dtype=torch.long)
            attn = torch.zeros((len(batch), max_len), dtype=torch.long)
            for r, (_, ids, _, _) in enumerate(batch):
                input_ids[r, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                attn[r, :len(ids)] = 1
            with torch.inference_mode():
                out = self.model(input_ids=input_ids.to(self.device),
                                 attention_mask=attn.to(self.device))
            logits = out.logits
            for r, (key, ids, s0, s1) in enumerate(batch):
                # logprob del token en posicion p sale de logits[p-1]
                sel = logits[r, s0 - 1:s1, :].float()
                logprobs = torch.log_softmax(sel, dim=-1)
                tgt = torch.tensor(ids[s0:s1 + 1], dtype=torch.long,
                                   device=logprobs.device)
                token_lp = logprobs.gather(1, tgt.unsqueeze(1)).squeeze(1)
                total = float(token_lp.sum().item())
                n_tok = s1 - s0 + 1
                results[key] = {"sum": total, "mean": total / n_tok,
                                "n_tokens": n_tok}
            del logits, out
            done = min(batch_start + self.batch_size, n_total)
            if done % (self.batch_size * 10) < self.batch_size or done == n_total:
                rate = done / max(time.time() - t_start, 1e-9)
                print(f"[scoring] {done}/{n_total} jobs ({rate:.1f} jobs/s)")
        return results


# --------------------------------------------------------------------------
# Estadistica (scipy si esta; fallback manual sin dependencia dura)
# --------------------------------------------------------------------------

def rankdata(values: list[float]) -> list[float]:
    """Rangos promedio (ties -> rango medio)."""
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_whitney_greater(x: list[float], y: list[float]) -> dict | None:
    """Mann-Whitney U one-sided (H1: x > y). scipy si esta disponible;
    si no, aproximacion normal con correccion por ties y continuidad."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return None
    try:
        from scipy.stats import mannwhitneyu
        res = mannwhitneyu(x, y, alternative="greater")
        return {"U": float(res.statistic), "p": float(res.pvalue),
                "n_x": nx, "n_y": ny, "method": "scipy"}
    except ImportError:
        pass
    allv = list(x) + list(y)
    ranks = rankdata(allv)
    rx = sum(ranks[:nx])
    u = rx - nx * (nx + 1) / 2
    mu = nx * ny / 2
    n = nx + ny
    ties = sum(c ** 3 - c for c in Counter(allv).values())
    var = nx * ny / 12.0 * ((n + 1) - ties / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return {"U": u, "p": 1.0, "n_x": nx, "n_y": ny,
                "method": "manual-degenerado"}
    z = (u - mu - 0.5) / math.sqrt(var)
    return {"U": u, "p": 1.0 - _phi(z), "n_x": nx, "n_y": ny,
            "method": "manual-normal-approx"}


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def _residualize(a: list[float], b: list[float]) -> list[float]:
    """Residuos de regresion lineal simple a ~ b (con intercepto)."""
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sbb = sum((v - mb) ** 2 for v in b)
    if sbb <= 0:
        return [v - ma for v in a]
    slope = sum((u - ma) * (v - mb) for u, v in zip(a, b)) / sbb
    return [u - (ma + slope * (v - mb)) for u, v in zip(a, b)]


def spearman_partial_greater(x: list[float], y: list[float],
                             z: list[float]) -> dict:
    """Spearman parcial de x vs y controlando z (rangos + regresion +
    Pearson sobre residuos). p one-sided (H1: rho > 0)."""
    n = len(x)
    if n < 5:
        return {"rho": None, "p": None, "n": n, "note": "n<5: sin potencia"}
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex, ey = _residualize(rx, rz), _residualize(ry, rz)
    rho = _pearson(ex, ey)
    if rho is None:
        return {"rho": None, "p": None, "n": n, "note": "varianza nula"}
    df = n - 3  # n - 2 - k, k=1 control
    if abs(rho) >= 1.0:
        return {"rho": rho, "p": 0.0 if rho > 0 else 1.0, "n": n}
    t_stat = rho * math.sqrt(df / (1.0 - rho ** 2))
    try:
        from scipy.stats import t as t_dist
        p = float(t_dist.sf(t_stat, df))
        method = "scipy-t"
    except ImportError:
        zf = math.atanh(max(min(rho, 1 - 1e-12), -1 + 1e-12)) * math.sqrt(max(n - 4, 1))
        p = 1.0 - _phi(zf)
        method = "manual-fisher-z"
    return {"rho": rho, "p": p, "n": n, "method": method}


def roc_auc(pos: list[float], neg: list[float]) -> float | None:
    """AUC por rangos (equivale a U/(n1*n0))."""
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return None
    ranks = rankdata(list(neg) + list(pos))
    r_pos = sum(ranks[n0:])
    return (r_pos - n1 * (n1 + 1) / 2) / (n0 * n1)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def percentile(values: list[float], pct: float) -> float:
    s = sorted(values)
    if not s:
        raise ValueError("percentile de lista vacia")
    k = (len(s) - 1) * pct / 100.0
    f = math.floor(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


# --------------------------------------------------------------------------
# r_fwd residual — auto-annealing exacto
# --------------------------------------------------------------------------

def r_fwd_residual_value(block: str, score_with: float | None,
                         score_without: float | None) -> float:
    """Si R(t)=vacio el bloque es '' y los dos prompts serian identicos:
    r_fwd = 0.0 EXACTO sin forward pass (pivot §4.3, auto-annealing)."""
    if block == "":
        return 0.0
    assert score_with is not None and score_without is not None
    return score_with - score_without


# --------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------

def belief_job(engine: ScoringEngine, instruction: str, turns: list[Turn],
               upto_obs: int, mode: str, comp: dict, variant: str) -> SpanJob:
    """Job de belief b_i(t) para la variante de wrapper dada (pi-webshop §4)."""
    hist = build_history_user_text(instruction, turns, upto_obs, mode)
    if variant == "prefill":
        # (a) el user message pide una accion (contexto real del pipeline);
        # el wrapper entra como prefill del assistant.
        prefix = engine.chat_prefix(hist + ACTION_TAIL) + comp["wrapper"]
        return SpanJob(prefix=prefix, value=" " + comp["value"])
    if variant == "user":
        # (b) la pregunta del wrapper reemplaza el pedido de accion.
        prefix = engine.chat_prefix(hist + "\n\n" + comp["wrapper"])
        return SpanJob(prefix=prefix, value=comp["value"])
    raise ValueError(f"variante de wrapper desconocida: {variant}")


def forward_job(engine: ScoringEngine, instruction: str, turns: list[Turn],
                t: int, mode: str, action: str, block: str) -> SpanJob:
    """Job del forward para la accion a_t (t 1-based): historial hasta la obs
    t-1, bloque PI (o '') al INICIO del user message (pi-webshop §5)."""
    base = build_history_user_text(instruction, turns, t - 1, mode) + ACTION_TAIL
    user = (block + "\n\n" + base) if block else base
    return SpanJob(prefix=engine.chat_prefix(user), value=action)


def run_pipeline(episodes: list[Episode], engine: ScoringEngine,
                 args: argparse.Namespace,
                 normalize_color: Callable[[str], str]) -> dict:
    """Computa beliefs, tau, gating, r_bwd y los dos r_fwd para todos los
    episodios. Devuelve la estructura de resultados completa."""
    primary_variant: str = args.wrapper_variant
    secondary_variant = "user" if primary_variant == "prefill" else "prefill"
    primary_mode = "full" if args.history == "full" else "window2"
    belief_sets: list[tuple[str, str]] = [(primary_variant, primary_mode)]
    if args.history == "both":
        belief_sets.append((primary_variant, "full"))
    if not args.skip_secondary_wrapper:
        belief_sets.append((secondary_variant, primary_mode))
    fwd_reduce = "mean" if args.length_norm else "sum"

    # -- componentes por episodio (del goal de scoring) --
    skipped = 0
    for ep in episodes:
        ep.components = decompose_goal_components(ep.goal_scoring)
        if not ep.components:
            skipped += 1
    episodes = [ep for ep in episodes if ep.components]
    if skipped:
        print(f"[WARN] {skipped} episodios sin componentes g_i — skipeados.")
    if not episodes:
        raise SystemExit("[ERROR] no quedaron episodios con componentes.")

    # -- fase 1: jobs de beliefs + forward none/full --
    jobs: dict[str, SpanJob] = {}
    for ep in episodes:
        instr = str(ep.goal_true.get("instruction_text") or "")
        T = len(ep.turns)
        for variant, mode in belief_sets:
            for ci, comp in enumerate(ep.components):
                for t in range(0, T + 1):
                    key = f"b|{ep.idx}|{ci}|{t}|{variant}|{mode}"
                    jobs[key] = belief_job(engine, instr, ep.turns, t, mode,
                                           comp, variant)
        block_full = serialize_residual_block(ep.components,
                                              [False] * len(ep.components))
        for t in range(1, T + 1):
            action = ep.turns[t - 1].action
            jobs[f"f|{ep.idx}|{t}|none"] = forward_job(
                engine, instr, ep.turns, t, primary_mode, action, "")
            jobs[f"f|{ep.idx}|{t}|full"] = forward_job(
                engine, instr, ep.turns, t, primary_mode, action, block_full)
    print(f"[INFO] fase 1: {len(jobs)} jobs (beliefs x{len(belief_sets)} sets "
          f"+ forwards none/full).")
    scores = engine.score(jobs)

    # -- beliefs por episodio --
    beliefs: dict[tuple[int, str, str], list[list[float]]] = {}
    for ep in episodes:
        T = len(ep.turns)
        for variant, mode in belief_sets:
            mat = [[scores[f"b|{ep.idx}|{ci}|{t}|{variant}|{mode}"]["mean"]
                    for t in range(0, T + 1)]
                   for ci in range(len(ep.components))]
            beliefs[(ep.idx, variant, mode)] = mat

    # -- tau por tipo (percentil sobre b pooled del set primario) --
    pooled_by_type: dict[str, list[float]] = {}
    for ep in episodes:
        mat = beliefs[(ep.idx, primary_variant, primary_mode)]
        for ci, comp in enumerate(ep.components):
            pooled_by_type.setdefault(comp["type"], []).extend(mat[ci])
    tau = {ctype: percentile(vals, args.tau_percentile)
           for ctype, vals in pooled_by_type.items()}
    print(f"[INFO] tau (p{args.tau_percentile} por tipo): "
          + ", ".join(f"{k}={v:.4f}" for k, v in sorted(tau.items())))

    # -- fase 2: gating + jobs residuales --
    residual_jobs: dict[str, SpanJob] = {}
    residual_blocks: dict[tuple[int, int], str] = {}
    known_masks: dict[tuple[int, int], list[bool]] = {}
    for ep in episodes:
        instr = str(ep.goal_true.get("instruction_text") or "")
        mat = beliefs[(ep.idx, primary_variant, primary_mode)]
        for t in range(1, len(ep.turns) + 1):
            known = [mat[ci][t - 1] >= tau[comp["type"]]
                     for ci, comp in enumerate(ep.components)]
            known_masks[(ep.idx, t)] = known
            block = serialize_residual_block(ep.components, known)
            residual_blocks[(ep.idx, t)] = block
            if block == "":
                continue  # R=vacio -> r_fwd=0 exacto, sin forward
            if not any(known):
                continue  # R = g completa -> mismo bloque que 'full', reuso
            residual_jobs[f"f|{ep.idx}|{t}|residual"] = forward_job(
                engine, instr, ep.turns, t, primary_mode,
                ep.turns[t - 1].action, block)
    print(f"[INFO] fase 2: {len(residual_jobs)} forwards residuales "
          f"(R no-vacio y != full).")
    if residual_jobs:
        scores.update(engine.score(residual_jobs))

    # -- resultados por turno --
    results: list[dict] = []
    for ep in episodes:
        T = len(ep.turns)
        mat = beliefs[(ep.idx, primary_variant, primary_mode)]
        firsts_scoring = first_appearance_turns(ep.components, ep.turns)
        comps_true = (ep.components if ep.goal_scoring is ep.goal_true
                      else decompose_goal_components(ep.goal_true))
        firsts_true = (firsts_scoring if ep.goal_scoring is ep.goal_true
                       else first_appearance_turns(comps_true, ep.turns))
        turn_records = []
        for t in range(1, T + 1):
            turn = ep.turns[t - 1]
            cls_true = classify_action(turn.action, ep.goal_true,
                                       normalize_color, turn.clicked_asin)
            informative = any(f == t for f in firsts_true)
            s_none = scores[f"f|{ep.idx}|{t}|none"][fwd_reduce]
            s_full = scores[f"f|{ep.idx}|{t}|full"][fwd_reduce]
            block = residual_blocks[(ep.idx, t)]
            if block == "":
                r_res = r_fwd_residual_value("", None, None)
            elif not any(known_masks[(ep.idx, t)]):
                r_res = s_full - s_none  # R = g completa
            else:
                r_res = r_fwd_residual_value(
                    block, scores[f"f|{ep.idx}|{t}|residual"][fwd_reduce],
                    s_none)
            rec = {
                "turn": t,
                "action": turn.action,
                "cls": cls_true["cls"], "kind": cls_true["kind"],
                "correct": cls_true["correct"],
                "informative": informative,
                "r_fwd_full": s_full - s_none,
                "r_fwd_residual": r_res,
                "r_bwd": sum(mat[ci][t] - mat[ci][t - 1]
                             for ci in range(len(ep.components))),
                "residual_size": sum(1 for k in known_masks[(ep.idx, t)]
                                     if not k),
                "n_action_tokens": scores[f"f|{ep.idx}|{t}|none"]["n_tokens"],
            }
            if ep.goal_scoring is not ep.goal_true:
                cls_sh = classify_action(turn.action, ep.goal_scoring,
                                         normalize_color, turn.clicked_asin)
                rec["cls_vs_scoring_goal"] = {
                    "kind": cls_sh["kind"], "correct": cls_sh["correct"],
                    "informative": any(f == t for f in firsts_scoring),
                }
            turn_records.append(rec)
        results.append({
            "episode": ep.idx,
            "score": ep.score,
            "n_turns": T,
            "K": len(ep.components),
            "components": [{"type": c["type"], "value": c["value"]}
                           for c in ep.components],
            "first_appearance_turn": firsts_scoring,
            "turns": turn_records,
            "beliefs": {f"{v}|{m}": beliefs[(ep.idx, v, m)]
                        for v, m in belief_sets},
        })
    return {
        "episodes_out": results,
        "episodes_in": episodes,
        "beliefs": beliefs,
        "tau": tau,
        "belief_sets": belief_sets,
        "primary_variant": primary_variant,
        "primary_mode": primary_mode,
        "secondary_variant": (None if args.skip_secondary_wrapper
                              else secondary_variant),
    }


# --------------------------------------------------------------------------
# Analisis: confirmatorias + descriptivas
# --------------------------------------------------------------------------

def _mw_block(name: str, hi: list[float], lo: list[float],
              alpha: float) -> dict:
    """Contraste direccional mediana(hi) > mediana(lo) + Mann-Whitney."""
    med_hi, med_lo = median(hi), median(lo)
    mw = mann_whitney_greater(hi, lo)
    effect = (med_hi - med_lo) if (med_hi is not None and med_lo is not None) else None
    passed: bool | None
    if mw is None or effect is None:
        passed = None
    else:
        passed = bool(effect > 0 and mw["p"] < alpha)
    return {"contrast": name, "median_hi": med_hi, "median_lo": med_lo,
            "n_hi": len(hi), "n_lo": len(lo), "effect_median_diff": effect,
            "mw": mw, "alpha": alpha, "pass": passed}


def analyze(pipe: dict, args: argparse.Namespace) -> dict:
    """Computa P1'/P2'/P3/P5 + descriptivas del prereg §4.3."""
    eps = pipe["episodes_out"]
    alpha_corr = ALPHA / BONFERRONI_FAMILY
    flat = [(e, t) for e in eps for t in e["turns"]]

    # ---- P1' (gate sobre r_fwd full-golden; residual como secundario) ----
    def p1_contrasts(score_key: str) -> dict:
        prod_c = [t[score_key] for _, t in flat
                  if t["kind"] == "product" and t["correct"] is True]
        prod_i = [t[score_key] for _, t in flat
                  if t["kind"] == "product" and t["correct"] is False]
        opt_c = [t[score_key] for _, t in flat
                 if t["kind"] == "option" and t["correct"] is True]
        opt_i = [t[score_key] for _, t in flat
                 if t["kind"] == "option" and t["correct"] is False]
        prod = _mw_block("click producto target > otro producto",
                         prod_c, prod_i, alpha_corr)
        opt = _mw_block("click opcion in goal_options > opcion fuera",
                        opt_c, opt_i, alpha_corr)
        sub = [b["pass"] for b in (prod, opt) if b["pass"] is not None]
        overall = (None if not sub else all(sub))
        return {"product": prod, "option": opt, "pass": overall}

    p1 = p1_contrasts("r_fwd_full")
    p1_residual = p1_contrasts("r_fwd_residual")

    # ---- P2' (r_bwd informativa > no informativa) ----
    inf = [t["r_bwd"] for _, t in flat if t["informative"]]
    noninf = [t["r_bwd"] for _, t in flat if not t["informative"]]
    p2 = _mw_block("r_bwd informativa > no informativa", inf, noninf,
                   alpha_corr)

    # ---- P3 (Spearman parcial Sum r_bwd vs score | largo) ----
    sums = [sum(t["r_bwd"] for t in e["turns"]) for e in eps]
    scores_f = [e["score"] for e in eps]
    lengths = [float(e["n_turns"]) for e in eps]
    p3_main = spearman_partial_greater(sums, scores_f, lengths)
    multi = [e for e in eps if e["n_turns"] >= 2]
    p3_rob = spearman_partial_greater(
        [sum(t["r_bwd"] for t in e["turns"][:-1]) for e in multi],
        [e["score"] for e in multi],
        [float(e["n_turns"]) for e in multi])
    def _p3_ok(b: dict) -> bool | None:
        if b.get("rho") is None or b.get("p") is None:
            return None
        return bool(b["rho"] > RHO_THRESHOLD and b["p"] < alpha_corr)
    p3_pass_main = _p3_ok(p3_main)
    rob_ok = (None if p3_rob.get("rho") is None or p3_rob.get("p") is None
              else bool(p3_rob["rho"] > 0 and p3_rob["p"] < alpha_corr))
    p3 = {"main": p3_main, "robustness_sin_ultimo_turno": p3_rob,
          "rho_threshold": RHO_THRESHOLD, "alpha": alpha_corr,
          "pass": (None if p3_pass_main is None or rob_ok is None
                   else bool(p3_pass_main and rob_ok))}

    # ---- P5 (AUC sabido vs no-sabido, por tipo y por variante) ----
    variants = [pipe["primary_variant"]]
    if pipe["secondary_variant"]:
        variants.append(pipe["secondary_variant"])
    mode = pipe["primary_mode"]
    p5_variants: dict[str, dict] = {}
    for variant in variants:
        by_type: dict[str, dict[str, list[float]]] = {}
        for e in eps:
            mat = e["beliefs"][f"{variant}|{mode}"]
            firsts = e["first_appearance_turn"]
            for ci, comp in enumerate(e["components"]):
                bucket = by_type.setdefault(comp["type"],
                                            {"pos": [], "neg": []})
                for t in range(0, e["n_turns"] + 1):
                    appeared = firsts[ci] is not None and firsts[ci] <= t
                    bucket["pos" if appeared else "neg"].append(mat[ci][t])
        aucs = {}
        for ctype, b in sorted(by_type.items()):
            auc = roc_auc(b["pos"], b["neg"])
            aucs[ctype] = {"auc": auc, "n_pos": len(b["pos"]),
                           "n_neg": len(b["neg"])}
        with_data = [v["auc"] for v in aucs.values() if v["auc"] is not None]
        v_pass = (None if not with_data
                  else all(a > AUC_THRESHOLD for a in with_data))
        p5_variants[variant] = {"auc_by_type": aucs, "pass": v_pass}
    candidates = [(v, d) for v, d in p5_variants.items() if d["pass"]]
    selected = None
    if candidates:
        selected = max(candidates, key=lambda vd: min(
            a["auc"] for a in vd[1]["auc_by_type"].values()
            if a["auc"] is not None))[0]
    p5 = {"variants": p5_variants, "selected_variant": selected,
          "auc_threshold": AUC_THRESHOLD,
          "pass": (None if all(d["pass"] is None
                               for d in p5_variants.values())
                   else any(bool(d["pass"]) for d in p5_variants.values()))}

    # ---- Descriptivas ----
    desc: dict[str, Any] = {}
    # P1/P2 between-type originales (pivot §8) — se reportan, no gatean.
    by_cls: dict[str, dict[str, list[float]]] = {}
    for _, t in flat:
        d = by_cls.setdefault(t["cls"], {"r_fwd_full": [], "r_bwd": [],
                                         "r_fwd_residual": []})
        for k in d:
            d[k].append(t[k])
    desc["between_type_P1_P2"] = {
        cls: {k: {"median": median(v), "n": len(v)} for k, v in d.items()}
        for cls, d in sorted(by_cls.items())}
    # b(0) y b(T) por tipo (variante primaria).
    b0_bt: dict[str, dict[str, list[float]]] = {}
    for e in eps:
        mat = e["beliefs"][f"{pipe['primary_variant']}|{mode}"]
        for ci, comp in enumerate(e["components"]):
            d = b0_bt.setdefault(comp["type"], {"b0": [], "bT": []})
            d["b0"].append(mat[ci][0])
            d["bT"].append(mat[ci][e["n_turns"]])
    desc["b0_bT_por_tipo"] = {
        ctype: {k: {"median": median(v),
                    "p25": percentile(v, 25) if v else None,
                    "p75": percentile(v, 75) if v else None}
                for k, v in d.items()}
        for ctype, d in sorted(b0_bt.items())}
    # Cuasi-ortogonalidad: Pearson de delta-b entre pares de componentes.
    pair_corrs: list[float] = []
    for e in eps:
        mat = e["beliefs"][f"{pipe['primary_variant']}|{mode}"]
        T = e["n_turns"]
        if T < 3 or e["K"] < 2:
            continue
        deltas = [[mat[ci][t] - mat[ci][t - 1] for t in range(1, T + 1)]
                  for ci in range(e["K"])]
        for i in range(e["K"]):
            for j in range(i + 1, e["K"]):
                r = _pearson(deltas[i], deltas[j])
                if r is not None:
                    pair_corrs.append(abs(r))
    desc["cuasi_ortogonalidad_abs_pearson_delta_b"] = {
        "median": median(pair_corrs), "n_pairs": len(pair_corrs)}
    # Fraccion de turnos con R=vacio; y en el ultimo turno de episodios de
    # score alto (>= mediana de scores).
    all_turn_flags = [t["residual_size"] == 0 for _, t in flat]
    med_score = median([e["score"] for e in eps])
    hi_eps = [e for e in eps if med_score is not None and e["score"] >= med_score]
    desc["fraccion_R_vacio"] = {
        "todos_los_turnos": (sum(all_turn_flags) / len(all_turn_flags)
                             if all_turn_flags else None),
        "ultimo_turno_episodios_score_alto": (
            sum(1 for e in hi_eps if e["turns"][-1]["residual_size"] == 0)
            / len(hi_eps) if hi_eps else None),
        "def_score_alto": "score >= mediana de scores del run",
    }
    # Share de Sum r_bwd por tipo (prediccion derivada pi-webshop §8).
    contrib: dict[str, float] = {}
    for e in eps:
        mat = e["beliefs"][f"{pipe['primary_variant']}|{mode}"]
        for ci, comp in enumerate(e["components"]):
            contrib[comp["type"]] = contrib.get(comp["type"], 0.0) + (
                mat[ci][e["n_turns"]] - mat[ci][0])
    total = sum(abs(v) for v in contrib.values()) or 1.0
    desc["share_sum_r_bwd_por_tipo"] = {
        ctype: {"delta_total": v, "share_abs": abs(v) / total}
        for ctype, v in sorted(contrib.items())}
    # Price alt-match (descriptivo, NO gatea): "$X.XX" vs forma canonica.
    alt_hits = 0
    for ep in pipe["episodes_in"]:
        for comp in ep.components or []:
            if comp["type"] != "price":
                continue
            amount = comp["value"].replace(" dollars", "")
            alt = f"${amount}"
            canon_hit = any(comp["value"].lower() in t.observation.lower()
                            for t in ep.turns)
            if not canon_hit and any(alt in t.observation for t in ep.turns):
                alt_hits += 1
    desc["price_alt_match_no_canonico"] = {
        "episodios_con_$X.XX_sin_match_canonico": alt_hits,
        "nota": "la regla congelada es verbatim canonico; esto solo "
                "cuantifica cuanto se pierde por el formato del precio"}
    # N.14: ventana-2 vs full (si --history both).
    if args.history == "both":
        n14 = {"caidas_window2": 0, "explicadas_por_ventana": 0,
               "caidas_reales": 0, "caidas_materiales_window2": 0,
               "materiales_explicadas_por_ventana": 0}
        for e in eps:
            m_w = e["beliefs"][f"{pipe['primary_variant']}|window2"]
            m_f = e["beliefs"][f"{pipe['primary_variant']}|full"]
            for ci in range(e["K"]):
                for t in range(1, e["n_turns"] + 1):
                    dw = m_w[ci][t] - m_w[ci][t - 1]
                    dfull = m_f[ci][t] - m_f[ci][t - 1]
                    if dw < 0:
                        n14["caidas_window2"] += 1
                        if dfull >= 0:
                            n14["explicadas_por_ventana"] += 1
                        else:
                            n14["caidas_reales"] += 1
                    if dw < -0.1:
                        n14["caidas_materiales_window2"] += 1
                        if dfull >= 0:
                            n14["materiales_explicadas_por_ventana"] += 1
        n14["nota"] = ("caida = delta b_i < 0 en window2; explicada = el "
                       "delta full-history en el mismo punto es >= 0. "
                       "'material' usa umbral -0.1 (eleccion descriptiva).")
        desc["N14_ventana_vs_full"] = n14

    return {"P1p": p1, "P1p_residual_secundario": p1_residual, "P2p": p2,
            "P3": p3, "P5": p5, "alpha_corr": alpha_corr,
            "descriptivas": desc}


def p4_compare(current: dict, baseline: dict) -> dict:
    """P4: efecto shuffled < 1/3 del real Y no significativo (alpha=0.05),
    para cada confirmatoria de la familia {P1', P2', P3}."""
    out: dict[str, Any] = {}
    checks: list[bool] = []

    def cmp_mw(name: str, real: dict, sh: dict) -> None:
        e_r, e_s = real.get("effect_median_diff"), sh.get("effect_median_diff")
        p_s = sh["mw"]["p"] if sh.get("mw") else None
        if e_r is None or e_s is None or p_s is None or e_r == 0:
            out[name] = {"ratio": None, "p_shuffled": p_s,
                         "pass": None, "note": "datos insuficientes"}
            return
        ratio = abs(e_s) / abs(e_r)
        ok = bool(ratio < 1.0 / 3.0 and p_s > ALPHA)
        out[name] = {"effect_real": e_r, "effect_shuffled": e_s,
                     "ratio": ratio, "p_shuffled": p_s, "pass": ok}
        checks.append(ok)

    cmp_mw("P1p_product", baseline["predictions"]["P1p"]["product"],
           current["P1p"]["product"])
    cmp_mw("P1p_option", baseline["predictions"]["P1p"]["option"],
           current["P1p"]["option"])
    cmp_mw("P2p", baseline["predictions"]["P2p"], current["P2p"])
    rho_r = baseline["predictions"]["P3"]["main"].get("rho")
    rho_s = current["P3"]["main"].get("rho")
    p_s = current["P3"]["main"].get("p")
    if rho_r in (None, 0) or rho_s is None or p_s is None:
        out["P3"] = {"ratio": None, "pass": None, "note": "datos insuficientes"}
    else:
        ratio = abs(rho_s) / abs(rho_r)
        ok = bool(ratio < 1.0 / 3.0 and p_s > ALPHA)
        out["P3"] = {"rho_real": rho_r, "rho_shuffled": rho_s,
                     "ratio": ratio, "p_shuffled": p_s, "pass": ok}
        checks.append(ok)
    out["pass"] = all(checks) if checks else None
    out["criterio"] = ("|efecto shuffled| < 1/3 |efecto real| Y p_shuffled > "
                       "0.05, para {P1'prod, P1'opt, P2', P3}")
    return out


# --------------------------------------------------------------------------
# Resumen imprimible
# --------------------------------------------------------------------------

def _fmt(x: float | None, nd: int = 4) -> str:
    return "NA" if x is None else f"{x:.{nd}f}"


def _verdict(p: bool | None) -> str:
    return "PASS" if p else ("FAIL" if p is False else "N/A")


def print_summary(pred: dict, tau: dict, n_eps: int, shuffled: bool,
                  p4: dict | None) -> None:
    tag = " [SHUFFLED — control P4]" if shuffled else ""
    print(f"\n================ FIGURA 1 — resumen{tag} ================")
    print(f"episodios={n_eps} | alpha Bonferroni={pred['alpha_corr']:.4f} | "
          "tau: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(tau.items())))
    p1 = pred["P1p"]
    print("=== P1' (forward within-type; gate sobre r_fwd full-golden) ===")
    for fam in ("product", "option"):
        b = p1[fam]
        p_str = _fmt(b["mw"]["p"]) if b["mw"] else "NA"
        print(f"  [{fam}] mediana_correct={_fmt(b['median_hi'])} "
              f"mediana_incorrect={_fmt(b['median_lo'])} "
              f"(n={b['n_hi']}/{b['n_lo']}) p={p_str} -> {_verdict(b['pass'])}")
    print(f"  P1' overall: {_verdict(p1['pass'])}")
    p2 = pred["P2p"]
    p_str = _fmt(p2["mw"]["p"]) if p2["mw"] else "NA"
    print("=== P2' (backward premia revelar informacion) ===")
    print(f"  mediana_informativa={_fmt(p2['median_hi'])} "
          f"mediana_no_informativa={_fmt(p2['median_lo'])} "
          f"(n={p2['n_hi']}/{p2['n_lo']}) p={p_str} -> {_verdict(p2['pass'])}")
    p3 = pred["P3"]
    print("=== P3 (Spearman parcial Sum r_bwd vs score | largo) ===")
    print(f"  rho={_fmt(p3['main'].get('rho'))} p={_fmt(p3['main'].get('p'))} "
          f"(umbral rho>{RHO_THRESHOLD}) | robustez sin ultimo turno: "
          f"rho={_fmt(p3['robustness_sin_ultimo_turno'].get('rho'))} "
          f"p={_fmt(p3['robustness_sin_ultimo_turno'].get('p'))} "
          f"-> {_verdict(p3['pass'])}")
    p5 = pred["P5"]
    print("=== P5 (AUC belief sabido vs no-sabido, por tipo y variante) ===")
    for variant, d in p5["variants"].items():
        aucs = " ".join(f"{t}={_fmt(v['auc'], 3)}(n+{v['n_pos']})"
                        for t, v in d["auc_by_type"].items())
        print(f"  [{variant}] {aucs} -> {_verdict(d['pass'])}")
    print(f"  P5 overall: {_verdict(p5['pass'])} "
          f"(variante seleccionada: {p5['selected_variant']})")
    if p4 is not None:
        print("=== P4 (control existencial — shuffled vs baseline) ===")
        for fam in ("P1p_product", "P1p_option", "P2p", "P3"):
            b = p4[fam]
            print(f"  [{fam}] ratio={_fmt(b.get('ratio'), 3)} "
                  f"p_shuffled={_fmt(b.get('p_shuffled'))} "
                  f"-> {_verdict(b['pass'])}")
        print(f"  P4 overall: {_verdict(p4['pass'])} "
              "(PASS = el efecto colapsa como debe)")
    print("=========================================================\n")


# --------------------------------------------------------------------------
# Self-test (sin forward de modelo: tokenizer + logica pura)
# --------------------------------------------------------------------------

def run_self_test(args: argparse.Namespace) -> None:
    """Asserts de la mecanica critica sin pesar el modelo: spans en frontera,
    clasificacion, gating exacto-0, derangement, stats."""
    print("[self-test] 1/6 estadistica ...")
    assert roc_auc([2.0, 3.0], [0.0, 1.0]) == 1.0
    assert roc_auc([0.0, 1.0], [2.0, 3.0]) == 0.0
    assert abs(roc_auc([1.0, 3.0], [2.0, 0.0]) - 0.75) < 1e-9
    mw = mann_whitney_greater([5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0])
    assert mw is not None and mw["p"] < 0.05, mw
    sp = spearman_partial_greater(
        [1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 2, 2, 3, 3, 4, 4])
    assert sp["rho"] is not None and sp["rho"] > 0.9, sp
    assert rankdata([10.0, 20.0, 20.0, 30.0]) == [1.0, 2.5, 2.5, 4.0]

    print("[self-test] 2/6 clasificacion de acciones (prereg §3) ...")
    nc = load_normalize_color()
    goal = {"asin": "B078GWRC1J", "name": "Bright Citrus Deodorant",
            "goal_options": {"size": "3 ounce", "scent": "bright citrus"}}
    c = classify_action("search[deodorant]", goal, nc)
    assert (c["cls"], c["kind"]) == ("search", "search")
    c = classify_action("click[Buy Now]", goal, nc)
    assert (c["cls"], c["kind"]) == ("click-opcion-buy", "buy")
    c = classify_action("click[Next >]", goal, nc)
    assert (c["cls"], c["kind"]) == ("click-nav", "nav")
    c = classify_action("click[B078GWRC1J]", goal, nc)
    assert (c["kind"], c["correct"]) == ("product", True)
    c = classify_action("click[b01abcd123]", goal, nc)
    assert (c["kind"], c["correct"]) == ("product", False)
    c = classify_action("click[3 ounce]", goal, nc)
    assert (c["cls"], c["kind"], c["correct"]) == ("click-opcion-buy",
                                                   "option", True)
    c = classify_action("click[8 fl oz]", goal, nc)
    assert (c["kind"], c["correct"]) == ("option", False)
    g2 = {"asin": "X", "goal_options": {"color": "midnight black"}}
    c = classify_action("click[black]", g2, nc)  # normalize_color colapsa
    assert c["correct"] is True, c

    print("[self-test] 3/6 gating + auto-annealing exacto ...")
    comps = decompose_goal_components({
        "name": "Foo Bar", "attributes": ["light"], "goal_options": {},
        "price_upper": 20.0})
    assert serialize_residual_block(comps, [True] * len(comps)) == ""
    assert r_fwd_residual_value("", None, None) == 0.0
    blk = serialize_residual_block(comps, [True, False, False])
    assert "Foo Bar" not in blk and "light" in blk

    print("[self-test] 4/6 derangement P4 ...")
    for n in (2, 3, 5):
        eps = [Episode(idx=i, goal_true={"id": i}, goal_scoring={"id": i},
                       turns=[Turn("a", "o")], score=0.0) for i in range(n)]
        derange_goals(eps, seed=123)
        assert all(e.goal_scoring["id"] != e.goal_true["id"] for e in eps)

    print("[self-test] 5/6 informatividad / appearance ...")
    comps2 = decompose_goal_components({
        "name": "Acme Lamp", "attributes": ["dimmable"],
        "goal_options": {}, "price_upper": None})
    turns = [Turn("search[lamp]", "results: Acme Lamp - warm light"),
             Turn("click[B000000000]", "Acme Lamp page, DIMMABLE switch")]
    firsts = first_appearance_turns(comps2, turns)
    assert firsts == [1, 2], firsts  # case-insensitive, verbatim

    print("[self-test] 6/6 spans frontera wrapper->valor (tokenizer real) ...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    class _Eng:  # minimo para reusar prepare()
        tokenizer = tok
        max_tokens = 1 << 30
    eng = _Eng()
    tricky = ["3 ounce", "49.99 dollars", "Bright Citrus Deodorant by Earth "
              "Mama", "（width）24 inches", "x", "16.9 fl oz",
              "click[buy now]"]
    chat_avail = bool(getattr(tok, "chat_template", None))
    base_user = "Task: find something.\n[Action 1: 'search[a]' -> Observation 1: 'b']"
    if chat_avail:
        prefix_root = tok.apply_chat_template(
            [{"role": "user", "content": base_user}],
            add_generation_prompt=True, tokenize=False)
    else:
        prefix_root = base_user + "\n\nAssistant:\n"
    for value in tricky:
        for prefix in (prefix_root,                       # variante user
                       prefix_root + "The answer is:"):   # variante prefill
            v = (" " + value) if prefix.endswith(":") else value
            ids, s0, s1 = ScoringEngine.prepare(eng, SpanJob(prefix, v))
            decoded = tok.decode(ids[s0:s1 + 1])
            assert value in decoded or decoded.strip() == value.strip(), \
                (value, decoded)
    print(f"[self-test] OK — todos los checks pasaron "
          f"(tokenizer={args.model}, chat_template={chat_avail}).")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", type=Path,
                   help="JSONL de rollouts congelados (contrato en docstring)")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "float32", "bfloat16", "float16"])
    p.add_argument("--wrapper-variant", default="prefill",
                   choices=["prefill", "user"],
                   help="variante PRIMARIA (r_bwd, gating); la otra se "
                        "computa igual para P5 salvo --skip-secondary-wrapper")
    p.add_argument("--skip-secondary-wrapper", action="store_true",
                   help="no computar la segunda variante (P5 quedara coja)")
    p.add_argument("--history", default="window2",
                   choices=["window2", "full", "both"],
                   help="window2 replica la ventana del pipeline real "
                        "(env_manager.py:392); both agrega full para N.14 "
                        "(los scores primarios siguen siendo window2)")
    p.add_argument("--tau-percentile", type=float, default=50.0,
                   help="percentil por tipo para tau (prereg: p50)")
    p.add_argument("--length-norm", action="store_true",
                   help="ablation: r_fwd como media por token (default: suma "
                        "estilo iStar)")
    p.add_argument("--shuffled", action="store_true",
                   help="P4: rota goals entre episodios (derangement seeded)")
    p.add_argument("--shuffle-seed", type=int, default=20260612)
    p.add_argument("--baseline-json", type=Path,
                   help="(con --shuffled) JSON del run real para el verdict P4")
    p.add_argument("--out", type=Path, default=Path("figura1_scores.json"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--limit", type=int, default=0,
                   help="cap de episodios (0 = todos)")
    p.add_argument("--self-test", action="store_true",
                   help="corre los asserts de mecanica (tokenizer, sin modelo)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_test:
        run_self_test(args)
        return
    if not args.rollouts:
        raise SystemExit("[ERROR] --rollouts es obligatorio (o usa --self-test).")
    if args.baseline_json and not args.shuffled:
        raise SystemExit("[ERROR] --baseline-json solo tiene sentido con "
                         "--shuffled (verdict P4).")
    t0 = time.time()
    episodes = load_rollouts(args.rollouts, args.limit)
    print(f"[INFO] {len(episodes)} episodios cargados de {args.rollouts}.")
    if args.shuffled:
        derange_goals(episodes, args.shuffle_seed)
        print(f"[INFO] goals rotados (derangement, seed={args.shuffle_seed}) — "
              "scores con goal ajeno, clasificacion contra goal verdadero.")
    normalize_color = load_normalize_color()
    engine = ScoringEngine(args.model, args.device, args.dtype,
                           args.batch_size, args.max_tokens)
    pipe = run_pipeline(episodes, engine, args, normalize_color)
    predictions = analyze(pipe, args)
    p4 = None
    if args.shuffled and args.baseline_json:
        with args.baseline_json.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        p4 = p4_compare(predictions, baseline)
        predictions["P4"] = p4
    output = {
        "config": {
            "script": "tools/figura1_scoring.py",
            "prereg": "research/synthesis/figura1-prereg.md (CONGELADO "
                      "2026-06-12)",
            "rollouts": str(args.rollouts),
            "model": args.model,
            "device": engine.device,
            "wrapper_variant_primary": pipe["primary_variant"],
            "secondary_variant": pipe["secondary_variant"],
            "history": args.history,
            "history_mode_primary": pipe["primary_mode"],
            "tau_percentile": args.tau_percentile,
            "length_norm": args.length_norm,
            "shuffled": args.shuffled,
            "shuffle_seed": args.shuffle_seed if args.shuffled else None,
            "n_episodes": len(pipe["episodes_out"]),
            "elapsed_seconds": round(time.time() - t0, 1),
        },
        "tau": pipe["tau"],
        "predictions": predictions,
        "episodes": pipe["episodes_out"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[OK] escrito {args.out} "
          f"({args.out.stat().st_size / 1024:.1f} KB, "
          f"{output['config']['elapsed_seconds']}s)")
    print_summary(predictions, pipe["tau"], len(pipe["episodes_out"]),
                  args.shuffled, p4)


if __name__ == "__main__":
    main()
