#!/usr/bin/env python3
"""Recompute the token and relay numbers of the cost-efficient-LLM talk.

Reads the per-session rows of the decide relay-cost API (or a saved copy of
its JSON) and prints: token totals, the share of tokens per kind, the share of
the bill per kind (input-token equivalents, ITE, and list-price USD), requests
per typed prompt for interactive sessions, and relay statistics.

ITE = in + 2 write + 0.1 read + 5 out. The API does not split cache writes
into 5-minute and 1-hour writes, so all writes are weighted with -w (default 2,
the 1-hour price of a main session).

Read-only: one HTTP GET (or one file read), output on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import urllib.request
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_URL = os.environ.get("RELAY_COST_URL", "")
TZ = ZoneInfo("Europe/Zurich")
KINDS = ("tok_in", "tok_cw", "tok_cr", "tok_out")
LABELS = {
    "tok_in": "fresh input",
    "tok_cw": "cache write",
    "tok_cr": "cache read",
    "tok_out": "output",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "-u",
        "--url",
        default=DEFAULT_URL,
        help="relay-cost API URL (default: environment variable RELAY_COST_URL)",
    )
    p.add_argument(
        "-f", "--file", help="read a saved API response instead of fetching the URL"
    )
    p.add_argument("-j", "--json", action="store_true", help="print the result as JSON")
    p.add_argument(
        "-w",
        "--write-weight",
        type=float,
        default=2.0,
        help="ITE weight of a cache write (default 2.0 = 1-hour write; 1.25 = 5-minute write)",
    )
    return p.parse_args(argv)


def load(url: str, path: str | None) -> dict[str, Any]:
    """Return the API document from a file or the URL."""
    if path:
        with open(path, encoding="utf-8") as fh:
            data: dict[str, Any] = json.load(fh)
        return data
    if not url:
        raise ValueError("no URL: pass -u/--url, set RELAY_COST_URL or use -f/--file")
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (operator-given URL)
        data = json.loads(resp.read().decode("utf-8"))
    return data


def median(values: list[float]) -> float | None:
    """Median, or None for an empty list."""
    return statistics.median(values) if values else None


def share(part: float, total: float) -> float:
    """Part / total, 0 when total is 0."""
    return part / total if total else 0.0


def token_stats(rows: list[dict[str, Any]], w_write: float) -> dict[str, Any]:
    """Totals, token shares and bill shares per token kind."""
    totals = {k: sum(int(r.get(k) or 0) for r in rows) for k in KINDS}
    weights = {"tok_in": 1.0, "tok_cw": w_write, "tok_cr": 0.1, "tok_out": 5.0}
    ite = {k: totals[k] * weights[k] for k in KINDS}
    all_tok = sum(totals.values())
    all_ite = sum(ite.values())
    return {
        "sessions": len(rows),
        "requests": sum(int(r.get("requests") or 0) for r in rows),
        "tokens": totals,
        "token_share": {k: share(totals[k], all_tok) for k in KINDS},
        "ite": ite,
        "ite_total": all_ite,
        "ite_share": {k: share(ite[k], all_ite) for k in KINDS},
        "usd_total": sum(float(r.get("usd") or 0.0) for r in rows),
        "usd_partial_rows": sum(1 for r in rows if r.get("usd_partial")),
    }


def usd_shares(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """USD per kind, derivable only for rows of one model at Opus 5.5 list prices.

    The API gives one USD total per row, not per kind. For claude-opus-5-5 rows the
    list prices are known ($4 in, $8 1-h write, $0.20 read, $20 out per million), so
    the split is derived there and checked against the row totals.
    """
    price = {"tok_in": 4.0, "tok_cw": 8.0, "tok_cr": 0.20, "tok_out": 20.0}
    sel = [r for r in rows if r.get("model") == "claude-opus-5-5"]
    usd = {k: sum(int(r.get(k) or 0) for r in sel) * price[k] / 1e6 for k in KINDS}
    tot = sum(usd.values())
    return {
        "model": "claude-opus-5-5",
        "sessions": len(sel),
        "usd": usd,
        "usd_share": {k: share(usd[k], tot) for k in KINDS},
        "usd_derived_total": tot,
        "usd_api_total": sum(float(r.get("usd") or 0.0) for r in sel),
    }


def interactive_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Start/max context, requests and requests per typed prompt of interactive sessions."""
    inter = [r for r in rows if r.get("type") == "interactive"]
    with_prompts = [r for r in inter if int(r.get("user_prompts") or 0) > 0]
    ratios = [
        int(r.get("requests") or 0) / int(r["user_prompts"]) for r in with_prompts
    ]
    req = sum(int(r.get("requests") or 0) for r in with_prompts)
    asks = sum(int(r["user_prompts"]) for r in with_prompts)
    return {
        "sessions": len(inter),
        "median_ctx_start": median([float(r.get("ctx_start") or 0) for r in inter]),
        "median_ctx_max": median([float(r.get("ctx_max") or 0) for r in inter]),
        "median_requests": median([float(r.get("requests") or 0) for r in inter]),
        "n_with_user_prompts": len(with_prompts),
        "requests_per_prompt_median": median(ratios),
        "requests_per_prompt_pooled": share(req, asks),
    }


def relay_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count and medians of all recorded relays (handovers)."""
    hos = [h for r in rows for h in (r.get("handovers") or [])]
    fees = [
        float(h["handover_ite"]) + float(h.get("tail_ite") or 0)
        for h in hos
        if h.get("handover_ite") is not None
    ]
    c_at = [float(h["c_at_relay"]) for h in hos if h.get("c_at_relay") is not None]
    with_verdict = [h for h in hos if h.get("verdict")]
    return {
        "relays": len(hos),
        "median_c_at_relay": median(c_at),
        "n_c_at_relay": len(c_at),
        "median_handover_plus_tail_ite": median(fees),
        "n_fee": len(fees),
        "share_fastpath": share(
            sum(1 for h in hos if h.get("path") == "fastpath"), len(hos)
        ),
        "share_verdict_good": share(
            sum(1 for h in with_verdict if h.get("verdict") == "good"),
            len(with_verdict),
        ),
        "n_verdict": len(with_verdict),
    }


def fmt_ts(ts: float | None) -> str:
    """Unix time as Europe/Zurich text."""
    if not ts:
        return "-"
    return datetime.fromtimestamp(float(ts), TZ).strftime("%Y-%m-%d %H:%M")


def report(doc: dict[str, Any], w_write: float) -> dict[str, Any]:
    """Assemble every statistic."""
    rows: list[dict[str, Any]] = doc.get("rows") or []
    return {
        "run_at": datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z"),
        "since": fmt_ts(doc.get("since")),
        "until": fmt_ts(doc.get("until")) if doc.get("until") else "now",
        "truncated": bool(doc.get("truncated")),
        "write_weight": w_write,
        "all": token_stats(rows, w_write),
        "usd_split_opus_5_5": usd_shares(rows),
        "interactive": interactive_stats(rows),
        "relays": relay_stats(rows),
    }


def pct(x: float) -> str:
    """Percent text."""
    return f"{100 * x:.1f}%"


def print_text(res: dict[str, Any]) -> None:
    """Human-readable output."""
    a = res["all"]
    print(
        f"run {res['run_at']} · window {res['since']} .. {res['until']}"
        f" · truncated {res['truncated']}"
    )
    print(
        f"sessions {a['sessions']:,} · requests {a['requests']:,}"
        f" · usd (list) ${a['usd_total']:,.0f}"
    )
    print(f"{'kind':<12} {'tokens':>16} {'tok share':>9} {'ITE share':>9}")
    for k in KINDS:
        print(
            f"{LABELS[k]:<12} {a['tokens'][k]:>16,}"
            f" {pct(a['token_share'][k]):>9} {pct(a['ite_share'][k]):>9}"
        )
    print(
        f"ITE total {a['ite_total'] / 1e9:.2f} billion (write weight {res['write_weight']})"
    )
    u = res["usd_split_opus_5_5"]
    shares = " · ".join(f"{LABELS[k]} {pct(u['usd_share'][k])}" for k in KINDS)
    print(
        f"USD split, {u['model']} ({u['sessions']} sessions, derived ${u['usd_derived_total']:,.0f}"
        f" vs API ${u['usd_api_total']:,.0f}): {shares}"
    )
    i = res["interactive"]
    print(
        f"interactive {i['sessions']}: median start {i['median_ctx_start']:,.0f}"
        f" · max {i['median_ctx_max']:,.0f}"
        f" · requests {i['median_requests']}"
    )
    print(
        f"  requests per typed prompt (n={i['n_with_user_prompts']} with user_prompts > 0):"
        f" median {i['requests_per_prompt_median']} · pooled {i['requests_per_prompt_pooled']:.1f}"
    )
    r = res["relays"]
    print(
        f"relays {r['relays']}: median c_at_relay {r['median_c_at_relay']} (n={r['n_c_at_relay']})"
        f" · median handover+tail ITE {r['median_handover_plus_tail_ite']} (n={r['n_fee']})"
        f" · fastpath {pct(r['share_fastpath'])}"
        f" · verdict good {pct(r['share_verdict_good'])} (n={r['n_verdict']})"
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    try:
        doc = load(args.url, args.file)
    except (OSError, ValueError) as exc:
        print(f"error: cannot load relay-cost data: {exc}", file=sys.stderr)
        return 1
    res = report(doc, args.write_weight)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_text(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
