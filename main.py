"""GTA VI trophy rarity forecast for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for each of GTA VI's 55 PS5 trophies, the earn rate Exophase will
show at the case's target, as five quantiles (0.05/0.25/0.5/0.75/0.95).

Model, in outline
-----------------
Exophase only changes the numbers it shows when it refreshes a game's page,
and a refresh re-bases the player count: the tracked-player base jumps (the
launch-night page sat unchanged at 638 players for hours, then jumped to
3,868 with every rate but the 100% one lower), while earners only grow as
players actually play. So a rate is a race between two counts:

    rate(t) = earners(t) / players_tracked(t)

and the forecast is built from those two counts, on the logit scale, never
from the last rate alone:

* When the history shows movement (the final case), fit, per trophy, the
  growth of earners and of the tracked-player base against log time since
  release, with recency weights, shrink each trophy's earner slope toward its
  tier's median slope, and extrapolate both counts to the target with damped
  slopes (growth slows after the launch surge). Blend that with a direct fit
  of the logit rate against log time. The spread grows with the horizon and
  with each trophy's fit residual, so a trophy whose history is noisy gets a
  wider band.

* When the history is flat -- every read the same stale page -- no trend can
  be fit. Then the forecast is a two-outcome mixture: either Exophase
  refreshes before the target (probability from a Weibull hazard on the
  refresh interval, the page being already stale for hours, tilted up because
  a refresh is the dominant event for a stale page) and the rate is diluted
  by the player jump, earners having grown only by early play; or it does
  not refresh and the page still shows the old rates. The quantiles are the
  mixture's quantiles, so the low quantiles sit at the diluted rate and the
  high ones at the stale rate.

Edge trophies: a trophy at 100% stays near 100% (everyone tracked has it);
a trophy at 0% is floored at 0.01% before the logit. All quantiles are
clamped to [0.01, 99.99] (100 allowed at the top for the 100% trophy) and
forced non-decreasing. No randomness anywhere: two runs are byte-identical.

Standard library only; reads every <id>.in in $MAKEFUN_INPUT_DIR (else
/input) and writes <id>.out to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# A refresh's typical player-base jump on launch week (first refresh went
# 638 -> 3,868 tracked players, about 6x) and the refresh cadence the
# launch-night page implies: unchanged for ~4.7 h, then refreshed.
REFRESH_JUMP = 6.0
REFRESH_SCALE_H = 5.0
REFRESH_SHAPE = 3.0

# How much earners grow between the page's first sighting and the target,
# by how early the trophy is (its last rate is the earliness proxy).
EARNER_GROWTH_MAX = 2.0

# Loose plausibility caps on a median rate, by tier, after a week of play.
TIER_CAP = {"platinum": 8.0, "gold": 25.0, "silver": 50.0, "bronze": 90.0}
# Earner-growth slope prior, per tier, for shrinkage in the trend fit.
TIER_SLOPE = {"platinum": 0.5, "gold": 0.8, "silver": 1.1, "bronze": 1.3}


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JSON output)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def clean_quantiles(qs):
    out = []
    prev = 0.01
    for q in qs:
        q = min(100.0, max(0.01, q))
        q = max(q, prev)
        out.append(round4(q))
        prev = out[-1]
    return out


def wls(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b, rms_resid)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(
        sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    )
    return a, b, resid


def distinct_reads(case):
    """Collapse consecutive identical reads; return [(t_hours, N, rates)]."""
    out = []
    for read in case["history"]:
        t = max(0.05, (parse_time(read["at"]) - RELEASE).total_seconds() / 3600.0)
        n = read["players_tracked"]
        rates = read["rates"]
        if out and out[-1][1] == n and out[-1][2] == rates:
            continue
        out.append((t, n, rates))
    return out


def flat_history_forecast(case, reads, horizon_h):
    """Two-outcome mixture for a history that never moved (stale page).

    Refresh outcome (weight p): players jump REFRESH_JUMP-fold, earners
    grow only by early play, so the rate is diluted. No-refresh outcome
    (weight 1-p): the page still shows the old rates at the target.
    """
    first_t = reads[0][0]
    checkpoint_t = max(
        0.05, (parse_time(case["at"]) - RELEASE).total_seconds() / 3600.0
    )
    stale_h = max(0.0, checkpoint_t - first_t)
    # Weibull hazard: P(refresh in the next horizon_h | none yet).
    p_weibull = 1.0 - math.exp(
        -(((stale_h + horizon_h) / REFRESH_SCALE_H) ** REFRESH_SHAPE)
        + ((stale_h / REFRESH_SCALE_H) ** REFRESH_SHAPE)
    )
    # A stale page's refresh is the dominant event for the target, so the
    # refresh outcome carries more weight than the raw hazard alone.
    p_refresh = min(0.80, p_weibull + 0.30)

    last_rates = reads[-1][2]
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last_rates[i]
        if r is None:
            r = 1.0
        if r >= 99.99:
            # The one trophy everyone tracked has: it stays at ~100%.
            rows.append({"id": trophy["id"], "q": [99.9, 99.95, 99.99, 100, 100]})
            continue
        earliness = r / (r + 1.0)
        earner_growth = 1.0 + EARNER_GROWTH_MAX * earliness
        refresh_rate = clamp_pct(r * earner_growth / REFRESH_JUMP)
        lo_r, hi_r = logit(refresh_rate), logit(r)
        sig_r, sig_n = 0.5, 0.08

        def mixture_cdf(l):
            return p_refresh * norm_cdf((l - lo_r) / sig_r) + (
                1.0 - p_refresh
            ) * norm_cdf((l - hi_r) / sig_n)

        qs = []
        lo, hi = logit(0.01), logit(99.99)
        for tau in LEVELS:
            a, b = lo, hi
            for _ in range(60):
                mid = 0.5 * (a + b)
                if mixture_cdf(mid) < tau:
                    a = mid
                else:
                    b = mid
            qs.append(inv_logit(0.5 * (a + b)))
        rows.append({"id": trophy["id"], "q": clean_quantiles(qs)})
    return {"trophies": rows}


def trend_forecast(case, reads, horizon_h):
    """Fit earners and players against log time since release; extrapolate."""
    m = len(reads)
    xs = [math.log1p(t) for t, _, _ in reads]
    ws = [0.2 + 0.8 * (j / (m - 1)) if m > 1 else 1.0 for j in range(m)]
    x_last = xs[-1]
    target_t = max(
        0.05, (parse_time(case["target"]) - RELEASE).total_seconds() / 3600.0
    )
    x_target = math.log1p(target_t)
    dx = max(0.0, x_target - x_last)

    # Tracked-player base: fit log N, extrapolate with a damped slope.
    log_n = [math.log(max(1, n)) for _, n, _ in reads]
    a_n, b_n, _ = wls(xs, log_n, ws)
    log_n_last = a_n + b_n * x_last
    log_n_target = log_n_last + (0.7 * b_n if b_n > 0 else b_n) * dx
    n_target = math.exp(log_n_target)
    n_target = min(max(n_target, reads[-1][1]), reads[-1][1] * 25.0, 3_000_000.0)

    # Per-trophy earner slopes, shrunk toward the tier median slope.
    n_trophies = len(case["trophies"])
    slopes = {}
    tier_slopes = {}
    for i, trophy in enumerate(case["trophies"]):
        ys, x2, w2 = [], [], []
        for j, (t, n, rates) in enumerate(reads):
            r = rates[i]
            if r is None:
                continue
            e = max(0.25, r / 100.0 * n)
            ys.append(math.log(e))
            x2.append(xs[j])
            w2.append(ws[j])
        if len(ys) >= 2:
            _, b_e, resid = wls(x2, ys, w2)
        else:
            b_e, resid = TIER_SLOPE.get(trophy["tier"], 1.0), 0.5
        slopes[i] = (b_e, resid)
        tier_slopes.setdefault(trophy["tier"], []).append(b_e)
    tier_median = {
        tier: sorted(v)[len(v) // 2] for tier, v in tier_slopes.items()
    }

    horizon_days = horizon_h / 24.0
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_r = None
        for _, _, rates in reversed(reads):
            if rates[i] is not None:
                last_r = rates[i]
                break
        if last_r is None:
            last_r = 1.0
        if last_r >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.9, 99.95, 99.99, 100, 100]})
            continue

        b_e, resid = slopes[i]
        prior = tier_median.get(trophy["tier"], TIER_SLOPE.get(trophy["tier"], 1.0))
        w_fit = m / (m + 4.0)
        b_e = w_fit * b_e + (1.0 - w_fit) * prior

        # Earners at the target, from the last read that lists this trophy.
        e_last = None
        x_e_last = x_last
        for j in range(m - 1, -1, -1):
            r = reads[j][2][i]
            if r is not None:
                e_last = max(0.25, r / 100.0 * reads[j][1])
                x_e_last = xs[j]
                break
        if e_last is None:
            e_last = 0.25
        e_target = e_last * math.exp(0.85 * b_e * max(0.0, x_target - x_e_last))
        e_target = min(e_target, 0.999 * n_target)
        rate_decomp = 100.0 * e_target / n_target

        # Direct fit of the logit rate against log time, damped.
        ys_logit, x2, w2 = [], [], []
        for j, (t, n, rates) in enumerate(reads):
            r = rates[i]
            if r is None:
                continue
            ys_logit.append(logit(r))
            x2.append(xs[j])
            w2.append(ws[j])
        if len(ys_logit) >= 2:
            a_r, b_r, _ = wls(x2, ys_logit, w2)
            logit_direct = (a_r + b_r * x_last) + 0.75 * b_r * dx
        else:
            logit_direct = logit(last_r)
        median_logit = 0.5 * (logit(rate_decomp) + logit_direct)

        cap = TIER_CAP.get(trophy["tier"], 90.0)
        median_logit = min(median_logit, logit(cap))
        median_logit = max(median_logit, logit(0.01))

        sigma = math.sqrt(
            0.30 ** 2
            + (0.45 * math.sqrt(horizon_days)) ** 2
            + (0.6 * min(resid, 1.5)) ** 2
            + 0.20 ** 2
        )
        qs = [inv_logit(median_logit + z * sigma) for z in Z]
        rows.append({"id": trophy["id"], "q": clean_quantiles(qs)})
    return {"trophies": rows}


def fallback_forecast(case):
    rows = []
    prior = {"platinum": 0.1, "gold": 0.3, "silver": 0.5, "bronze": 1.0}
    for trophy in case["trophies"]:
        med = prior.get(trophy["tier"], 1.0)
        qs = [inv_logit(logit(med) + z * 1.5) for z in Z]
        rows.append({"id": trophy["id"], "q": clean_quantiles(qs)})
    return {"trophies": rows}


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("unexpected levels")
    history = case.get("history") or []
    if not history:
        return fallback_forecast(case)
    reads = distinct_reads(case)
    horizon_h = case["horizon_minutes"] / 60.0
    if len(reads) <= 1:
        return flat_history_forecast(case, reads, horizon_h)
    return trend_forecast(case, reads, horizon_h)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
