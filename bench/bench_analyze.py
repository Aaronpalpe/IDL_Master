#!/usr/bin/env python3
"""
bench_analyze.py — Compare benchmark results from bench_submit.sh runs.

Usage:
    python3 bench_analyze.py <file_A.csv> <file_B.csv> [--metric elapsed_s] [--alpha 0.05] [--out comparison.csv]

Compares a metric between two experiment CSVs and runs statistical tests
to determine if the difference is significant.
"""

import argparse
import csv
import math
import sys

VALID_METRICS = [
    "elapsed_s", "total_cpu_s", "user_cpu_s", "system_cpu_s",
    "max_rss_kb", "max_vmsize_kb", "ave_rss_kb",
]


def read_metric(filepath, metric):
    """Read a single metric column from a bench CSV, filtering to COMPLETED jobs."""
    values = []
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if metric not in reader.fieldnames:
            print(f"Error: metric '{metric}' not found in {filepath}", file=sys.stderr)
            print(f"Available columns: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)
        for row in reader:
            if row["state"].strip() != "COMPLETED":
                continue
            val = row[metric].strip()
            if val == "":
                continue
            values.append(float(val))
    return values


def descriptive_stats(values):
    """Compute descriptive statistics using only the standard library."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "median": float("nan"), "min": float("nan"), "max": float("nan"),
                "ci95_lo": float("nan"), "ci95_hi": float("nan")}

    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0

    sorted_v = sorted(values)
    if n % 2 == 1:
        median = sorted_v[n // 2]
    else:
        median = (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2

    # 95% CI using t-distribution approximation (t ≈ 1.96 for large n,
    # use a lookup for small n)
    t_values = {
        2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262, 11: 2.228,
        12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
        17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 25: 2.064,
        30: 2.045, 40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990,
        100: 1.984,
    }
    if n < 2:
        t_crit = 0
    elif n in t_values:
        t_crit = t_values[n]
    else:
        # Interpolate or use closest
        closest = min(t_values.keys(), key=lambda k: abs(k - n))
        t_crit = t_values[closest]

    margin = t_crit * std / math.sqrt(n) if n > 1 else 0
    ci95_lo = mean - margin
    ci95_hi = mean + margin

    return {
        "n": n, "mean": mean, "std": std, "median": median,
        "min": min(values), "max": max(values),
        "ci95_lo": ci95_lo, "ci95_hi": ci95_hi,
    }


def cohens_d(vals_a, vals_b):
    """Compute Cohen's d effect size."""
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_a = sum(vals_a) / n_a
    mean_b = sum(vals_b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in vals_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in vals_b) / (n_b - 1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_a - mean_b) / pooled_std


def effect_size_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def run_tests(vals_a, vals_b, alpha):
    """Run statistical tests. Requires scipy."""
    try:
        from scipy import stats
    except ImportError:
        print("=" * 60, file=sys.stderr)
        print("ERROR: scipy is required for statistical tests.", file=sys.stderr)
        print("Install it with:  pip3 install --user scipy numpy", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        sys.exit(1)

    results = {}

    # Normality tests (need n >= 3)
    if len(vals_a) >= 3:
        stat_a, p_a = stats.shapiro(vals_a)
        results["shapiro_a"] = {"statistic": stat_a, "p": p_a, "normal": bool(p_a > alpha)}
    else:
        results["shapiro_a"] = {"statistic": None, "p": None, "normal": None}

    if len(vals_b) >= 3:
        stat_b, p_b = stats.shapiro(vals_b)
        results["shapiro_b"] = {"statistic": stat_b, "p": p_b, "normal": bool(p_b > alpha)}
    else:
        results["shapiro_b"] = {"statistic": None, "p": None, "normal": None}

    both_normal = (results["shapiro_a"]["normal"] is True and
                   results["shapiro_b"]["normal"] is True)

    # Main comparison test
    if both_normal:
        stat, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        results["test"] = {
            "name": "Welch's t-test", "statistic": stat, "p": p,
            "significant": p < alpha,
        }
    else:
        stat, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        results["test"] = {
            "name": "Mann-Whitney U", "statistic": stat, "p": p,
            "significant": p < alpha,
        }

    return results


def fmt(val, decimals=3):
    """Format a number for display."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "N/A"
    if isinstance(val, int):
        return str(val)
    return f"{val:.{decimals}f}"


def label_from_path(path):
    """Extract a short label from a CSV file path."""
    import os
    name = os.path.basename(path)
    if name.endswith(".csv"):
        name = name[:-4]
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Compare benchmark results between two experiments.")
    parser.add_argument("file_a", help="CSV from first experiment (e.g., baseline)")
    parser.add_argument("file_b", help="CSV from second experiment (e.g., optimized)")
    parser.add_argument("--metric", default="elapsed_s", choices=VALID_METRICS,
                        help="Metric to compare (default: elapsed_s)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance level (default: 0.05)")
    parser.add_argument("--out", default=None,
                        help="Write comparison results to this CSV file")
    args = parser.parse_args()

    label_a = label_from_path(args.file_a)
    label_b = label_from_path(args.file_b)

    vals_a = read_metric(args.file_a, args.metric)
    vals_b = read_metric(args.file_b, args.metric)

    if not vals_a or not vals_b:
        print("Error: one or both files have no COMPLETED jobs.", file=sys.stderr)
        sys.exit(1)

    stats_a = descriptive_stats(vals_a)
    stats_b = descriptive_stats(vals_b)

    d = cohens_d(vals_a, vals_b)
    test_results = run_tests(vals_a, vals_b, args.alpha)

    # ── Pretty print ────────────────────────────────────────────────────────
    w = 60
    print("=" * w)
    print(f"  Benchmark Comparison: {args.metric}")
    print(f"  Alpha: {args.alpha}")
    print("=" * w)
    print("")

    # Descriptive stats table
    hdr = f"{'':>18}  {'A: ' + label_a:>18}  {'B: ' + label_b:>18}"
    print(hdr)
    print("-" * len(hdr))
    for field in ["n", "mean", "std", "median", "min", "max", "ci95_lo", "ci95_hi"]:
        va = stats_a[field]
        vb = stats_b[field]
        print(f"  {field:>16}  {fmt(va):>18}  {fmt(vb):>18}")
    print("")

    # Percent change
    if stats_a["mean"] != 0:
        pct = (stats_b["mean"] - stats_a["mean"]) / stats_a["mean"] * 100
        direction = "faster" if pct < 0 else "slower"
        print(f"  Change: {fmt(abs(pct), 1)}% {direction} (B relative to A)")
    print("")

    # Normality
    print("  Normality (Shapiro-Wilk):")
    sha = test_results["shapiro_a"]
    shb = test_results["shapiro_b"]
    print(f"    A: p={fmt(sha['p'])}  {'normal' if sha['normal'] else 'non-normal' if sha['normal'] is not None else 'N/A (n<3)'}")
    print(f"    B: p={fmt(shb['p'])}  {'normal' if shb['normal'] else 'non-normal' if shb['normal'] is not None else 'N/A (n<3)'}")
    print("")

    # Main test
    t = test_results["test"]
    print(f"  Test: {t['name']}")
    print(f"    Statistic: {fmt(t['statistic'])}")
    print(f"    p-value:   {fmt(t['p'])}")
    print(f"    Significant at alpha={args.alpha}? {'YES' if t['significant'] else 'NO'}")
    print("")

    # Effect size
    print(f"  Cohen's d: {fmt(d)} ({effect_size_label(d)})")
    print("")
    print("=" * w)

    # ── Optional CSV output ─────────────────────────────────────────────────
    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "metric", "label_a", "label_b",
                "n_a", "mean_a", "std_a", "median_a", "ci95_lo_a", "ci95_hi_a",
                "n_b", "mean_b", "std_b", "median_b", "ci95_lo_b", "ci95_hi_b",
                "pct_change", "test_name", "statistic", "p_value",
                "significant", "cohens_d", "effect_size",
            ])
            pct = ((stats_b["mean"] - stats_a["mean"]) / stats_a["mean"] * 100
                   if stats_a["mean"] != 0 else float("nan"))
            writer.writerow([
                args.metric, label_a, label_b,
                stats_a["n"], stats_a["mean"], stats_a["std"], stats_a["median"],
                stats_a["ci95_lo"], stats_a["ci95_hi"],
                stats_b["n"], stats_b["mean"], stats_b["std"], stats_b["median"],
                stats_b["ci95_lo"], stats_b["ci95_hi"],
                pct, t["name"], t["statistic"], t["p"],
                t["significant"], d, effect_size_label(d),
            ])
        print(f"  Comparison CSV written to: {args.out}")


if __name__ == "__main__":
    main()
