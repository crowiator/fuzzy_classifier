import numpy as np
from pathlib import Path
from collections import defaultdict
# Nacitaj pravidla
rules_path = Path("cache/cart_rules.npy")
rules = np.load(rules_path, allow_pickle=True)
print(len(rules))
# === Spočítaj pravidlá pre každú triedu ===
rule_counts = defaultdict(int)
support_counts = defaultdict(int)
for rule in rules:
    label = rule ["target"]
    rule_counts [label] += 1
    support_counts[label] += rule["support"]
print(rule_counts)
print(support_counts)
# === Výpis ===
print("\n\u2728 \u0160tatistika pravidiel podľa triedy:\n")
print(f"{'Trieda':<10} {'# pravidiel':>12} {'Podpora':>12}")
print("-"*36)
for label in sorted(rule_counts.keys()):
    print(f"{label:<10} {rule_counts[label]:>12} {support_counts[label]:>12}")

# === Zhrnutie ===
total_rules = sum(rule_counts.values())
total_support = sum(support_counts.values())
print("\n∑ Počet pravidiel:", total_rules)
print("∑ Celková podpora:", total_support)

# === Odporúčanie na zosilnenie ===
avg_support = total_support / total_rules
print("\n⚠ï¸ Potenciálne poddimenzované triedy:")
for label in sorted(rule_counts.keys()):
    if support_counts[label] < avg_support * 0.5:
        print(f" - {label}: podpora {support_counts[label]} < 50% priemernej ({avg_support:.1f})")
