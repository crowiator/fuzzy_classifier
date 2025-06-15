import numpy as np, pprint, json
rules = np.load("cache/cart_rules.npy", allow_pickle=True)
print("🔢 počet pravidiel:", len(rules))

# ukáž prvé tri v krajšom formáte
for r in rules[:3]:
    print("\n➡️  cieľ:", r["target"], " | podpora:", r["support"])
    for feat, op, thr in r["conds"]:
        print(f"   • {feat} {op} {thr}")