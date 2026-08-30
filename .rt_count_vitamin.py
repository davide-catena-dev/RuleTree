import csv
from pathlib import Path
from collections import Counter
p = Path(r"c:\Users\david\RuleTree\datasets\CLF\vitamin.csv")
with p.open(newline='') as f:
    r = csv.reader(f)
    header = next(r)
    try:
        idx = header.index("disease_diagnosis")
    except ValueError:
        print("COLUMN_NOT_FOUND", header)
        raise SystemExit(1)
    c = Counter(row[idx] for row in r)
    total = sum(c.values())
    print(total)
    for k, v in c.most_common():
        print(f"{k}\t{v}")
