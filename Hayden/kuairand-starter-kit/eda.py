"""Phase-0 EDA for KuaiRand-Pure.

Answers the questions that gate the phase order in Refined-Strategy.md:
  1. How long are user histories really? (README claims "hundreds to thousands")
  2. How big are the within-user groups the listwise loss would operate on?
  3. What is the user composition (all-neg / all-pos / discriminative) on valid?
  4. What is the oracle ceiling on valid? (our real denominator)
  5. Which auxiliary feedback signals exist, and how dense are they? (multi-task viability)
  6. Cold users / cold items in valid relative to train.
  7. Duration <-> long_view coupling.
  8. train -> valid drift.

Read-only. Does not modify any kit file. Run:
    python3 eda.py [data_dir]
"""
import csv, os, sys, collections
import numpy as np
from evaluate import evaluate

D = sys.argv[1] if len(sys.argv) > 1 else './KuaiRand-Pure/data'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
LOGS = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')


def pct(x, ps=(50, 75, 90, 95, 99)):
    a = np.asarray(x)
    return {f'p{p}': float(np.percentile(a, p)) for p in ps}


def hr(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")


# ---------------------------------------------------------------- columns
hr("1. AVAILABLE COLUMNS (what signals exist for multi-task)")
with open(os.path.join(D, LOGS[0])) as fh:
    log_cols = next(csv.reader(fh))
print(f"log columns ({len(log_cols)}):")
for c in log_cols:
    print(f"   - {c}")

for f in ('user_features_pure.csv', 'video_features_basic_pure.csv',
          'video_features_statistic_pure.csv'):
    p = os.path.join(D, f)
    if os.path.exists(p):
        with open(p) as fh:
            cols = next(csv.reader(fh))
        print(f"\n{f} ({len(cols)} cols): {cols[:14]}{' ...' if len(cols) > 14 else ''}")

# ---------------------------------------------------------------- load logs
hr("2. LOADING LOGS")
rows = []            # (date, user, video, tab, duration, long_view, dict_of_aux)
AUX = [c for c in log_cols
       if c.startswith('is_') or c in ('play_time_ms', 'long_view', 'time_ms')]
for f in LOGS:
    with open(os.path.join(D, f)) as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r['date']), r['user_id'], r['video_id'], r['tab'],
                         float(r['duration_ms']),
                         1 if r['long_view'] != '0' else 0,
                         {c: r.get(c, '') for c in AUX}))
splits = {n: [x for x in rows if lo <= x[0] <= hi] for n, (lo, hi) in SPLITS.items()}
print({n: len(v) for n, v in splits.items()})

# ---------------------------------------------------------------- histories
hr("3. USER HISTORY LENGTH  (README claims 'hundreds to thousands')")
for n in ('train', 'valid', 'test'):
    c = collections.Counter(x[1] for x in splits[n])
    v = list(c.values())
    print(f"{n:6s} users={len(c):6,d}  rows/user mean={np.mean(v):7.1f} "
          f"min={min(v)} max={max(v)}  {pct(v)}")

all_users = collections.Counter(x[1] for x in rows)
print(f"\nAll splits combined: users={len(all_users):,d} "
      f"mean rows/user={np.mean(list(all_users.values())):.1f}")

# random-exposure log, for reference only (NOT training data - see Refined-Strategy)
rp = os.path.join(D, 'log_random_4_22_to_5_08_pure.csv')
if os.path.exists(rp):
    rc = collections.Counter()
    nrand = 0
    with open(rp) as fh:
        for r in csv.DictReader(fh):
            rc[r['user_id']] += 1
            nrand += 1
    print(f"log_random (reference only): rows={nrand:,d} users={len(rc):,d} "
          f"mean={nrand/len(rc):.1f}")

# ---------------------------------------------------------------- groups
hr("4. WITHIN-USER GROUP SIZES  (what listwise loss operates on)")
for n in ('train', 'valid', 'test'):
    byu = collections.Counter(x[1] for x in splits[n])
    byud = collections.Counter((x[1], x[0]) for x in splits[n])
    gu, gud = list(byu.values()), list(byud.values())
    print(f"{n:6s} group=(user)       n={len(gu):6,d} mean={np.mean(gu):6.1f} {pct(gu)}")
    print(f"{'':6s} group=(user,date)  n={len(gud):6,d} mean={np.mean(gud):6.1f} {pct(gud)}")
    sing = sum(1 for g in gud if g == 1)
    print(f"{'':6s}   -> (user,date) groups of size 1 (zero listwise gradient): "
          f"{sing:,d} = {100*sing/len(gud):.1f}%")

# ------------------------------------------------------- user composition
hr("5. USER COMPOSITION  (all-neg / all-pos / discriminative)")
print("Published test figures: all-neg 27.1% | all-pos 9.2% | discriminative 63.7%\n")
for n in ('train', 'valid', 'test'):
    byu = collections.defaultdict(list)
    for x in splits[n]:
        byu[x[1]].append(x[5])
    tot = len(byu)
    neg = sum(1 for v in byu.values() if sum(v) == 0)
    pos = sum(1 for v in byu.values() if sum(v) == len(v))
    dis = tot - neg - pos
    print(f"{n:6s} users={tot:6,d} | all-neg {100*neg/tot:5.1f}% | "
          f"all-pos {100*pos/tot:5.1f}% | discriminative {100*dis/tot:5.1f}%")

# ---------------------------------------------------------------- oracle
hr("6. ORACLE CEILING  (the real denominator for progress)")
print("Published: valid primary 0.8484 | test primary 0.8645\n")
for n in ('valid', 'test'):
    u = [x[1] for x in splits[n]]
    y = [x[5] for x in splits[n]]
    o = evaluate(u, y, [float(v) for v in y])         # true labels as scores
    print(f"{n:6s} ORACLE  GAUC {o['GAUC']:.4f} | nDCG@5 {o['nDCG@5']:.4f} | "
          f"primary {o['primary']:.4f}")

# ------------------------------------------------------------ aux signals
hr("7. AUXILIARY FEEDBACK SIGNALS  (multi-task viability)")
tr = splits['train']
for c in AUX:
    vals = [x[6][c] for x in tr]
    if c in ('play_time_ms', 'time_ms'):
        nums = np.array([float(v) for v in vals if v not in ('', None)])
        print(f"{c:16s} numeric  mean={nums.mean():12.1f} median={np.median(nums):12.1f}")
    else:
        rate = np.mean([1 if v not in ('0', '', None) else 0 for v in vals])
        print(f"{c:16s} positive rate = {rate:.4f}  ({rate*len(tr):,.0f} / {len(tr):,d})")

# --------------------------------------------------------------- coverage
hr("8. COLD USERS / COLD ITEMS  (valid & test vs train vocabulary)")
tru = {x[1] for x in tr}
trv = {x[2] for x in tr}
for n in ('valid', 'test'):
    s = splits[n]
    cu = sum(1 for x in s if x[1] not in tru)
    ci = sum(1 for x in s if x[2] not in trv)
    uu = {x[1] for x in s}
    ui = {x[2] for x in s}
    print(f"{n:6s} rows with unseen user: {cu:7,d} ({100*cu/len(s):5.2f}%) | "
          f"unseen video: {ci:7,d} ({100*ci/len(s):5.2f}%)")
    print(f"{'':6s} distinct users {len(uu):,d} ({100*len(uu-tru)/len(uu):.1f}% new) | "
          f"videos {len(ui):,d} ({100*len(ui-trv)/len(ui):.1f}% new)")

# --------------------------------------------------------------- duration
hr("9. DURATION <-> long_view COUPLING")
dur = np.array([x[4] for x in tr])
lab = np.array([x[5] for x in tr])
edges = np.quantile(dur, np.linspace(0, 1, 11)[1:-1])
b = np.searchsorted(edges, dur)
print(f"overall train long_view rate = {lab.mean():.4f}\n")
print(f"{'bucket':>6} {'n':>10} {'dur_ms p50':>12} {'long_view rate':>15}")
for i in range(11):
    m = b == i
    if m.sum():
        print(f"{i:>6} {m.sum():>10,d} {np.median(dur[m]):>12,.0f} {lab[m].mean():>15.4f}")

# ------------------------------------------------------------------ drift
hr("10. TRAIN -> VALID -> TEST DRIFT")
for n in ('train', 'valid', 'test'):
    s = splits[n]
    y = np.mean([x[5] for x in s])
    d = np.mean([x[4] for x in s])
    print(f"{n:6s} long_view rate {y:.4f} | mean duration_ms {d:10,.0f}")

print("\ntab distribution (share of rows):")
for n in ('train', 'valid', 'test'):
    c = collections.Counter(x[3] for x in splits[n])
    tot = sum(c.values())
    top = ', '.join(f"{k}:{100*v/tot:.1f}%" for k, v in c.most_common(6))
    print(f"{n:6s} {top}")

print("\nlong_view rate by date:")
for n in ('train', 'valid', 'test'):
    byd = collections.defaultdict(list)
    for x in splits[n]:
        byd[x[0]].append(x[5])
    ds = sorted(byd)
    print(f"{n:6s} " + ' '.join(f"{d%100:02d}:{np.mean(byd[d]):.3f}" for d in ds))

print("\nEDA complete.")
