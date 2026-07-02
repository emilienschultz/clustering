# Bug report — clustering robustness pipeline

Audit of `process_dataset.py` and its dependencies (`src/model_select.py`,
`src/model_eval.py`, `src/model_fit.py`), 2026-07-02.

**Provenance.** All bugs below except #4 are inherited **verbatim** from the
parent project `../Scales-of-Nationalism` (the `src/` modules are byte-identical
copies; `4_clustering.ipynb` imports these same functions). They therefore
affect the parent's real-data results (`output/models/*.csv`, the
`5_clustering_results_*` write-ups, `real_result.pkl`, the working-paper
Table 2) as much as this repo's app and benchmark. The gap-statistic machinery
is this project's own methodology (Bonikowski & DiMaggio 2016 used LCA-only
tooling), so none of this traces back further than `Scales-of-Nationalism`.

Line numbers refer to this repo's current files.

---

## 1. Gap statistic is orientation-inconsistent across validity indices — FIXED (branch `fix_bugs`, 2026-07-02)

**Where.** `src/model_select.py` — `compute_gap` (line 73), `get_gap` (lines 100–110).

**Problem.** The gap is computed identically for every CVI:

```python
gap = np.log(np.mean(rand_ind)) - np.log(mod_ind)
```

This is Tibshirani's formula, designed for within-cluster dispersion — a
**lower-is-better** quantity. Of the four CVIs, only Davies-Bouldin is
lower-is-better. For silhouette, Calinski-Harabasz and Dunn (higher-is-better)
the gap curve comes out **inverted**: more structure = lower gap.

`get_gap` then applies one selection rule to all indices, with a further
deviation from the standard: instead of Tibshirani's "smallest k such that
Gap(k) ≥ Gap(k+1) − s(k+1)", it collects **all** satisfying k and returns the
one with the smallest margin (`argmin` of `gap[k] − gap[k+1] + s[k+1]`).

Working through the geometry: for the three inverted indices the combined rule
behaves like an elbow detector on the correctly-oriented curve (defensible);
for Davies-Bouldin the same rule selects on the *declining* side of the curve
and the smallest-margin choice drifts toward the plateau.

**Effect on results.** The k selected "by Davies-Bouldin" is chosen by a
qualitatively different criterion than the k selected by the other three
indices, with a tendency toward larger k. Every downstream artefact that uses
gap-selected candidates per index — `candidate_models`, `selected_pools`, the
Table 2 comparison, the app's solution picker, the benchmark's per-index best
solutions — treats these selections as comparable when they are not.

**Fix applied** (`compute_gap`): every index is now oriented to its normal
logic — a larger gap uniformly means "more structure than the uniform null".
Davies-Bouldin (lower-is-better, like Tibshirani's within-dispersion) keeps
`gap = log(mean(rand)) − log(mod)`; silhouette / Calinski-Harabasz / Dunn
(higher-is-better) are flipped to `gap = log(mod) − log(mean(rand))`
(`LOWER_IS_BETTER` set in `src/model_select.py`). The argmin-margin selection
rule in `get_gap` is kept, but it now has the same semantics for all four
indices. **This changes which k gets selected** — silhouette/CH/Dunn
selections move from the old inverted-curve behaviour to the same
stops-increasing criterion as DB — so the parent's real-data pipeline should
be rerun and compared before/after when porting this fix.

## 2. Uniform null reference is inflated by +1 for the distance-based models — FIXED (branch `fix_bugs`, 2026-07-02)

**Where.** `src/model_select.py` — `bootstrap_gap`, lines 18–21:

```python
rand_data = np.random.uniform(low=data.min(axis=0), high=data.max(axis=0) + 1, ...)
```

**Problem.** The `+1` exists for the latent path: StepMix's categorical
measurement integer-casts floats (verified empirically), so for 0–4 Likert
data, uniform on [0, 5) floors to uniform integers 0–4 — correct. But the same
function receives **standardized continuous data** for k-means and AHC
(`process_dataset.py` passes `data_n`), where the `+1` extends the null
reference a full unit above the observed max in every dimension — on
standardized columns (range ≈ ±2.5) that is roughly 40% extra range, applied
asymmetrically upward.

**Effect on results.** The reference datasets against which k-means and AHC
gaps are computed are systematically more dispersed than the data's bounding
box. This shifts the level of every distance-model gap curve and inflates `s`.
The bias is shared across k within one (model, params), so it partially
cancels in the consecutive-k differences the selection rule uses — the
selected k is distorted only second-order — but every reported gap value and
tolerance for k-means/AHC is biased. Latent selections are unaffected.

**Fix applied** (`bootstrap_gap`): the `+1` is now conditional on the model
being a *categorical* latent one (checking `params['msrt']`, since a
continuous-measurement latent model also runs on standardized floats):

```python
categorical = model == 'latent' and 'categorical' in params.get('msrt', '')
high = data.max(axis=0) + 1 if categorical else data.max(axis=0)
```

## 3. HDBSCAN validity scores count noise points as one extra cluster — FIXED (branch `fix_bugs`, 2026-07-02)

**Where.** `src/model_eval.py` — `get_metrics`, lines 93–96, and `clust_size`,
lines 82–87.

**Problem.** `get_metrics` computes the denoised variables and then never uses
them — the intent is sitting in dead code:

```python
noise = pred_clust == -1
denoised_data = data[~noise]          # computed…
denoised_pred_clust = pred_clust[~noise]
# …then all four CVIs are scored on (data, pred_clust) with -1 included
```

All four CVIs therefore treat HDBSCAN's noise pool as one additional (often
large, incoherent) cluster. `min_clust_size` / `max_clust_size` also include
the noise group.

**Effect on results.** HDBSCAN's silhouette/CH/DB/Dunn scores are computed
under a different convention than k-means/AHC/LCA, yet `selected_pools` ranks
them against each other to build the "best / second-best distance-based
configuration" rows. HDBSCAN wins and losses in Table 2 (parent repo and this
one) are therefore not a like-for-like comparison; a degenerate
"1 cluster + noise" fit is even scoreable (the noise pool acts as the second
group). The `*` unbalanced flag can also fire on the noise group's size rather
than a real cluster's. Note `n_clust` for HDBSCAN already *excludes* noise, so
the reported cluster count and the scored partition are inconsistent with each
other. This is the most unambiguous bug of the set (the fix was clearly
intended), and it affects the parent's published comparison tables.

**Fix applied** (`get_metrics`, `clust_size`): all four CVIs and
`min_clust_size` / `max_clust_size` are now computed on the denoised
partition; `clust_size` returns NaN for an all-noise fit (the selection steps
already tolerate NaN). The stored `pred_clust` keeps the full labels, noise
included. Convention: CVIs describe clustered points only — this slightly
favours HDBSCAN the other way (standard practice). A side effect closes the
degenerate case: a "1 cluster + noise" fit now scores NaN (silhouette needs
≥ 2 groups) and drops out of the pools instead of being selectable.
**Note:** the same fix should be applied in `Scales-of-Nationalism` and its
results regenerated, or the two repos will disagree.

## 4. Benchmark counted clusters from a *re-fit*, not the scored fit — FIXED (2026-07-02)

**Where.** `benchmark.py` (`summarize_run`), rooted in `src/model_fit.py`
(`FlexibleKMeans` with `random_state=None`; StepMix built without
`random_state`).

**Problem.** The benchmark recovered each winning solution's labels by
re-fitting it. K-means and StepMix are unseeded random-restart optimizers, so
the refit could land on a different local optimum than the fit that earned the
recorded CVI score — worst exactly where the benchmark looks (low separation,
high noise, flat objective landscapes).

**Effect on results.** `n_clust_effective`, `n_singleton` and `ari` could
describe a different partition than the `score` in the same row, and
re-summarizing the same cached run gave different numbers.

**Fix applied.** The pipeline now persists each fit's partition
(`pred_clust` in `all_models`, added in `get_metrics`); the benchmark looks
labels up instead of re-fitting (`solution_labels`), with the old refit path
kept only as a fallback for pre-fix caches. Bootstrap rows drop the column to
bound memory. Verified: re-summarizing a payload is now bit-identical.
Deliberately **not** fixed by seeding: `FlexibleKMeans._single_fit` seeds the
*global* numpy RNG, so a constant seed would make every gap-bootstrap
iteration within a joblib worker draw identical random data and collapse the
gap statistic's variance.

---

## Moderate issues (not fixed)

**5. Mismatched moments in the gap tolerance.** `compute_gap` uses
`log(mean(rand))` for the gap but `std(log(rand))` for `s`
(`src/model_select.py:73–74`); Tibshirani uses the mean of logs for both. `s`
is therefore not the standard deviation of the quantity the gap is built from.
Small numerically (Jensen's inequality), but the "±s" tolerance is internally
inconsistent. Fix: use `np.mean(np.log(rand_ind))` in the gap.

**6. k = 1 is unselectable — the procedure cannot conclude "no structure".**
All four CVIs are undefined at k=1 (NaN), so in `get_gap` the k=1 vs k=2
comparison never fires and LCA always reports k ≥ 2; k-means/AHC start at k=2
by construction, and k = `max_clust` is also never selectable (the rule needs
k+1). Interpretation limit, inherent to gap-on-CVI: in the benchmark's
class_sep → 0 sweep, expect selected k to floor at 2 even on pure noise —
this is a property of the procedure, not evidence of structure. Fix if
desired: complement with a statistic defined at k=1 (e.g. the original
within-dispersion gap, or Hopkins as a pre-test — `src/hopkins.py` exists).

**7. `FlexibleKMeans` silently loses clusters.** An emptied cluster keeps a
center at the origin (`_compute_centers` skips it, leaving zeros — the global
mean on standardized data), and `n_clust` records the *requested* k even when
fewer clusters survive. Gap values "at k" can therefore describe partitions
with fewer than k clusters. The benchmark's singleton-excluded count
compensates at reporting time; the gap selection itself does not. Fix:
re-initialize empty clusters (standard practice) or record the realized
cluster count alongside the requested one.

**8. Nothing is seeded.** `bootstrap_gap`'s uniform draws, k-means inits and
StepMix restarts all use unseeded RNG: two identical pipeline invocations give
different `all_models` / `candidate_models`. The disk caches (app and
benchmark) mask this — but regenerating a cache is not guaranteed to reproduce
the run it replaces. Fix with care: seed *per task* (e.g. hash of
(model, params, n, iter)), never globally constant (see #4).

## Minor

- **AHC is fitted at n=1 for nothing** (`process_dataset.py:148`): all CVIs
  are NaN there and the bootstrap grid correctly starts at 2 — wasted fits.
- **CVIs use fixed metrics regardless of the fitted model's metric**
  (silhouette = manhattan, Dunn = cityblock, CH/DB = euclidean): chebyshev- or
  hamming-based models are judged in a geometry they didn't optimize.
  Presumably a deliberate common yardstick — worth one documenting sentence.
- **Hamming distance runs on standardized floats** (AHC/HDBSCAN `hamming`):
  only meaningful because Likert data has few distinct values per column;
  on continuous features it degenerates to "all distances ≈ 1".
- **`s` uses `np.std` with `ddof=0`**: negligible at `gap_iters=500`, visible
  at the benchmark's light setting of 50.
