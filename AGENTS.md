# Slack serving result delivery

- By default, also save this project's result deliverables locally under
  `/home/eyan/Slack_serving/results/`, grouped by experiment and job ID.
- Sync plots, review/acceptance reports, summary JSON, metric CSV files, and
  the corresponding run configuration when producing or updating results.
- Keep NSCC originals and their job/exit provenance. Prefer local result links
  in replies so the user can open the files directly.
- Preserve the existing presentation and unrelated local files. Result syncing
  does not authorize deleting or replacing unrelated material.
- For the experiment overview, follow slide 4 of the local reference deck:
  four horizontal panels, serif fonts, blue bandwidth, black KV, gray dashed
  queued requests, and black request-latency points. Keep labels restrained.
- State any time aggregation explicitly and retain the original 1 s data/plots.
  Mark actual load stages; do not copy T1/T2 positions from the reference.

## Meeting deck follow-up

- For this two-GPU/NVLink cooperation task, including its four-GPU extensions,
  append subsequent experiment results and analysis to the end of
  `C:\Users\24165\Vault\Work\组会\组会_Slack_Serving_20261030.pptx` by default.
- Edit this local Windows presentation from WSL at
  `/mnt/c/Users/24165/Vault/Work/组会/组会_Slack_Serving_20261030.pptx`;
  do not use this path on NSCC. Preserve all existing slides and any current
  unsaved user edits in the open presentation.
- Continue retaining raw NSCC results and local result copies. Include data
  scope, timing definitions and source job IDs in the appended slides/notes.
  Do not reintroduce experiment data that the user has deleted or excluded.
- Keep appended meeting slides concise and figure-led: one clear point per
  slide, few slides, and minimal visible text. Put detailed configurations,
  full metric tables and provenance in speaker notes rather than adding pages.

## Prefill timing follow-up

- After multi-turn two-GPU KV correctness is validated, compare prefill stage
  times against single-GPU native and the same custom kernel with all-local KV.
- Match batch, committed token histories, prefix/context lengths, new query
  lengths and chunk boundaries. Report initial and appended prefill separately.
  Per-request B1 numerical references are not B2 performance baselines.
- Include attention and necessary merges/reductions, GEMM and its split-K
  reductions, KV writes/layout, other operators/gaps, plus migration and page
  management outside model forward. Record whole-input-to-first-token elapsed
  time so migration overhead is not hidden.
- Disable numerical-oracle gathering, payload checks/poisoning and precision
  diagnostic branches for performance runs. Collect unprofiled primary timing
  and separate diagnostic traces; disclose profiler impact.
- Label an enlarged C+R single-GPU pool as an equal-work timing control; do not
  imply the original fixed-C pool can admit the same expanded workload.
- Retain the timing configuration in nvlink_prefill/prefill_timing_plan.json.
  Preserve the scope of previous decode results: the new dynamic-layout and
  residual-probability kernels need independent performance measurements.

## GPU experiment queue default

- By default, submit GPU experiments through `normal` with `walltime=02:00:01`, so two-GPU jobs route to `g2`. Use this for validation as well as formal runs unless the user specifies another queue or duration.
- NSCC g2 requires at least 02:00:01; a literal 02:00:00 does not meet its minimum. Record scheduler-side queue/walltime changes without modifying frozen launch sources.

## Performance repetition default

- Run each future performance configuration once (`plain_trials=1`) unless the user explicitly requests repetitions. Keep necessary correctness checks and warmup separate from the one measured run.

## Organized project paths (2026-11-01)

- Final cooperation entry: `final/cooperative_4gpu/`; two-GPU attention foundation: `final/attention_2gpu/`. Use `final/README.md` and release provenance before selecting a version.
- Historical code/configuration: `experiments/<stage>/<project>/`; results: `results/<stage>/<project>/<job_id>/`. Preserve these stage groups for new local result copies. NSCC historical result entries may be symlink indexes into the matching experiment's runs directory.
- Frozen historical sources retain original path strings. Optional legacy aliases can be restored with `archive/organization_20261101/restore_legacy_paths.py`; final runtime code is self-contained and does not need these aliases.
- The 2026-11-01 packaging adaptation was checked statically; its new entry paths have not had a new GPU execution. Historical measured jobs remain the performance/correctness evidence.
