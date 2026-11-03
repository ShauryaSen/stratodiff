# Working rules for this repo

Read `SPEC.md` (what we are building, milestones) and `DECISIONS.md` (every choice and its basis) before
changing anything. The model card shared with Sarah mirrors these:
https://claude.ai/artifact/2F8YYtQGSb6PvJBQE2XGx6

## Rules (verbatim, from Shaurya)

- Never make an undisclosed design choice. Cheap, reversible choices: decide and list them under "Assumptions I made." Expensive or result-affecting choices: ask me.
- Every claim and design choice is CITED, MEASURED, or ASSUMED. Nothing arbitrary goes unlabeled. Verify citations; don't cite from memory.
- Experiments: classify knobs as scientific, nuisance, or fixed (Google Tuning Playbook). Ablate one change at a time. Report multiple seeds with confidence intervals, not single runs.
- Before any training run, write down the hypothesis and expected result.
- Keep "verified" separate from "I believe" when interpreting results.

## How to work with Shaurya

- Ask design questions in chat, a few at a time, in dependency order. For each: say in one or two plain
  sentences what the decision really is, then the recommendation, the reasoning and the main alternative.
- A new decision goes into `DECISIONS.md` with its tag in the same change that relies on it.

## Conventions

- `uv` for everything: `uv run pytest -q`, `uv run stratodiff ...`.
- Settings are frozen dataclasses in `stratodiff/config.py` with a plain comment per field; a YAML file in
  `configs/` overrides them, and an unknown key is an error, never ignored.
- No paths in code. Data and run locations come from the cluster profile in `configs/clusters/`.
- Every run writes its resolved config, git commit, dirty state and seed to its run directory
  (`stratodiff/runlog.py`). A run from a dirty tree is refused unless `--allow-dirty` is passed, and then
  the diff is saved beside it.
- Tests run on a laptop CPU in seconds. Anything needing the network or a GPU is marked
  (`@pytest.mark.network`, `@pytest.mark.gpu`) and skipped by default.
- Unimplemented pieces raise `NotImplementedError` naming the milestone that builds them. Never fake output.
