# CLIP refinement pipeline

## Behavior

1. Read the merged Step3 inputs and scores using their existing filenames.
2. Select only samples below the configured CLIP threshold.
3. Before every batch round, keep the target fixed and deterministically sample
   a new father image from another class in the same parent category.
4. Rebuild the target description and context scene from the newly sampled
   father, then render with a new deterministic seed.
5. Score the complete candidate batch, atomically replace passing images, and keep failed candidates under logs.
6. After all GPU shards finish generation and batched CLIP scoring, repeat for
   failures with another father until every sample passes, the round limit is
   reached, or the explicit API cost ceiling stops the run.
7. Keep production image paths stable, so the augmented dataset manifest stays valid.

Each refinement round creates exactly one task shard per GPU. A worker generates its whole shard, unloads Flux, and then scores that whole shard in CLIP mini-batches. Only after every GPU result is collected are rejected descriptions grouped for Step2 re-extension.

Klein's local Diffusers implementation repeats one reference-image set across a prompt batch. Since every sample has different target/father references, model batch size remains one. A batch round is partitioned across seven GPU workers, and CLIP scores are computed in batches after generation.

## Boundaries

- configuration.py: typed YAML settings.
- records.py: typed task, attempt, and result records.
- storage.py: paths, checkpoints, reports, backups, and atomic replacement.
- resampling.py: deterministic father-only sampling from the completed Step2
  target pool.
- api.py: rebuilding descriptions and context scenes for the sampled father,
  with target/father inputs logged per sample.
- text_backend.py/model.text_llm: concurrent DeepSeek calls with thread-local
  sample attribution and a process-wide locked cost ledger.
- worker.py: one GPU candidate generation and CLIP scoring.
- pipeline.py: batch-round orchestration.
- cli.py: command parsing only.

Business logic receives dataclasses. Model initialization remains in the worker/backend. All auxiliary artifacts live in logs/clip_refinement/.

## Resume and cost

A checkpoint records every prompt, seed, candidate, and score. Passed samples are skipped on resume. The original image and metadata are backed up before the first accepted replacement. DeepSeek uses a ledger separate from Step2. The plan command reports sample counts and estimated cost without loading Flux, CLIP, or calling the API.

Each description has a finite retry limit. If all attempts return the same valid
target description, refinement keeps that description and still applies the new
father and regenerated scene; invalid outputs exhaust the attempt limit with an
explicit error instead of consuming the API budget indefinitely.

## Round contract

The initial run is Step2 followed by Step3 and CLIP evaluation. A failed sample
then enters refinement round 1. Every refinement round performs this exact
pipeline:

1. Keep the target image identity and return to its original Step1 description.
2. Draw a new father from another label in the same coarse parent pool.
3. Run Step2 with the original Step1 description plus only the newly drawn
   father's description and scene.
4. Run Step3 with the new Step2 description, a new deterministic seed, and the
   new father.
5. Score the candidate with the target's fine-grained semantic label.
6. Accept a passing candidate; send only failures to the next round.

A failed Step2 description is recorded for audit but is never used as the next
round's target anchor. The coarse parent category is used only to construct the
father pool. It must never replace the fine-grained class sense in a target
description or CLIP text. The configured maximum round count is a strict
inclusive ceiling, including after resume; round 10 cannot enter round 11.
