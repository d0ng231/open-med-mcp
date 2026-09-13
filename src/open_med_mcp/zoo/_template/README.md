# Adding a model (adapter template)

1. Copy this folder to `src/open_med_mcp/zoo/<adapter>/` (bundled) or to any directory listed in
   `OMM_MODEL_DIRS` (user models, no fork needed).
2. Edit the manifest (`<name>.yaml`): tasks, params, weights, container image, local requirements.
3. Implement `run.py`: read `request.json` through `omm_job.Job`, write outputs, call `job.finish()`.
4. `open-med-mcp models check <name>` shows which backends can run it; `open-med-mcp run <name> --image x.nii.gz`
   executes it end-to-end; `open-med-mcp models build <name>` builds the container.

The contract is documented in `docs/models.md` and `zoo/_sdk/omm_job.py`.
