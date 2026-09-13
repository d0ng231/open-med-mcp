# Contributing

Thanks for helping make medical image analysis usable by agents. The project is deliberately
modular; most contributions fall into one of three buckets.

## 1. Add a model (adapter)

1. Copy `src/open_med_mcp/zoo/_template/` to `src/open_med_mcp/zoo/<adapter>/`.
2. Write the manifest (`<name>.yaml`): tasks, params (with defaults and descriptions - agents read
   them), weights (URL + sha256), container image, local requirements.
3. Implement `run.py` against the job contract using `omm_job` (see `docs/models.md`). Keep it
   standalone: it runs inside a container without `open_med_mcp` installed.
4. Add a `Dockerfile` (build context = repo root) and check that
   `open-med-mcp models def <name>` produces a sane Apptainer definition.
5. Add the adapter to the matrix in `.github/workflows/containers.yml`.
6. Test locally: `open-med-mcp run <name> --image sample.nii.gz` and, if the model is light enough,
   add a test under `tests/`.

Models that need a GPU are exercised manually (see `docs/hpc.md`); CI runs CPU-only tests.

## 2. Add or improve a guideline

Guidelines live in `src/open_med_mcp/guidelines/presets/*.md` (YAML front matter + Markdown).
Write for an agent: concrete tool calls, decision tables, explicit QC criteria, what to report.
Keep them modality/task specific and short enough to be followed in one pass.

## 3. Viewer renderers

Implement a class with `render(image, masks, spec) -> RenderResult` and register it with
`@register_renderer("name")` or through the `open_med_mcp.renderers` entry-point group in your
own package. `ViewSpec` is the single input; see `docs/viewer.md`.

## Development

```bash
git clone https://github.com/d0ng231/open-med-mcp && cd open-med-mcp
uv venv && uv pip install -e ".[dev]"
.venv/bin/ruff check src tests && .venv/bin/python -m pytest -q
```

Please keep tool signatures stable (agents depend on them), document every parameter with a
`Field(description=...)`, and add a CHANGELOG entry.
