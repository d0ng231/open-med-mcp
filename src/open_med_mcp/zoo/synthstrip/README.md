# synthstrip (wrapped image)

This adapter has no `run.py`: it drives the official `freesurfer/synthstrip` container through the
manifest's `container.command` template (see `docs/models.md`, "Wrapped images"). The server stages
the input, runs `mri_synthstrip` inside the image with the job directory mounted at `/job`, and
turns the produced files into the standard response.

```text
run_model("synthstrip", {"image": "t1.nii.gz"}, params={"no_csf": true})
```
