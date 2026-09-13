# totalsegmentator

Automatic anatomy segmentation for CT (`total`, 117 structures) and MR (`total_mr`), plus the
specialised upstream tasks. No prompts. The multi-label output comes with a label-name table so an
agent can pick `liver`, `spleen`, `vertebrae_L1`, ... directly.

```text
run_model("totalsegmentator", {"image": "ct.nii.gz"}, {"roi_subset": ["liver", "spleen"], "fast": true})
```

Weights are fetched on first use into `$OMM_HOME/weights/totalsegmentator`. Pre-fetch with
`open-med-mcp models download totalsegmentator` (runs the adapter's `download` task).
Install for the local runner: `pip install "open-med-mcp[totalsegmentator]"`.
