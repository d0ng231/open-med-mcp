# voxtell

[VoxTell](https://github.com/MIC-DKFZ/VoxTell) (Rokuss et al., CVPR 2026): free-text promptable
universal 3D medical image segmentation. One text prompt per structure; the output label `i`
corresponds to prompt `i`.

```text
segment("ct.nii.gz", model="voxtell", prompts=[{"type": "text", "text": "liver"}, {"type": "text", "text": "spleen"}])
run_model("voxtell", {"image": "ct.nii.gz"}, params={"text": ["left kidney", "right kidney", "aorta"]})
run_model("voxtell", {}, task="download")     # pre-fetch checkpoint + text encoder
```

Requirements: GPU, `pip install "open-med-mcp[voxtell]"` (PyTorch < 2.9), or the container. The
checkpoint comes from Hugging Face (`mrokuss/VoxTell`), the text encoder is `Qwen/Qwen3-Embedding-4B`;
both are cached under `$OMM_HOME/weights/voxtell/`.
