# vlm

Vision-language question answering / description with Hugging Face image-text-to-text models.
Default `google/medgemma-4b-it` (gated: accept the licence, `hf auth login`); tested with the open
`Qwen/Qwen2.5-VL-3B-Instruct`.

```text
ask_vlm("cxr.png", "Is there a pleural effusion? Answer with the side and your confidence.")
ask_vlm("ct.nii.gz", "Describe this axial slice.", slice=97, window="soft-tissue", model_id="Qwen/Qwen2.5-VL-3B-Instruct")
```

The tool returns the answer and the exact image the model saw, so the agent can check the text.
