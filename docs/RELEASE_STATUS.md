# Release status and provenance

This source release is prepared for https://github.com/Jeon-minseong/Afterimage. It is not a final paper artifact or a finalized experimental protocol.

- Author code licensing and the final paper citation have not been selected; the repository is Jeon-minseong/Afterimage. No blanket MIT license has been invented.
- Vendored Ultralytics code and its license metadata remain under the original upstream terms. Check `afterimage/engine/vendor/ultralytics-8.3.221.dist-info/licenses` and upstream documentation before distribution.
- YOLO and RMBG-2.0 checkpoints are external. `configs/weights.json` specifies exact file hashes and upstream locations. Review upstream checkpoint/code terms and provide an authorized retrieval method for these exact revisions before a public release. No checkpoint license is granted here.
- Dataset video files and old fixed-input bundles are excluded from the source package. Confirm acquisition/redistribution instructions for Weizmann, UT-interaction and KISA before publication. The catalog alone is not the video dataset.
- GT is supplied separately in `gt/references.json`: 303 GT1 and 90 GT2 records with original file hashes, labeled model-assisted and not fully human-verified.
- Interpretation afterimage v3 prompts are preserved. General-frame guidance and the judge rubric remain candidates for discussion. The grading code does not make those choices scientifically final.
- The generation engine is copied from the portable C41 implementation. Model-path expressions now use `AF_MODELS_ROOT`, and a dormant server-path mask import fallback now resolves the local module. Algorithmic selection/composition code is preserved. `docs/PROVENANCE.json` identifies the original source hashes.
- The public-facing CLI and JSON schemas are new. Inputs are content-addressed; manifest order retains duplicates. Generated runtime logs may contain the executing machine's paths; `runs/` is excluded from the source archive.
- The current generation baseline is Windows/CUDA. Cross-platform support and a fresh-machine installation have not been verified. Use the declared environment rather than claiming environment-independent bit identity.
- No API keys, source videos, model weights, user-specific virtual environment, historical caches or paid responses are in the source archive. The README GIF is a small attributed research illustration; raw dataset videos and full fixed-input bundles are not included.

Before submission: finalize protocol and rubric; confirm dataset/weight access instructions and licensing; test a clean installation; run the intended API smoke test and full experiment; archive exact configs, source/version hashes, manifests, raw API responses, usage, scoring and the paper tables. Keep pilot/app/CLI results separate from confirmatory API results.
