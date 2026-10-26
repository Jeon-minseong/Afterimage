# Third-party materials and licensing

## Ultralytics

The preserved Ultralytics 8.3.221 source under `afterimage/engine/vendor/ultralytics` is distributed with its original copyright/license notices. Its AGPL-3.0 license is included at [the original license file](afterimage/engine/vendor/ultralytics-8.3.221.dist-info/licenses/LICENSE). The authors do not relicense this dependency.

## Model assets

YOLO and RMBG-2.0 weights are not included. Follow upstream terms at [Ultralytics](https://github.com/ultralytics/ultralytics) and [BRIA RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0). `configs/weights.json` pins required bytes. Downloading the latest weights does not guarantee these hashes.

## Data and example

Weizmann, UT-interaction and KISA video sets are external. The catalog identifies benchmark files by hashes, not by a grant of redistribution rights. Obtain authorized source videos separately. The README GIF is an attributed small illustration derived from the Weizmann run example; see [its provenance](assets/PROVENANCE.json). It is resized and annotated for explanation, and is not an experimental input or a new result. No dataset-wide redistribution license is implied.

## New code, GT and paper

A license for the authors' new code and reference annotations remains undecided. The source is available for inspection; do not assume MIT or Apache licensing. GT is model-assisted and has not received complete human verification. No paper acceptance, DOI or final citation is asserted.
