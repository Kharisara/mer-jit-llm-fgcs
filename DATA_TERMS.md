# Dataset Terms and Primary-Input Reconstruction

ReplayBench-PG does not redistribute source content from the MELD dataset.

Users who wish to reconstruct the primary ReplayBench-PG workload must obtain the MELD annotation files from the original MELD provider. The ReplayBench-PG release contains only an ordered identifier manifest and deterministic reconstruction code.

The required provider files are:

* `train_sent_emo.csv`
* `dev_sent_emo.csv`
* `test_sent_emo.csv`

Run:

`python scripts/prepare_primary_replay_v260.py --meld-root <PATH_TO_MELD>`

The reconstruction script creates split-qualified record identifiers from `Dialogue_ID` and `Utterance_ID`, selects the 11,351 records used in the study according to `data_provenance/meld_v260_record_ids.csv`, derives the frozen binary diagnostic action from the MELD `Emotion` annotation, and emits only:

`source_record_id,diagnostic_action`

The positive diagnostic-action classes are:

`anger`, `disgust`, `fear`, and `sadness`.

The release does not include MELD utterance text, emotion annotations, audio, video, or source-derived embeddings.

For ReplayBench-PG v2.6.0, successful reconstruction must satisfy:

* selected rows: 11,351
* unique source-record identifiers: 11,351
* diagnostic-positive records: 2,609
* ordered selection-manifest SHA-256: `3ecd5826976393d0c44ae3d59d5d7e7a8b8b6ccd416571dd96b564504f261646`
* canonical reconstructed replay-table SHA-256: `2b46fd5e6887305d2eb43b1f4383e23ed3c1062a4826e1b7a428a10bd8f84f44`

The generated primary replay table is intentionally not redistributed. The recorded SHA-256 allows an independently reconstructed copy to be checked against the exact input used for the reported v2.6.0 experiments.

MetroPT-3 likewise remains available from its original provider and is not redistributed by ReplayBench-PG.
