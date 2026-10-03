# Data

**No dataset is committed, redistributed, or auto-downloaded.** Datasets live under
`LIVENESS_DATA_DIR` (default `./data`, gitignored).

Access information below was rechecked against the linked official pages on **2026-10-03**.
The agreement you actually sign is authoritative. Do not use third-party mirrors: their
redistribution rights are unclear.

| Dataset | Access | Key terms | Status |
|---|---|---|---|
| [Replay-Attack (Idiap)](https://www.idiap.ch/en/dataset/replay-attack) | Use Idiap's “Get data” flow; if it resolves to the wrong record, contact Idiap through the page's Contact link | Follow Idiap's current grant; do not redistribute biometric videos | **Owner action required** |
| [CASIA-FASD](http://www.cbsr.ia.ac.cn/english/FASDB_Agreement/Agreement.pdf) | Sign the CASIA Face Antispoofing Database Release Agreement and submit it through CASIA's access site; approval supplies an account | Scientific research by educational institutes; no redistribution, modification, or commercial use; publication of subject images is restricted to the IDs listed in the agreement; cite Zhang et al. (ICB 2012) | **Owner/institution action required**; the historical official endpoint is intermittently unavailable |
| [OULU-NPU](https://sites.google.com/site/oulunpudatabase/) | Download and sign the EULA, then email the data controller named in it | Signatory must hold a permanent institutional position; institutional email required; Gmail/Yahoo/Hotmail requests are rejected | **Permanent staff signatory required** |
| [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof) | Download only after accepting the published dataset agreement | Non-commercial research only; no commercial exploitation or redistribution; internal copies only at one site | Optional; not used in required cross-dataset pair |
| SiW | Separate owner request/license | Research use; verify the supplied agreement | Optional; not used in required cross-dataset pair |

Faculty or institutional signature is often required on these agreements.

## Licensing consequence

Code is MIT. Models trained on non-commercial data are treated as non-commercial:
**trained weights are not published.** This is stated in the README and MODEL_CARD.md.

## Synthetic fixture (CI)

`liveness.data.synthetic.generate_synthetic_dataset` builds cartoon faces with generic
degradations (blur, colour cast, interference pattern) under a temp dir. It only exercises the
pipeline. **No metric from it may appear in RESULTS.md.**

## Normalized layout and checksums

The code never guesses a proprietary archive layout. After access is granted, extract each
dataset locally and create `<dataset-root>/manifest.csv` with these columns:

```csv
relative_path,subject_id,label,attack_type,split,sha256
frames/train/client001/live_0001.jpg,client001,bona_fide,none,train,<optional sha256>
frames/devel/client021/print_0001.jpg,client021,attack,print,val,<optional sha256>
frames/test/client041/replay_0001.jpg,client041,attack,replay,test,<optional sha256>
```

- `relative_path` must stay under the dataset root; absolute and `..` paths are rejected.
- `label` accepts `bona_fide`/`live`/`real` or `attack`/`spoof`/`fake`.
- `attack_type` is `none`, `print`, or `replay` (`photo` and `video` aliases are accepted).
- `split` is `train`, `val`/`devel`, or `test` and must preserve the official subject split.
- `sha256` is optional per row, but when present it is verified before loading.

`load_manifest_dataset()` rejects missing files, duplicate paths, checksum mismatches,
single-class splits, and subject leakage. Keep raw data, manifests containing local paths, and
generated failure grids outside Git. Dataset-specific attack subtypes should be normalized as:
hard-copy/warped/cut photos -> `print`; phone/tablet/display video -> `replay`; bona fide ->
`none`. Record any ambiguous mapping in a local README before training.

For video datasets, extract frames at a fixed, documented rate and keep all frames from one
video under its subject's official split. The current CNN loader consumes images; temporal
fusion operates on their per-frame scores.
