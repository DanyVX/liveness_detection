# Edge-case matrix

Each spec edge case maps to a test or is listed as "not covered" with a reason (CLAUDE.md §7).
Filled in per milestone.

| Edge case | Milestone | Test / status |
|---|---|---|
| Subject appears in two splits | M0 | `tests/test_protocols.py::test_leakage_is_rejected`, `test_property_no_subject_in_two_splits` |
| Same file in two splits | M0 | `test_same_file_in_two_splits_is_rejected` |
| Split with a single class | M0 | `test_single_class_split_is_rejected` |
| Same subject ID in different datasets | M0 | `test_same_subject_id_in_different_datasets_is_not_leakage` |
| Invalid config value | M0 | `tests/test_data.py::test_settings_bad_value_fails_fast` |
| Grayscale input, mixed resolutions, JPEG-artifact bias | M1 | not yet implemented |
| Class imbalance, crop margin, NaN loss, AMP, resume | M2 | not yet implemented |
| Everything else in the spec matrix | M2-M6 | not yet implemented |
