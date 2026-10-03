# Edge-case matrix

Each spec edge case maps to a test or is listed as "not covered" with a reason (CLAUDE.md §7).
Filled in per milestone.

| Edge case | Milestone | Test / status |
|---|---|---|
| Subject appears in two splits | M0 | `test_protocols.py::test_leakage_is_rejected`, `test_property_no_subject_in_two_splits` |
| Same file in two splits | M0 | `test_same_file_in_two_splits_is_rejected` |
| Split with a single class | M0 | `test_single_class_split_is_rejected` |
| Same subject ID in different datasets | M0 | `test_same_subject_id_in_different_datasets_is_not_leakage` |
| Invalid config value | M0 | `test_data.py::test_settings_bad_value_fails_fast` |
| Grayscale / RGBA / palette input | M1 | `test_load_image_modes`, `test_grayscale_and_rgb_give_same_shaped_features` |
| Different resolutions (resolution as shortcut) | M1 | `test_native_resolution_is_not_a_shortcut` |
| JPEG-artifact bias between classes | M1 | `test_flags_constructed_bias`, `test_unbiased_set_not_flagged`, `test_png_marks_quality_not_available` |
| Unsafe paths / duplicate files / checksum mismatch in licensed data | M0 | `test_manifest_data.py` |
| Face not detected -> logged centre-crop fallback | M1 | `test_crop_face_fallback_is_logged_and_flagged`, `test_face_fallback_is_logged_and_counted` |
| Crop margin stays inside the image | M1 | `test_crop_with_margin_inside_bounds_and_nonempty` (hypothesis) |
| Moire raises FFT high-band energy | M1 | `test_fft_high_band_ratio_larger_with_grating` |
| Empty / NaN / single-class scores | M3 | `test_degenerate_inputs_raise`, `test_scored_split_validates` |
| Ties at the threshold | M3 | `test_apcer_per_type_max_mean_and_tie_accepts`, `test_auc_ties_get_half_credit` |
| Threshold chosen on val only | M3 | `test_thresholds_chosen_from_val_only`, `test_target_scores_do_not_change_source_thresholds` |
| Bootstrap resamples subjects | M3 | `test_bootstrap_resamples_subjects_not_samples` |
| Small sample warning | M3 | `test_small_sample_warning` |
| Cross-domain threshold shift (diagnostic only) | M3 | `test_cross_dataset_uses_source_threshold_and_labels_shift_diagnostic` |
| Class imbalance / balanced exposure | M2 | `test_balanced_sampler_exposes_both_classes` |
| NaN loss / gradient | M2 | `train._run_epoch` fails before checkpoint advancement |
| Resume curve parity | M2 | `test_resume_matches_uninterrupted_curve` |
| JPEG augmentation | M2 | `RandomJPEGCompression` in the training transform |
| Crop-margin ablation | M2 | `liveness.train --face-margins ...` |
| Multiple seeds (>=3) | M2 | CLI rejects fewer than three distinct seeds |
| Synthetic results marked / absent shown as TBD | M3 | `test_smoke_rows_are_marked_and_small_sample_flagged`, `test_expected_but_absent_is_tbd` |
| Results JSON carries env + commit | M3 | `test_results_json_contains_env_and_commit` |
| Window shorter than needed | M5 | `test_window_shorter_than_needed_is_insufficient_evidence` |
| No-face frames | M5 | `test_no_face_frames_do_not_count_as_evidence`, `test_no_face_frames_age_out_old_evidence` |
| Score jitter / flapping | M5 | `test_hysteresis_prevents_flapping_on_noisy_scores_near_threshold` |
| NaN / inf score | M5 | `test_non_finite_or_out_of_range_score_rejected` |
| Corrupt input | M6 | `test_corrupt_image_is_invalid_input_with_model_version` |
| Tiny image / no face / multiple faces -> INSUFFICIENT_QUALITY, not SPOOF | M6 | `test_tiny_image_is_insufficient_quality_not_spoof`, `test_no_face_is_insufficient_quality`, `test_multiple_faces_is_insufficient_quality` |
| Batch / size limits | M6 | `test_batch_limit_exceeded_returns_413_with_model_version`, `test_oversize_image_returns_413_with_model_version` |
| Model version always returned | M6 | `test_validation_and_routing_errors_keep_contract` |
| Thresholds are config, not code | M6 | `test_threshold_change_via_settings_changes_decision` |
| Non-finite model output | M6 | `test_non_finite_logit_raises` |
| Uploads not persisted; no image bytes in logs | M6 | `test_uploads_are_not_persisted`, `test_logs_contain_no_image_bytes_or_scores` |
| Multiple faces | M4 | `test_multiple_faces_fail_after_tolerance`, `test_transient_second_face_is_tolerated` |
| Face lost mid-challenge -> timeout FAIL | M4 | `test_face_lost_mid_challenge_times_out`, `test_brief_face_loss_is_tolerated` |
| Out-of-order / duplicate timestamps | M4 | `test_duplicate_and_backwards_timestamps_fail` |
| Very long session (cap, bounded memory) | M4 | `test_session_hard_cap_duration_and_frames`, `test_long_session_memory_is_bounded` |
| Low-FPS camera (min FPS enforced and reported) | M4 | `test_low_fps_reported_not_silent`, `test_low_fps_is_insufficient_evidence` |
| Mirrored front camera (left/right swap) | M4 | `test_mirrored_swaps_left_and_right`, `test_mirrored_camera_left_turn`, `test_mirror_flag_wrong_means_wrong_direction` |
| Steps out of issued order | M4 | `test_steps_must_follow_issued_order` |
| Challenge replay / expiry / unknown nonce | M4 | `test_replayed_challenge_is_rejected`, `test_expired_at_start_and_mid_session`, `test_unknown_nonce`, `test_concurrent_consume_succeeds_exactly_once` |
| Slow vs fast blinks | M4 | `test_fast_and_slow_blinks_counted` |
| Glasses / occluded eyes -> not a fail | M4 | `test_low_confidence_reports_eyes_occluded`, `test_occluded_eyes_are_insufficient_not_fail` |
| EAR scale/translation invariance | M4 | `test_ear_invariant_to_scale_and_translation` (hypothesis) |
| Accessibility: alternative challenge sets | M4 | `generate_challenge(allowed_kinds=...)` in `test_active_challenge.py`; limits in ACTIVE_LIVENESS.md |

## Not covered (reason)
- Low light, backlight, sunglasses, masks, makeup, skin tones, OLED vs LCD, matte vs glossy,
  bent paper, cut-outs, fisheye, rolling shutter, time-of-day: need real data or real captures;
  unmeasured.
- Real Haar detection on real faces and MediaPipe landmark extraction: not exercised on real
  faces here (no camera / data).
