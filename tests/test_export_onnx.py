import copy
import dataclasses
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import timm
import torch
from torch import nn

from liveness import export_onnx
from liveness.export_onnx import (
    ExportError,
    ParityReport,
    export_from_checkpoint,
    export_to_onnx,
    load_checkpoint,
    verify_parity,
)
from liveness.infer import OnnxScorer

SIZE = 64
ARCH = "mobilenetv3_small_050"
PARITY_ATOL = 1e-4


def make_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return timm.create_model(ARCH, pretrained=False, num_classes=1).eval()


@pytest.fixture(scope="module")
def model() -> nn.Module:
    return make_model()


@pytest.fixture(scope="module")
def onnx_path(model: nn.Module, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return export_to_onnx(model, tmp_path_factory.mktemp("onnx") / "m.onnx", SIZE, "unit-v1")


def test_export_is_valid_onnx_with_contract_names_and_dynamic_batch(onnx_path: Path) -> None:
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto, full_check=True)
    (inp,), (out,) = proto.graph.input, proto.graph.output
    assert (inp.name, out.name) == ("input", "logit")
    in_dims = inp.type.tensor_type.shape.dim
    out_dims = out.type.tensor_type.shape.dim
    assert in_dims[0].dim_param and not in_dims[0].dim_value
    assert [d.dim_value for d in in_dims[1:]] == [3, SIZE, SIZE]
    assert out_dims[0].dim_param and out_dims[1].dim_value == 1
    assert inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert proto.opset_import[0].version == 17


def test_metadata_has_model_version_and_export_info(onnx_path: Path) -> None:
    meta = ort.InferenceSession(str(onnx_path)).get_modelmeta().custom_metadata_map
    assert meta["model_version"] == "unit-v1"
    assert meta["export_torch_version"] == str(torch.__version__)
    assert meta["export_onnx_version"] == onnx.__version__
    assert meta["export_image_size"] == f"{SIZE}x{SIZE}"
    assert meta["export_opset"] == "17"
    assert "dynamo=False" in meta["export_exporter"]


def test_pytorch_onnx_parity_within_tight_tolerance(model: nn.Module, onnx_path: Path) -> None:
    report = verify_parity(model, onnx_path, SIZE, n=8, atol=PARITY_ATOL, rtol=1e-3, seed=3)
    assert report.passed
    assert report.n_inputs == 8 + 8  # random images + synthetic-fixture images
    assert report.max_abs_diff < PARITY_ATOL
    assert report.max_abs_diff_score < PARITY_ATOL


def test_parity_detects_a_different_model(onnx_path: Path) -> None:
    other = make_model(seed=1)
    report = verify_parity(other, onnx_path, SIZE, n=4, use_fixture=False)
    assert isinstance(report, ParityReport) and not report.passed
    assert report.max_abs_diff > PARITY_ATOL


def test_parity_needs_inputs(model: nn.Module, onnx_path: Path) -> None:
    with pytest.raises(ExportError, match="no inputs"):
        verify_parity(model, onnx_path, SIZE, n=0, use_fixture=False)


def test_exported_model_loads_in_onnx_scorer(onnx_path: Path) -> None:
    scorer = OnnxScorer(onnx_path)
    assert scorer.model_version == "unit-v1"
    assert scorer.input_size == (SIZE, SIZE)
    rng = np.random.default_rng(0)
    for n in (1, 5):
        chips = [rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8) for _ in range(n)]
        scores = scorer.score_chips(chips)
        assert len(scores) == n
        assert np.all(np.isfinite(scores)) and all(0.0 <= s <= 1.0 for s in scores)


def test_dynamic_batch_gives_same_rows_for_any_batch_size(onnx_path: Path) -> None:
    sess = ort.InferenceSession(str(onnx_path))
    x = np.random.default_rng(1).standard_normal((5, 3, SIZE, SIZE)).astype(np.float32)
    full = sess.run(["logit"], {"input": x})[0]
    single = np.concatenate([sess.run(["logit"], {"input": x[i : i + 1]})[0] for i in range(5)])
    assert full.shape == (5, 1)
    assert np.allclose(full, single, atol=PARITY_ATOL)


def test_train_mode_model_is_exported_in_eval_mode_and_mode_restored(tmp_path: Path) -> None:
    m = make_model(2)
    m.train()
    for _ in range(3):  # make BatchNorm running stats differ from batch stats
        m(torch.randn(8, 3, SIZE, SIZE) * 2 + 1)
    probe = torch.randn(4, 3, SIZE, SIZE)
    path = export_to_onnx(m, tmp_path / "t.onnx", SIZE, "train-mode")
    assert m.training
    onnx_out = ort.InferenceSession(str(path)).run(["logit"], {"input": probe.numpy()})[0]
    train_copy = copy.deepcopy(m)
    m.eval()
    with torch.no_grad():
        eval_out = m(probe).numpy()
        train_out = train_copy(probe).numpy()
    assert np.allclose(onnx_out, eval_out, atol=PARITY_ATOL)
    assert not np.allclose(onnx_out, train_out, atol=1e-3)
    m.train()
    assert verify_parity(m, path, SIZE).passed and m.training


def test_export_is_deterministic(model: nn.Module, tmp_path: Path) -> None:
    a = export_to_onnx(model, tmp_path / "a.onnx", SIZE, "v").read_bytes()
    b = export_to_onnx(model, tmp_path / "b.onnx", SIZE, "v").read_bytes()
    assert a == b


def test_non_square_image_size(model: nn.Module, tmp_path: Path) -> None:
    path = export_to_onnx(model, tmp_path / "r.onnx", (64, 48), "rect")
    assert OnnxScorer(path).input_size == (64, 48)
    assert verify_parity(model, path, (64, 48), n=2, use_fixture=False).passed


def test_invalid_export_arguments(model: nn.Module, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="model_version"):
        export_to_onnx(model, tmp_path / "x.onnx", SIZE, "")
    with pytest.raises(ExportError, match="image_size"):
        export_to_onnx(model, tmp_path / "x.onnx", 0, "v")
    two_class = timm.create_model(ARCH, pretrained=False, num_classes=2)
    with pytest.raises(ExportError, match=r"\[N,1\]"):
        export_to_onnx(two_class, tmp_path / "x.onnx", SIZE, "v")
    assert not (tmp_path / "x.onnx").exists()


def save_ckpt(path: Path, model: nn.Module, **extra: object) -> Path:
    torch.save({"model": model.state_dict(), **extra}, path)
    return path


def test_export_from_checkpoint_round_trip(model: nn.Module, tmp_path: Path) -> None:
    ckpt = save_ckpt(tmp_path / "c.pt", model, epoch=3)
    out = export_from_checkpoint(ckpt, tmp_path / "o.onnx", SIZE, "from-ckpt")
    assert verify_parity(model, out, SIZE).passed
    assert OnnxScorer(out).model_version == "from-ckpt"
    state, arch = load_checkpoint(ckpt)
    assert arch is None and set(state) == set(model.state_dict())


def test_checkpoint_errors_are_clear(model: nn.Module, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="not found"):
        load_checkpoint(tmp_path / "missing.pt")
    torch.save(model.state_dict(), tmp_path / "bare.pt")
    with pytest.raises(ExportError, match="'model' state_dict"):
        load_checkpoint(tmp_path / "bare.pt")
    (tmp_path / "junk.pt").write_bytes(b"not a checkpoint")
    with pytest.raises(ExportError, match="could not read"):
        load_checkpoint(tmp_path / "junk.pt")
    ckpt = save_ckpt(tmp_path / "c.pt", model)
    with pytest.raises(ExportError, match="does not match"):
        export_from_checkpoint(ckpt, tmp_path / "o.onnx", SIZE, "v", arch="mobilenetv3_small_075")
    with pytest.raises(ExportError, match="cannot build"):
        export_from_checkpoint(ckpt, tmp_path / "o.onnx", SIZE, "v", arch="no_such_arch")


def test_checkpoint_arch_key_is_used(model: nn.Module, tmp_path: Path) -> None:
    ckpt = save_ckpt(tmp_path / "c.pt", model, arch=ARCH)
    assert load_checkpoint(ckpt)[1] == ARCH


def test_cli_exports_and_verifies(
    model: nn.Module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ckpt = save_ckpt(tmp_path / "c.pt", model)
    out = tmp_path / "cli.onnx"
    argv = ["--checkpoint", str(ckpt), "--out", str(out), "--image-size", str(SIZE)]
    argv += ["--model-version", "cli-v1"]
    assert export_onnx.main(argv) == 0
    assert "passed=True" in capsys.readouterr().out
    assert OnnxScorer(out).model_version == "cli-v1"
    assert export_onnx.main([*argv, "--no-verify"]) == 0
    assert export_onnx.main(["--checkpoint", str(tmp_path / "no.pt"), *argv[2:]]) == 2


def test_cli_returns_1_when_parity_fails(
    model: nn.Module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ckpt = save_ckpt(tmp_path / "c.pt", model)
    failed = ParityReport(1.0, 1.0, 1.0, 1, 1e-4, 1e-3, False)
    monkeypatch.setattr(export_onnx, "verify_parity", lambda *a, **k: failed)
    argv = ["--checkpoint", str(ckpt), "--out", str(tmp_path / "f.onnx"), "--image-size", "64"]
    assert export_onnx.main([*argv, "--model-version", "v"]) == 1
    assert dataclasses.asdict(failed)["passed"] is False
