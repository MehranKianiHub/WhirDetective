"""Convert WhirDetective PyTorch baseline models to deployable TFLite blobs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from whirdetective.ml import BaselineBearingCNN


@dataclass(frozen=True, slots=True)
class TfliteConversionArtifacts:
    """Outputs from one PyTorch-to-TFLite conversion run."""

    output_model_path: Path
    report_path: Path
    max_abs_diff: float
    mean_abs_diff: float


def build_parser() -> argparse.ArgumentParser:
    """Create parser for PyTorch-to-TFLite conversion."""
    parser = argparse.ArgumentParser(
        prog="whirdetective-tflite-convert",
        description="Convert WhirDetective BaselineBearingCNN state_dict to a TFLite model.",
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-package-dir",
        type=Path,
        required=True,
        help="Directory containing model_state_dict.pt and edgeos_model_manifest.json.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        required=True,
        help="Output .tflite model path.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Fixed sequence length baked into exported model input shape.",
    )
    parser.add_argument(
        "--parity-samples",
        type=int,
        default=8,
        help="Number of random samples for PyTorch-vs-TFLite parity check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for parity check samples.",
    )
    parser.add_argument(
        "--fail-on-parity-threshold",
        type=float,
        default=2e-2,
        help="Fail conversion if max absolute logit error exceeds this threshold.",
    )
    return parser


def run_conversion_from_args(args: argparse.Namespace) -> TfliteConversionArtifacts:
    """Run conversion and parity validation."""
    workspace_root = args.workspace_root.resolve()
    source_package_dir = _resolve_path(workspace_root, args.source_package_dir)
    output_model_path = _resolve_path(workspace_root, args.output_model)

    if args.sequence_length < 32:
        raise ValueError("sequence_length must be >= 32")
    if args.parity_samples <= 0:
        raise ValueError("parity_samples must be > 0")
    if args.fail_on_parity_threshold <= 0.0:
        raise ValueError("fail_on_parity_threshold must be > 0")

    state_path = source_package_dir / "model_state_dict.pt"
    manifest_path = source_package_dir / "edgeos_model_manifest.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing model_state_dict.pt: {state_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing edgeos_model_manifest.json: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_cfg = manifest.get("input", {})
    class_cfg = manifest.get("classification", {})
    input_channels = int(input_cfg.get("channels", 0))
    num_classes = int(class_cfg.get("num_classes", 0))
    if input_channels <= 0:
        raise ValueError("manifest input.channels must be > 0")
    if num_classes <= 1:
        raise ValueError("manifest classification.num_classes must be > 1")

    state_dict = torch.load(state_path, map_location="cpu")
    torch_model = BaselineBearingCNN(input_channels=input_channels, num_classes=num_classes)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    keras_model = _build_keras_equivalent(
        state_dict=state_dict,
        input_channels=input_channels,
        num_classes=num_classes,
        sequence_length=args.sequence_length,
    )

    import tensorflow as tf  # type: ignore[import-untyped]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = []
    tflite_bytes = converter.convert()

    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    output_model_path.write_bytes(tflite_bytes)

    parity = _evaluate_parity(
        torch_model=torch_model,
        tflite_model_path=output_model_path,
        input_channels=input_channels,
        sequence_length=args.sequence_length,
        num_classes=num_classes,
        samples=args.parity_samples,
        seed=args.seed,
    )
    max_abs_diff = float(parity["max_abs_diff"])
    mean_abs_diff = float(parity["mean_abs_diff"])

    if max_abs_diff > args.fail_on_parity_threshold:
        raise ValueError(
            "parity check failed: "
            f"max_abs_diff={max_abs_diff:.6f} > {args.fail_on_parity_threshold:.6f}"
        )

    report_path = output_model_path.with_suffix(".conversion_report.json")
    report_payload = {
        "source_package_dir": str(source_package_dir),
        "output_model": str(output_model_path),
        "sequence_length": args.sequence_length,
        "input_channels": input_channels,
        "num_classes": num_classes,
        "parity": parity,
        "threshold": {
            "max_abs_diff": args.fail_on_parity_threshold,
        },
        "evaluation": {
            "passed": max_abs_diff <= args.fail_on_parity_threshold,
        },
    }
    report_path.write_text(json.dumps(report_payload, indent=2, sort_keys=True), encoding="utf-8")

    return TfliteConversionArtifacts(
        output_model_path=output_model_path,
        report_path=report_path,
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for PyTorch-to-TFLite conversion."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        artifacts = run_conversion_from_args(args)
    except Exception as exc:
        print(f"[ERROR] tflite conversion failed: {exc}", file=sys.stderr)
        return 2

    print(f"output_model: {artifacts.output_model_path}")
    print(f"report: {artifacts.report_path}")
    print(f"max_abs_diff: {artifacts.max_abs_diff:.6f}")
    print(f"mean_abs_diff: {artifacts.mean_abs_diff:.6f}")
    return 0


def _resolve_path(base_dir: Path, value: Path) -> Path:
    if value.is_absolute():
        return value.resolve()
    return (base_dir / value).resolve()


def _build_keras_equivalent(
    *,
    state_dict: dict[str, torch.Tensor],
    input_channels: int,
    num_classes: int,
    sequence_length: int,
) -> object:
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(sequence_length, input_channels), name="input")

    x = tf.keras.layers.ZeroPadding1D(padding=7, name="pad1")(inputs)
    x = tf.keras.layers.Conv1D(32, kernel_size=15, strides=2, padding="valid", use_bias=True, name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn1")(x)
    x = tf.keras.layers.ReLU(name="relu1")(x)

    x = tf.keras.layers.ZeroPadding1D(padding=4, name="pad2")(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=9, strides=2, padding="valid", use_bias=True, name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn2")(x)
    x = tf.keras.layers.ReLU(name="relu2")(x)

    x = tf.keras.layers.ZeroPadding1D(padding=2, name="pad3")(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=5, strides=1, padding="valid", use_bias=True, name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9, name="bn3")(x)
    x = tf.keras.layers.ReLU(name="relu3")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="gap")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="fc1")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="fc2")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="whirdetective_baseline")

    conv1_kernel = state_dict["features.0.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    conv1_bias = state_dict["features.0.bias"].detach().cpu().numpy()
    model.get_layer("conv1").set_weights([conv1_kernel, conv1_bias])

    bn1_gamma = state_dict["features.1.weight"].detach().cpu().numpy()
    bn1_beta = state_dict["features.1.bias"].detach().cpu().numpy()
    bn1_mean = state_dict["features.1.running_mean"].detach().cpu().numpy()
    bn1_var = state_dict["features.1.running_var"].detach().cpu().numpy()
    model.get_layer("bn1").set_weights([bn1_gamma, bn1_beta, bn1_mean, bn1_var])

    conv2_kernel = state_dict["features.3.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    conv2_bias = state_dict["features.3.bias"].detach().cpu().numpy()
    model.get_layer("conv2").set_weights([conv2_kernel, conv2_bias])

    bn2_gamma = state_dict["features.4.weight"].detach().cpu().numpy()
    bn2_beta = state_dict["features.4.bias"].detach().cpu().numpy()
    bn2_mean = state_dict["features.4.running_mean"].detach().cpu().numpy()
    bn2_var = state_dict["features.4.running_var"].detach().cpu().numpy()
    model.get_layer("bn2").set_weights([bn2_gamma, bn2_beta, bn2_mean, bn2_var])

    conv3_kernel = state_dict["features.6.weight"].detach().cpu().numpy().transpose(2, 1, 0)
    conv3_bias = state_dict["features.6.bias"].detach().cpu().numpy()
    model.get_layer("conv3").set_weights([conv3_kernel, conv3_bias])

    bn3_gamma = state_dict["features.7.weight"].detach().cpu().numpy()
    bn3_beta = state_dict["features.7.bias"].detach().cpu().numpy()
    bn3_mean = state_dict["features.7.running_mean"].detach().cpu().numpy()
    bn3_var = state_dict["features.7.running_var"].detach().cpu().numpy()
    model.get_layer("bn3").set_weights([bn3_gamma, bn3_beta, bn3_mean, bn3_var])

    fc1_kernel = state_dict["classifier.1.weight"].detach().cpu().numpy().transpose(1, 0)
    fc1_bias = state_dict["classifier.1.bias"].detach().cpu().numpy()
    model.get_layer("fc1").set_weights([fc1_kernel, fc1_bias])

    fc2_kernel = state_dict["classifier.4.weight"].detach().cpu().numpy().transpose(1, 0)
    fc2_bias = state_dict["classifier.4.bias"].detach().cpu().numpy()
    model.get_layer("fc2").set_weights([fc2_kernel, fc2_bias])

    return model


def _evaluate_parity(
    *,
    torch_model: BaselineBearingCNN,
    tflite_model_path: Path,
    input_channels: int,
    sequence_length: int,
    num_classes: int,
    samples: int,
    seed: int,
) -> dict[str, float]:
    import tensorflow as tf

    rng = np.random.default_rng(seed)
    test_input = rng.normal(size=(samples, input_channels, sequence_length)).astype(np.float32)

    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(test_input)).detach().cpu().numpy()

    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    tflite_out = np.zeros((samples, num_classes), dtype=np.float32)
    for i in range(samples):
        sample = test_input[i : i + 1]
        sample_keras = np.transpose(sample, (0, 2, 1))
        interpreter.set_tensor(input_details["index"], sample_keras.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        tflite_out[i] = output[0]

    abs_diff = np.abs(torch_out - tflite_out)
    return {
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
