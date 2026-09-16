from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from crosscorr_8param_common import (
    CrossCorrEmulator,
    NUM_COARSE_BINS,
    PARAMETER_NAMES,
    coarse_cross_spectrum,
    load_npy_sample,
    match_npy_pairs,
)


DATA_VERSION = "v100_random_two_noise_100"
#MAX_PAIRS = 1_0000  # this will be changed, number of samples
MAX_PAIRS = 600000  # this will be changed, number of samples
SEED = 42
HIDDEN_DIM = 256
BATCH_SIZE = 256
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GPU_INDEX = 0
PROGRESS_EVERY = 100

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
VERSION_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPTS_DIR / "processed_data_superimposed" / DATA_VERSION
OUTPUT_DIR = VERSION_DIR / f"outputs_crosscorr_emulator_{DATA_VERSION}"
SPLIT_MANIFEST_PATH = VERSION_DIR / f"outputs_dataset_{DATA_VERSION}" / "data_split_v1_a.json"
CACHE_TAG = "all" if MAX_PAIRS <= 0 else str(MAX_PAIRS)
CACHE_PATH = OUTPUT_DIR / f"crosscorr_training_cache_manifest_{CACHE_TAG}.npz"
MODEL_PATH = OUTPUT_DIR / "best_crosscorr_emulator_8params.pt"
RESUME_PATH = OUTPUT_DIR / f"crosscorr_emulator_training_resume_{CACHE_TAG}.pt"
PARTIAL_PARAMETER_PATH = OUTPUT_DIR / f"crosscorr_parameters_partial_{CACHE_TAG}.npy"
PARTIAL_SPECTRUM_PATH = OUTPUT_DIR / f"crosscorr_spectra_partial_{CACHE_TAG}.npy"
PARTIAL_STATE_PATH = OUTPUT_DIR / f"crosscorr_cache_partial_{CACHE_TAG}.json"


def choose_pairs_from_manifest(
    pairs: list[tuple[Path, Path]],
) -> tuple[list[tuple[Path, Path]], np.ndarray]:
    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"TimeFlow split manifest not found: {SPLIT_MANIFEST_PATH}\n"
            "Run train_flow_by_pop.py first or copy its data_split_v1_a.json to this path."
        )

    manifest = json.loads(SPLIT_MANIFEST_PATH.read_text(encoding="utf-8"))
    pair_map = {pair[0].name: pair for pair in pairs}
    split_names = ["train_records", "val_records", "test_records"]
    available: list[list[tuple[Path, Path]]] = []
    for split_name in split_names:
        matched = []
        for record in manifest.get(split_name, []):
            pair = pair_map.get(Path(record["h1"]).name)
            if pair is not None:
                matched.append(pair)
        if not matched:
            raise RuntimeError(f"No current NPY pairs matched manifest split {split_name}")
        available.append(matched)

    total_available = sum(len(items) for items in available)
    target_total = total_available if MAX_PAIRS <= 0 else min(MAX_PAIRS, total_available)
    raw_counts = np.asarray([len(items) for items in available], dtype=np.float64)
    raw_counts = raw_counts / raw_counts.sum() * target_total
    counts = np.floor(raw_counts).astype(int)
    for index in np.argsort(raw_counts - counts)[::-1][: target_total - int(counts.sum())]:
        counts[index] += 1

    rng = np.random.default_rng(SEED)
    selected_pairs: list[tuple[Path, Path]] = []
    split_codes = []
    for split_code, (items, count) in enumerate(zip(available, counts)):
        indices = np.sort(rng.choice(len(items), size=int(count), replace=False))
        selected_pairs.extend(items[index] for index in indices)
        split_codes.extend([split_code] * int(count))
    print(
        "Selected from TimeFlow manifest: "
        f"train={counts[0]}, val={counts[1]}, test={counts[2]}"
    )
    return selected_pairs, np.asarray(split_codes, dtype=np.int8)


def build_cache(
    pairs: list[tuple[Path, Path]],
    split_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if CACHE_PATH.exists():
        with np.load(CACHE_PATH, allow_pickle=False) as cache:
            print(f"Loaded cache: {CACHE_PATH}")
            return cache["parameters"], cache["spectra"], cache["freqs"], cache["split_codes"]

    partial_paths = (PARTIAL_PARAMETER_PATH, PARTIAL_SPECTRUM_PATH, PARTIAL_STATE_PATH)
    if any(path.exists() for path in partial_paths) and not all(path.exists() for path in partial_paths):
        raise RuntimeError(
            "Incomplete partial-cache files were found. Keep all three files or move all of them "
            f"aside before retrying: {partial_paths}"
        )

    start_index = 0
    common_freqs = None
    if all(path.exists() for path in partial_paths):
        state = json.loads(PARTIAL_STATE_PATH.read_text(encoding="utf-8"))
        if state.get("pair_count") != len(pairs) or state.get("data_version") != DATA_VERSION:
            raise RuntimeError(
                f"Partial cache does not match the current run: {PARTIAL_STATE_PATH}. "
                "Move the three partial-cache files aside before starting a different run."
            )
        parameters = np.lib.format.open_memmap(PARTIAL_PARAMETER_PATH, mode="r+")
        spectra = np.lib.format.open_memmap(PARTIAL_SPECTRUM_PATH, mode="r+")
        start_index = int(state["completed"])
        common_freqs = (
            None if state.get("freqs") is None
            else np.asarray(state["freqs"], dtype=np.float64)
        )
        print(f"Resuming spectrum extraction from {start_index}/{len(pairs)}")
    else:
        parameters = np.lib.format.open_memmap(
            PARTIAL_PARAMETER_PATH, mode="w+", dtype=np.float32,
            shape=(len(pairs), len(PARAMETER_NAMES)),
        )
        spectra = np.lib.format.open_memmap(
            PARTIAL_SPECTRUM_PATH, mode="w+", dtype=np.float64,
            shape=(len(pairs), NUM_COARSE_BINS),
        )
        PARTIAL_STATE_PATH.write_text(
            json.dumps(
                {
                    "completed": 0,
                    "pair_count": len(pairs),
                    "data_version": DATA_VERSION,
                    "freqs": None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    start = time.time()
    for zero_index in range(start_index, len(pairs)):
        h1_path, l1_path = pairs[zero_index]
        index = zero_index + 1
        h1, label_h1 = load_npy_sample(h1_path)
        l1, label_l1 = load_npy_sample(l1_path)
        if not np.array_equal(label_h1, label_l1):
            raise ValueError(f"H1/L1 label mismatch: {h1_path.name}, {l1_path.name}")
        freqs, spectrum = coarse_cross_spectrum(h1, l1)
        if common_freqs is None:
            common_freqs = freqs
        elif not np.array_equal(common_freqs, freqs):
            raise ValueError(f"Frequency grid mismatch at {h1_path.name}")
        parameters[zero_index] = label_h1
        spectra[zero_index] = spectrum
        if index % PROGRESS_EVERY == 0 or index == len(pairs):
            parameters.flush()
            spectra.flush()
            PARTIAL_STATE_PATH.write_text(
                json.dumps(
                    {
                        "completed": index,
                        "pair_count": len(pairs),
                        "data_version": DATA_VERSION,
                        "freqs": np.asarray(common_freqs).tolist(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            elapsed = time.time() - start
            print(f"Extracted {index}/{len(pairs)} pairs in this run in {elapsed / 60.0:.1f} min")

    parameter_array = np.asarray(parameters, dtype=np.float32)
    spectrum_array = np.asarray(spectra, dtype=np.float64)
    np.savez_compressed(
        CACHE_PATH,
        parameters=parameter_array,
        spectra=spectrum_array,
        freqs=np.asarray(common_freqs),
        split_codes=split_codes,
    )
    print(f"Saved cache: {CACHE_PATH}")
    return parameter_array, spectrum_array, np.asarray(common_freqs), split_codes


def safe_std(values: np.ndarray, axis: int = 0) -> np.ndarray:
    std = values.std(axis=axis)
    positive = std[std > 0]
    floor = float(np.median(positive) * 1e-6) if positive.size else 1.0
    return np.maximum(std, floor)


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = match_npy_pairs(DATA_ROOT / "H1_splits", DATA_ROOT / "L1_splits")
    selected_pairs, selected_split_codes = choose_pairs_from_manifest(all_pairs)
    print(f"Matched pairs: {len(all_pairs)}, selected: {len(selected_pairs)}")
    parameters, spectra, freqs, split_codes = build_cache(selected_pairs, selected_split_codes)
    train_idx = np.flatnonzero(split_codes == 0)
    val_idx = np.flatnonzero(split_codes == 1)
    test_idx = np.flatnonzero(split_codes == 2)
    if min(train_idx.size, val_idx.size, test_idx.size) == 0:
        raise RuntimeError("Train/validation/test split contains an empty subset")

    parameter_mean = parameters[train_idx].mean(axis=0)
    parameter_std = safe_std(parameters[train_idx])
    spectrum_mean = spectra[train_idx].mean(axis=0)
    spectrum_std = safe_std(spectra[train_idx])
    x = ((parameters - parameter_mean) / parameter_std).astype(np.float32)
    y = ((spectra - spectrum_mean) / spectrum_std).astype(np.float32)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y[train_idx])),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    device = torch.device(f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu")
    model = CrossCorrEmulator(hidden_dim=HIDDEN_DIM).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    start_epoch = 1

    if RESUME_PATH.exists():
        resume = torch.load(RESUME_PATH, map_location="cpu", weights_only=False)
        expected = {
            "data_version": DATA_VERSION,
            "max_pairs": MAX_PAIRS,
            "hidden_dim": HIDDEN_DIM,
            "train_size": int(train_idx.size),
            "val_size": int(val_idx.size),
            "test_size": int(test_idx.size),
        }
        mismatches = [
            f"{key}: checkpoint={resume.get(key)!r}, current={value!r}"
            for key, value in expected.items()
            if resume.get(key) != value
        ]
        if mismatches:
            raise RuntimeError(
                f"Resume checkpoint does not match the current run: {RESUME_PATH}\n"
                + "\n".join(mismatches)
                + "\nRename or remove this resume checkpoint before starting a different run."
            )
        model.load_state_dict(resume["model_state_dict"])
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        best_val = float(resume["best_val_mse"])
        best_state = resume["best_state_dict"]
        start_epoch = int(resume["epoch"]) + 1
        torch.set_rng_state(resume["torch_rng_state"])
        if torch.cuda.is_available() and resume.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all(resume["cuda_rng_state_all"])
        np.random.set_state(resume["numpy_rng_state"])
        print(f"Resuming training from epoch {start_epoch}/{EPOCHS}: {RESUME_PATH}")

    x_val = torch.from_numpy(x[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * batch_x.size(0)
            count += batch_x.size(0)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(x_val), y_val).item())
        train_loss = total / max(count, 1)
        print(f"Epoch {epoch:03d}/{EPOCHS}: train_mse={train_loss:.6g}, val_mse={val_loss:.6g}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_val_mse": best_val,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state(),
                "data_version": DATA_VERSION,
                "max_pairs": MAX_PAIRS,
                "hidden_dim": HIDDEN_DIM,
                "train_size": int(train_idx.size),
                "val_size": int(val_idx.size),
                "test_size": int(test_idx.size),
            },
            RESUME_PATH,
        )
        print(f"Saved resume checkpoint: {RESUME_PATH}")

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        val_prediction = model(x_val).cpu().numpy()
        test_prediction = model(torch.from_numpy(x[test_idx]).to(device)).cpu().numpy()

    residual = y[val_idx] - val_prediction
    residual_covariance = np.cov(residual, rowvar=False)
    diagonal_scale = float(np.median(np.diag(residual_covariance)))
    residual_covariance += np.eye(NUM_COARSE_BINS) * max(diagonal_scale * 1e-3, 1e-6)
    test_mse = float(np.mean((y[test_idx] - test_prediction) ** 2))
    test_sse = float(np.sum((y[test_idx] - test_prediction) ** 2))
    test_mean = y[test_idx].mean(axis=0, keepdims=True)
    test_sst = float(np.sum((y[test_idx] - test_mean) ** 2))
    test_r2 = 1.0 - test_sse / test_sst if test_sst > 0.0 else float("nan")

    checkpoint = {
        "model_state_dict": best_state,
        "parameter_names": PARAMETER_NAMES,
        "parameter_mean": parameter_mean.astype(np.float64),
        "parameter_std": parameter_std.astype(np.float64),
        "spectrum_mean": spectrum_mean.astype(np.float64),
        "spectrum_std": spectrum_std.astype(np.float64),
        "residual_covariance": residual_covariance.astype(np.float64),
        "freqs": freqs.astype(np.float64),
        "hidden_dim": HIDDEN_DIM,
        "num_coarse_bins": NUM_COARSE_BINS,
        "data_version": DATA_VERSION,
        "train_size": int(train_idx.size),
        "val_size": int(val_idx.size),
        "test_size": int(test_idx.size),
        "best_val_mse": best_val,
        "test_mse": test_mse,
        "test_r2": test_r2,
    }
    torch.save(checkpoint, MODEL_PATH)
    metrics = {
        key: checkpoint[key]
        for key in (
            "data_version",
            "train_size",
            "val_size",
            "test_size",
            "best_val_mse",
            "test_mse",
            "test_r2",
        )
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Test MSE: {test_mse:.6g}")
    print(f"Test R2: {test_r2:.6g}")
    print(f"Saved emulator: {MODEL_PATH}")
    print(f"Training resume checkpoint kept at: {RESUME_PATH}")


if __name__ == "__main__":
    main()
