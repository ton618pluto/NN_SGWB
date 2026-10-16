from __future__ import annotations

from pathlib import Path


# ---------------------------- User controls ----------------------------
NUM_SBC_SAMPLES = 1_000
GPU_INDEX = 1
USE_MODEL2 = False

FRAME_DURATION_SECONDS = 2_048
SAMPLING_FREQUENCY_HZ = 512
MINIMUM_FREQUENCY_HZ = 10.0
REFERENCE_FREQUENCY_HZ = 25.0
Z_MAX = 10.0
LOCAL_RATE_GPC3_YR = 320.0

# Match the v100 training simulator: individually resolvable events are removed.
NETWORK_SNR_THRESHOLD = 12.0
NOISE_SCALE = 1.0 / 100.0

# The training population files require at least one event. For strict IID
# evaluation of that conditional simulator, retry the event realization while
# keeping theta fixed. Set False only to test the unconditional simulator,
# which can produce pure-noise frames and is not represented in the training set.
CONDITION_ON_NONEMPTY_POPULATION = True
MAX_POPULATION_ATTEMPTS = 200

# Reproducible, disjoint random streams. Training signal seeds occupy roughly
# 0..299999 and training-noise seeds start near 42, so these ranges do not overlap.
HYPERPARAMETER_SEED = 20_260_925
SIGNAL_SEED_BASE = 1_000_000
NOISE_SEED_BASE = 3_000_000
POSTERIOR_SEED_BASE = 4_000_000

NUM_POSTERIOR_SAMPLES = 999
NUM_RANK_BINS = 20
UNIFORM_ENVELOPE_PROBABILITY = 0.99
PROGRESS_SAVE_EVERY = 1
RESUME_PROGRESS = True
RESET_PROGRESS = False
OUTPUT_DPI = 800

# The eight inferred parameters use independent uniform priors.
INFERRED_PARAMETER_RANGES = {
    "zp": (0.85, 2.85),
    "alpha_z": (1.0, 3.0),
    "beta_z": (1.0, 5.0),
    "alpha_m": (-3.0, 7.0),
    "delta_m": (0.0, 10.0),
    "mu_m": (25.0, 50.0),
    "sigma_m": (0.01, 5.0),
    "beta_q": (-5.0, 5.0),
}

FIXED_PARAMETERS = {
    "m_max": 100.0,
    "m_min": 2.0,
    "lambda_peak": 0.04,
}

FULL_LABEL_NAMES = (
    "zp",
    "alpha_z",
    "beta_z",
    "alpha_m",
    "m_max",
    "delta_m",
    "m_min",
    "lambda_peak",
    "mu_m",
    "sigma_m",
    "beta_q",
)

THIS_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = THIS_DIR.parent
TIMEFLOW_DIR = SCRIPTS_DIR / "timeflow_v1_a"
OUTPUT_ROOT = THIS_DIR / "outputs"
HYPERPARAMETER_PATH = OUTPUT_ROOT / "iid_hyperparameters_v100.npz"
DATASET_ROOT = OUTPUT_ROOT / "processed_iid_sbc_v100"
H1_DIR = DATASET_ROOT / "H1_splits"
L1_DIR = DATASET_ROOT / "L1_splits"
GENERATION_LOG_PATH = OUTPUT_ROOT / "generation_log.csv"
O4_ASD_PATH = SCRIPTS_DIR / "O4_mock.txt"

MODEL_DATA_VERSION = "v100_random_two_noise_100"
MODEL_OUTPUT_DIR = TIMEFLOW_DIR / f"outputs_dataset_{MODEL_DATA_VERSION}"
MODEL_CHECKPOINT_PATH = MODEL_OUTPUT_DIR / (
    "best_flow_v1_a_model2.pt" if USE_MODEL2 else "best_flow_v1_a.pt"
)
MODEL_RUN_CONFIG_PATH = MODEL_OUTPUT_DIR / "run_config_resume.json"

SBC_PROGRESS_PATH = OUTPUT_ROOT / "strict_iid_sbc_progress_v1_a.npz"
SBC_METRICS_PATH = OUTPUT_ROOT / "strict_iid_sbc_metrics_v1_a.txt"
SBC_FIGURE_PATH = OUTPUT_ROOT / "strict_iid_sbc_rank_hist_v1_a.pdf"


def signal_seed(population_index: int, attempt_index: int = 0) -> int:
    return (
        SIGNAL_SEED_BASE
        + population_index * MAX_POPULATION_ATTEMPTS
        + attempt_index
    )


def noise_seed(population_index: int) -> int:
    return NOISE_SEED_BASE + population_index


def posterior_seed(population_index: int) -> int:
    return POSTERIOR_SEED_BASE + population_index


def validate_config() -> None:
    if NUM_SBC_SAMPLES < 1:
        raise ValueError("NUM_SBC_SAMPLES must be positive.")
    if NUM_POSTERIOR_SAMPLES + 1 < NUM_RANK_BINS:
        raise ValueError("There must be at least one possible rank per bin.")
    if (NUM_POSTERIOR_SAMPLES + 1) % NUM_RANK_BINS != 0:
        raise ValueError("NUM_POSTERIOR_SAMPLES + 1 must divide NUM_RANK_BINS exactly.")
    maximum_signal_seed = signal_seed(NUM_SBC_SAMPLES - 1, MAX_POPULATION_ATTEMPTS - 1)
    if maximum_signal_seed >= NOISE_SEED_BASE:
        raise ValueError("Signal and noise seed ranges overlap; increase NOISE_SEED_BASE.")
    maximum_noise_seed = noise_seed(NUM_SBC_SAMPLES - 1)
    if maximum_noise_seed >= POSTERIOR_SEED_BASE:
        raise ValueError("Noise and posterior seed ranges overlap; increase POSTERIOR_SEED_BASE.")
