"""Project-wide configuration for the Rawlsian MARL prototype."""

# --- Environment & random-policy evaluation (v0.1 / v0.2) ---
ENV_ID = "merge-v0"
N_EPISODES = 100
MAX_STEPS = 200
RAWLSIAN_XI = 0.2
SPEED_NORMALIZER = 30.0
BASE_SEED = 42

# --- v0.3: first learning-based prototype (DQN) ---
# Training timesteps are small and intended for pipeline validation, not final experiments.
DQN_TOTAL_TIMESTEPS = 20000
DQN_LEARNING_RATE = 1e-4
DQN_BUFFER_SIZE = 50000
DQN_LEARNING_STARTS = 1000
DQN_BATCH_SIZE = 32
DQN_GAMMA = 0.99
DQN_TRAIN_FREQ = 4
DQN_TARGET_UPDATE_INTERVAL = 1000

EVAL_EPISODES = 50

MODEL_DIR = "models"
LOG_DIR = "logs"

BASELINE_MODEL_PATH = "models/dqn_baseline.zip"
RAWLSIAN_MODEL_PATH = "models/dqn_rawlsian.zip"

# --- Results layout (by prototype version) ---
RESULTS_V02_DIR = "results/v0.2"
RESULTS_V03_DIR = "results/v0.3"
RESULTS_V04_DIR = "results/v0.4"
RESULTS_V04_RANDOM_DIR = "results/v0.4/random"
RESULTS_V04_TRAINED_DIR = "results/v0.4/trained"
RESULTS_V04_DIAGNOSTIC_DIR = "results/v0.4/diagnostic"

RESULTS_V05_DIR = "results/v0.5"
RESULTS_V05_RANDOM_DIR = "results/v0.5/random"
RESULTS_V05_TRAINED_DIR = "results/v0.5/trained"
RESULTS_V05_DIAGNOSTIC_DIR = "results/v0.5/diagnostic"

RANDOM_BASELINE_CSV = "results/v0.5/random/random_baseline.csv"
RANDOM_RAWLSIAN_CSV = "results/v0.5/random/random_rawlsian.csv"
RANDOM_SUMMARY_CSV = "results/v0.5/random/random_summary.csv"

TRAINED_BASELINE_CSV = "results/v0.5/trained/trained_baseline_eval.csv"
TRAINED_RAWLSIAN_CSV = "results/v0.5/trained/trained_rawlsian_eval.csv"
TRAINED_SUMMARY_CSV = "results/v0.5/trained/trained_summary.csv"

LEAST_ADVANTAGED_DIAGNOSTIC_CSV = "results/v0.4/diagnostic/least_advantaged_diagnostics.csv"
NEIGHBOURHOOD_SCOPE_DIAGNOSTIC_CSV = "results/v0.4/diagnostic/neighbourhood_scope_diagnostics.csv"
EXPERIENCE_COMPONENTS_DIAGNOSTIC_CSV = (
    "results/v0.5/diagnostic/experience_components_diagnostics.csv"
)

# --- v0.4: scoped Rawlsian fairness ---
# Default ego_neighbourhood because v0.3 diagnostics showed global min experience
# was usually dominated by non-ego background vehicles.
RAWLSIAN_SCOPE = "ego_neighbourhood"
EGO_NEIGHBOURHOOD_RADIUS = 50.0

# --- v0.5: safety-mobility experience function ---
EXPERIENCE_MODE = "safety_mobility"

TARGET_SPEED = 30.0
LOW_SPEED_THRESHOLD = 2.0

W_MOBILITY = 1.0
W_COLLISION = 2.0
W_LOW_SPEED = 0.3
W_RISK = 0.5

RISK_DISTANCE_NORMALIZER = 50.0
MIN_DISTANCE_EPSILON = 1e-6

# --- v0.6.1: multi-seed robustness ---
SEEDS = [0, 1, 2, 3, 4]

MULTISEED_MODEL_DIR = "models/v0.6.1"
MULTISEED_LOG_DIR = "logs/v0.6.1"
MULTISEED_RESULTS_DIR = "results/v0.6.1"

MULTISEED_TRAINED_EPISODE_CSV = "results/v0.6.1/multiseed_episode_results.csv"
MULTISEED_AGGREGATE_CSV = "results/v0.6.1/multiseed_aggregate_summary.csv"
