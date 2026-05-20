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

TRAINED_BASELINE_CSV = "results/trained_baseline_eval.csv"
TRAINED_RAWLSIAN_CSV = "results/trained_rawlsian_eval.csv"
TRAINED_SUMMARY_CSV = "results/trained_summary.csv"
