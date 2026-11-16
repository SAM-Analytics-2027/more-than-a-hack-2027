"""All the constants and hyperparameters used in models/, in one place.

Each constant is documented with:
- what it represents, physically or algorithmically;
- where it comes from (official dataset specifications, kinematic analysis of the tracking,
  modelling choices, coordinate search of the Viterbi decoder, etc.);
- the source files that use it.
"""
import numpy as np

# ==================================================================================================
# 1. DATA FORMAT, PITCH AND COMPETITION METRICS (epts_reader.py, dataset.py)
# ==================================================================================================

# Frames per second of the tracking (FIFA EPTS / Metrica Sports).
# 1 frame = 1/25 s = 0.040 s (40 milliseconds).
FPS = 25

# Numeric indices of the home team (0) and the away team (1).
HOME = 0
AWAY = 1

# Possession class codes:
# 0 = Home, 1 = Away, 2 = Dead ball, 3 = outside the halves.
H, A, D, X = 0, 1, 2, 3
CODE = {'H': 0, 'A': 1, 'D': 2, 'X': 3}

# Inverse array to turn the numeric index (0..3) into the letter written in the output CSVs.
LETTERS = np.array(['H', 'A', 'D', 'X'])

# Total number of frames of the tracking of the competition match Barça–Athletic (data/tactical/).
# Defined in the official dataset specifications (games.py and DATA.md).
OUR_FRAMES = 199319

# Pitch size of Barça–Athletic in metres (length X = 105.3 m, width Y = 68.0 m), as in games.py
# (DATA.md: measured from the corrected tracking; the XML metadata report 105 × 68).
OUR_LENGTH = 105.3
OUR_WIDTH = 68.0

# Inclusive frame intervals of the two halves of Barça–Athletic (0-based):
# 1st half: from frame 11,185 to frame 90,651
# 2nd half: from frame 110,808 to frame 190,197
# All the frames in between or outside have period=0 and label 'X'.
OUR_HALVES = ((1, 11185, 90651), (2, 110808, 190197))

# Tolerance in metres beyond the pitch boundary lines (goal lines and touchlines).
# If the ball goes beyond (L/2 + 0.5) or (W/2 + 0.5) it is marked as out of the pitch (ball_out = 1).
# The 0.5 m cover the ball radius (~11 cm) and the uncertainty of the degraded optical tracking.
FIELD_BOUND_TOLERANCE = 0.5

# Conventional identifier of the competition match (source 'barca', ID 'BAR-ATH').
BARCA = ('barca', 'BAR-ATH')

# The three sources of the 30 external matches used for training and validation:
# - 'dfl': 7 German league matches (DFL tracking, high quality, original labels at every touch)
# - 'metrica': 3 Metrica Sports sample matches (high quality)
# - 'skillcorner': 20 matches with automatic tracking from TV footage (much noise and occlusion)
SOURCES = ('dfl', 'metrica', 'skillcorner')

# The 10 matches with high-precision tactical tracking on which the model is evaluated and tuned
# (the second page of the dashboard is based on the mean over these 10 matches).
DM_SOURCES = ('dfl', 'metrica')
DM = DM_SOURCES

# Complete, fixed list of the 30 external matches (dataset.py).
MATCHES = {
    'dfl': ['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9'],
    'metrica': ['Sample_Game_1', 'Sample_Game_2', 'Sample_Game_3'],
    'skillcorner': [
        '1874553', '1886347', '1899585', '1925299', '1927964', '1953632', '1959846', '1986691',
        '1996435', '1996436', '2006229', '2006363', '2007448', '2007721', '2010085', '2011166',
        '2013725', '2015213', '2016236', '2017461',
    ],
}

# Version of the feature table saved in the cache.
VERSION = 'v2'


# ==================================================================================================
# 2. KINEMATIC FEATURES AND SPATIAL THRESHOLDS (features.py)
# ==================================================================================================

# Half window (in frames) for the centred speeds and accelerations:
# dt = 2 * SPEED_HALF = 4 frames = 0.16 seconds.
# Chosen empirically: a one-frame difference amplifies the sensor noise;
# a ±2-frame window smooths the jitter without a noticeable delay in changes of direction.
SPEED_HALF = 2

# Speed threshold in m/s (2.0 m/s = 7.2 km/h):
# below this speed a player is considered walking or standing still.
# Used by the feature `walking_share` (share of players walking),
# a key signal to tell dead ball from live play.
WALK = 2.0
WALK_SPEED_THRESH = 2.0

# Radius in metres (3.0 m) to find the players "near" the ball.
# Used to count the immediate pressure (`near_home`, `near_away`).
NEAR = 3.0
NEAR_DIST_THRESH = 3.0

# Radius in metres (1.8 m) within which the ball counts as "touched" or controlled by a player.
# It is 1.8 m (instead of a real ~1.0 m) because in the degraded data the ball coordinates
# are noisy and can be shifted by up to ~0.8 m.
TOUCH = 1.8
TOUCH_DIST_THRESH = 1.8

# Centred half windows (in frames) for the moving averages of the base features:
# - 12 frames: ±0.48 s (~±0.5 s, immediate dynamics of the duel)
# - 37 frames: ±1.48 s (~±1.5 s, short-term possession)
# - 112 frames: ±4.48 s (~±4.5 s, build-up phase and team positioning)
WINDOWS = (12, 37, 112)
CONVOLUTION_HALF_WINDOWS = WINDOWS

# Asymmetric windows (in frames) for the past (_p) and future (_f) means:
# - 25 frames = 1.0 s (immediate inertia or outlook)
# - 75 frames = 3.0 s (medium-term context of the action)
ASYMMETRIC_WINDOWS = (25, 75)

# Time windows (in frames) to count the received touches and the ball holding:
# - 50 frames = 2.0 s
# - 125 frames = 5.0 s
# They let the model capture the official rules ("more than two consecutive passes", "keeps the ball").
TOUCH_COUNT_WINDOWS = (50, 125)

# Saturation (in frames) of the time since the ball/touch was last seen and until the next one.
# 1500 frames = 60.0 seconds. If the ball has not been seen for over 1 minute, the feature saturates at 1500.
GAP_CLIP_FRAMES = 1500

# Additive offset (in frames, 10.0 s) given to `since_ball_seen` or `until_ball_seen`
# when the ball has never been seen since the start of the half or until its end.
NO_BALL_INITIAL_OFFSET = 250

# Clipping limit of the run of consecutive passes of the same team (`_streak`).
# Between -20 (20 consecutive away passes) and +20 (20 consecutive home passes).
MAX_STREAK = 20

# Maximum number of frames without touches before the count of consecutive passes resets (`_streak`),
# so that passes made before a restart/dead ball are not added to the next action.
STREAK_RESET_GAP = 100

# Maximum gap (in frames, 125 frames = 5.0 s) over which the ball path is linearly interpolated
# (`bx_lin`, `by_lin`) when both ends are inside the pitch.
BALL_INTERP_MAX_GAP = 125

# Percentile of the speeds of all the players on the pitch in a single frame (90th percentile).
# It represents the top speed of the most active players in the action, ignoring outliers.
PERCENTILE_VELOCITY = 90


# ==================================================================================================
# 3. SAMPLING, TRAINING AND CROSS-VALIDATION (cross_validation.py, model_families.py, reproduce.py)
# ==================================================================================================

# Global random seed (2027, the year of the hackathon) for full reproducibility
# of the fold split, the weight initialisation and the stochastic models.
SEED = 2027

# Number of folds of the cross-validation, balanced by source.
# Each fold holds 6 whole matches (DFL, Metrica and SkillCorner balanced).
N_FOLDS = 5

# Sampling step when training the tabular models (HGB, XGB, LGBM):
# 1 frame every 8 (0.32 s = ~3.1 Hz).
# Contiguous frames at 25 fps are nearly redundant; sampling every 8 frames cuts the training time
# by 87% without losing accuracy in the feature distributions.
STEP = 8
TRAIN_SUBSAMPLE_STEP = 8

# Metadata columns of the feature DataFrame, excluded from the training set.
META = ('frame', 'period', 'y')

# Relative weight of the frames of the SkillCorner matches in training (1/4 = 0.25).
# The SkillCorner matches come from TV footage with frequent occlusions.
# At weight 0.25 the 20 of them count as much as 5 tactical matches, so that the TV noise
# does not drown the clean signal of the 10 DFL + Metrica matches.
TV_WEIGHT = 0.25

# Weighting by possession duration (`short_spell_weights`):
# weight w = sqrt(median_duration / possession_duration), clipped to [1/3, 3.0].
# Short possessions (1-2 seconds) are the hardest to predict and were often ignored
# by the model: giving them more weight clearly raises spellF1 on quick actions.
SHORT_SPELL_WEIGHT_MIN = 1.0 / 3.0
SHORT_SPELL_WEIGHT_MAX = 3.0

# Half window (±5 frames = ±0.2 s) around a change of possession,
# used to train the auxiliary event model (`cross_validation.py`).
EV_HALF = 5
EVENT_HALF_WINDOW = 5

# Subsampling of the frames with an event (1 every 2) and without (1 every 16), with matching weights,
# to balance how rare the changes of possession are in the dataset.
EVENT_SUBSAMPLE_POS = 2
EVENT_SUBSAMPLE_NEG = 16
EVENT_WEIGHT_POS = 2.0
EVENT_WEIGHT_NEG = 16.0

# Rounding to 4 decimals applied to the ensemble probabilities before Viterbi.
# Essential to get exactly the same labels bit for bit
# regardless of the processor architecture (x86 vs ARM/Apple Silicon, Windows vs macOS).
DECIMALS = 4
PROB_ROUND_DECIMALS = 4

# Number of threads of the gradient boosting models (XGBoost, LightGBM).
THREADS = 8


# ==================================================================================================
# 4. ARCHITECTURE AND HYPERPARAMETERS OF THE NEURAL NETWORK (tcn.py)
# ==================================================================================================

# Temporal decimation step of the sequence for the convolutional network:
# 5 frames per second (0.2 s step). The network works at 5 Hz instead of 25 Hz for efficiency,
# and its predictions are then linearly interpolated back to 25 fps.
STRIDE = 5
NET_STRIDE = 5

# Length of the training sequences (temporal crop):
# 300 steps at 5 fps = exactly 60 seconds of continuous play.
CROP = 300
NET_CROP = 300

# Number of training epochs of each network (default: 12 epochs with importance sampling).
EPOCHS = 12
NET_EPOCHS = 12

# Intermediate channels of the 1D convolutional network.
CHANNELS = 96
NET_CHANNELS = 96

# Temporal dilations of the 7 residual convolutional blocks:
# dilations 1, 2, 4, 8, 16, 32, 64 steps.
# Resulting receptive field: ±(1 + 2 + 4 + 8 + 16 + 32 + 64) = ±127 steps at 5 fps,
# i.e. ±25.4 seconds of temporal context (past and future) around each frame.
DILATIONS = (1, 2, 4, 8, 16, 32, 64)
NET_DILATIONS = DILATIONS

# Dropout inside each residual block, against overfitting.
DROPOUT = 0.15
NET_DROPOUT = DROPOUT

# Maximum learning rate of the AdamW optimiser with the OneCycleLR scheduler.
LR = 2e-3
NET_MAX_LR = LR

# Weight decay (L2 regularisation) for AdamW.
WEIGHT_DECAY = 1e-4
NET_WEIGHT_DECAY = WEIGHT_DECAY

# Batch size (32 crops of 60 seconds, sampled in proportion to the match length).
BATCH_SIZE = 32
NET_BATCH_SIZE = BATCH_SIZE

# Gradient norm clipping threshold.
GRAD_CLIP = 1.0
NET_GRAD_CLIP = GRAD_CLIP

# Feature sampling step (1 row every 40 frames = 1.6 s) to compute quickly and robustly
# the median and the interquartile range (IQR) used for normalisation.
NORM_SUBSAMPLE_STEP = 40
NET_NORM_SUBSAMPLE_STEP = NORM_SUBSAMPLE_STEP

# Clipping of the features standardised with (x - med) / IQR, so that outliers do not blow up:
# values are limited to [-6.0, +6.0].
NET_FEATURE_CLIP = (-6.0, 6.0)

# Number of independent convolutional networks in the final ensemble (N_NETS = 3).
# Each network has the same architecture but a different seed
# (seed, seed + 1000, seed + 2000), which reduces the variance of the predictions.
N_NETS = 3

# Auxiliary temporal smoothing loss (truncated MSE) between adjacent frames with the same label,
# and weight of the multi-task transition head of the TCN.
SMOOTH_WEIGHT = 0.15
SMOOTH_TAU = 4.0
TRANS_WEIGHT = 0.15

# Weights of the log-opinion pool (weighted geometric mean of the tree family and the TCN family).
W_TREE = 0.15
W_NETS = 0.85
ALPHA_LGBM = 0.50

# Maximum distance (in metres) between the tactical and the broadcast player centroids for a broadcast frame
# to count as live play and not a TV replay, when fusing the ball for Barça–Athletic.
BROADCAST_MAX_CENTROID_DIST = 12.0

# Name of the reference model
MODEL_NAME = 'SAM-model'
NAME = MODEL_NAME


# ==================================================================================================
# 5. HYPERPARAMETERS OF THE TREE MODELS (cross_validation.py, model_families.py)
# ==================================================================================================

# Hyperparameters of scikit-learn's HistGradientBoostingClassifier:
HGB_PARAMS = {
    'max_iter': 260,               # maximum number of boosting trees
    'learning_rate': 0.08,          # learning rate of each added tree
    'max_leaf_nodes': 63,          # maximum number of leaves per tree (~depth 6)
    'min_samples_leaf': 200,       # minimum number of samples per leaf (avoids memorising noise)
    'l2_regularization': 1.0,       # L2 penalty on the leaf values
    'early_stopping': True,        # stop early when the loss stops decreasing
    'validation_fraction': 0.1,    # internal share of the data kept for early stopping
    'n_iter_no_change': 20,        # stop training if there is no improvement for 20 iterations
    'random_state': SEED,          # reproducible boosting
}

# Hyperparameters of the XGBoost tried in model_families.py
XGB_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'max_depth': 8,
    'eta': 0.06,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 50,
    'reg_lambda': 2.0,
    'eval_metric': 'mlogloss',
    'nthread': THREADS,
    'seed': SEED,
}

# Hyperparameters of LightGBM (model_families.py and the final model in reproduce.py)
LGBM_PARAMS = {
    'learning_rate': 0.05,
    'num_leaves': 63,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 2.0,
    'num_threads': THREADS,
    'seed': SEED,
    'verbose': -1,
}


# ==================================================================================================
# 6. VITERBI DECODING AND FROZEN PARAMETERS (decoder.py, frozen/viterbi_params.json)
# ==================================================================================================

# Infinity value for the dynamic programming matrix of Viterbi.
INF = 1e18
INF_COST = INF

# Minimum probability allowed when converting to costs (-log(p)):
# avoids divisions by zero and infinite costs (p=0 becomes -log(1e-4) ~ 9.21).
PROB_CLIP_MIN = 1e-4

# Minimum probability allowed for the probabilities of the auxiliary events.
EVENT_PROB_CLIP_MIN = 1e-3

# Default parameters before tuning
DEFAULT = dict(turn=6.0, stop=6.0, rest=6.0, m_pos=20, m_dead=40, bias_d=0.0)
DEFAULT_VITERBI_PARAMS = DEFAULT

# Starting point of the coordinate search of the Viterbi decoder for the final model `SAM-model`
# (2 rounds, maximising spellF1 on the 10 DFL + Metrica matches); the tuned values are in frozen/viterbi_params.json.
START = dict(
    turn=9.5, stop=4.0, rest=9.0, m_pos=5, m_dead=60, bias_d=-0.6,
    pen_no_touch=2.0, touch_th=2.8, m_pre=15, pen_pre=0.5, pen_same=1.0,
)
VITERBI_PARAMS_FROZEN = START

# Search grid of the coordinate search of the Viterbi decoder (cross_validation.py).
GRID = {
    'turn': [6.0, 8.0, 9.0, 9.5, 10.0, 11.5, 13.0, 18.0],
    'stop': [3.0, 4.0, 5.0, 6.0, 9.0, 13.0],
    'rest': [6.0, 8.0, 9.0, 9.5, 11.0, 13.0],
    'm_pos': [4, 5, 6, 8, 12, 20],
    'm_dead': [40, 50, 55, 60, 70],
    'bias_d': [-0.7, -0.6, -0.5, -0.4, -0.25, 0.0, 0.25],
    'pen_no_touch': [0.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    'touch_th': [2.0, 2.2, 2.5, 2.8],
    'm_pre': [15, 25, 35, 50],
    'pen_pre': [0.0, 0.5, 1.0, 1.5],
    'pen_same': [-1.0, 0.0, 0.5, 1.0, 1.5],
}
VITERBI_GRID = GRID


# ==================================================================================================
# 7. RICOCHET FILTER ON THE DFL LABELS (dfl_ricochet_filter.py)
# ==================================================================================================

# Radius in metres (1.60 m) within which a player can physically reach and control the ball.
REACH = 1.60
RICOCHET_REACH = REACH

# Maximum ball speed in m/s (16.0 m/s = 57.6 km/h) for a controlled pass.
# Above this speed it is a shot or a hard clearance, not control.
PASS_SPEED = 16.0
RICOCHET_PASS_SPEED = PASS_SPEED

# Minimum duration in seconds (0.24 s = 6 frames at 25 fps) of continuous control
# for a contact to count as an intentional possession and not a simple ricochet.
MIN_CTRL_S = 0.24
RICOCHET_MIN_CTRL_S = MIN_CTRL_S

# Maximum duration in seconds (1.60 s = 40 frames at 25 fps) for a stretch
# to count as a potential isolated/short possession to be checked by the ricochet filter.
MAX_PROBE_S = 1.60
RICOCHET_MAX_PROBE_S = MAX_PROBE_S
