# Data parameters
LANGUAGE_PAIR = ('en', 'de')
D_MODEL = 512

# Optimizer configuration
#LEARNING_RATE = 1E-5
#LEARNING_RATE = 5E-5
#LEARNING_RATE = 4E-5
#LEARNING_RATE = 1E-4  # v2
#LEARNING_RATE = 2E-4  # v3
#LEARNING_RATE = 5E-4 (too big)
BETA1 = 0.9
BETA2 = 0.98
EPS = 1E-9

# Scheduler params
# LR_REDUCTION_FACTOR = 0.5

# Training parameters
BATCH_SIZE = 128
NUM_WARMUP = 40
#NUM_EPOCHS = 1_000
#NUM_EPOCHS = 100
NUM_EPOCHS = 30
