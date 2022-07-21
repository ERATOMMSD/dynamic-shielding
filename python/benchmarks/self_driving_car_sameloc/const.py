# modes
NO_SHIELD = 0
SAFE_PADDING = 1
DYNAMIC_PREEMPTIVE_MODE = 2


##################################################################################
GRID = 60  # size of the output abstraction grid
XSIZE = 480.
YSIZE = 480.
VIZ = False
TOTAL_TRAINING_STEP = 200000
TEST_EVERY_STEP = 10000
MAX_CAR_STEP = 44
SAT_CORNER = 4
CAR_SPEED = 3
USE_GRID = True

####################################################################################

MODE_NAME_LIST = ['NO_SHIELD', 'PADDING', 'SHIELD']

AREA_1 = [0, 160, 320, 480]
AREA_2 = [0, 160, 0, 140]
AREA_3 = [330, 480, 0, 140]
AREA_4 = [330, 480, 320, 480]