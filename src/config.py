# src/config.py

TARGET = "Y"
ID_COL = "ID"

# Business costs
COST_FP = 1
COST_FN = 5

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

THRESH_GRID = [i / 100 for i in range(1, 100)]  # 0.01..0.99