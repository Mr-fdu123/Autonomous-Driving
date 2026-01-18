###################################
# User Configuration Section
###################################
CARLA_DATA_PATH="/d/Bench2Drive-mini" # dataset root containing route folders
CARLA_MAP_PATH="/d/Autonomous-Driving/Autonomous-Driving/Town01_HD_map.npz" # map root or map file (e.g., "Town01_HD_map.npz")

TRAIN_SET_PATH="/d/Bench2Drive-mini/processed_data" # output npz dir
OUTPUT_LIST_PATH="/d/Autonomous-Driving/Autonomous-Driving/train_data.json" # output list json
CACHE_DIR="/d/Bench2Drive-mini/cache_data" # cache dir (leave empty to disable)
###################################

python data_process_carla.py \
--data_path "$CARLA_DATA_PATH" \
--map_path "$CARLA_MAP_PATH" \
--train_set_path "$TRAIN_SET_PATH" \
--output_list "$OUTPUT_LIST_PATH" \
--cache_dir "$CACHE_DIR"
