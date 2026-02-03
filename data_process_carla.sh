###################################
# User Configuration Section
###################################
CARLA_DATA_PATH="/home/dataset-local/Bench2Drive_Data/Bench2Drive-Base-Tar" # dataset root containing route folders
CARLA_MAP_PATH="/home/dataset-local/Bench2Drive_Data/Bench2Drive-Map" # map root or map file (e.g., "Town01_HD_map.npz")

TRAIN_SET_PATH="/home/dataset-local/b2d_processed_data/processed_data" # output npz dir
OUTPUT_LIST_PATH="/home/dataset-local/wsz/b2d_diifusion/train_data.json" # output list json
CACHE_DIR="/home/dataset-local/b2d_processed_data/cache" # cache dir (leave empty to disable)
###################################

python data_process_carla.py \
--data_path "$CARLA_DATA_PATH" \
--map_path "$CARLA_MAP_PATH" \
--train_set_path "$TRAIN_SET_PATH" \
--output_list "$OUTPUT_LIST_PATH" \
--cache_dir "$CACHE_DIR"
