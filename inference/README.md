# Basic evaluation on TEST set
python inference/evaluate_map.py \
  --model_path fasterrcnn_turbine_adam_map.pth \
  --config config/config.yaml

# With custom batch size and workers
python inference/evaluate_map.py \
  --model_path fasterrcnn_turbine_adam_map.pth \
  --config config/config.yaml \
  --batch_size 8 \
  --num_workers 8

# With custom run name
python inference/evaluate_map.py \
  --model_path fasterrcnn_turbine_adam_map.pth \
  --config config/config.yaml \
  --run_name "final_test_baseline"