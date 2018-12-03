echo "Running ShuffleNetv2 1.0"
python -W ignore run_benchmark_speed.py 1.0
python -W ignore run_benchmark_accuracy.py 1.0

echo "Running ShuffleNetv2 0.5"
python -W ignore run_benchmark_speed.py 0.5
python -W ignore run_benchmark_accuracy.py 0.5


