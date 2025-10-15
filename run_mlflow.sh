root_dir=$(pwd)

mlflow ui --backend-store-uri file://$root_dir/data/mlruns --port 5002