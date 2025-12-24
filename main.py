import argparse
from algae_fusion.engine.pipeline import run_pipeline, run_loo_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algae Fusion Modular Pipeline")
    parser.add_argument("--target", type=str, default="Dry_Weight", help="Target variable")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "cnn_only", "boost_only"])
    # [Removed] cv_method, max_folds

    parser.add_argument("--loo", action="store_true", help="Run LOO experiment")
    parser.add_argument("--stochastic", action="store_true", help="Use individual image matching for sliding window")

    parser.add_argument("--condition", type=str, default="All", choices=["All", "Light", "Dark"], help="Filter by condition")

    args = parser.parse_args()

    if args.loo:
        run_loo_experiment(target_name=args.target, stochastic_window=args.stochastic)
    else:
        run_pipeline(
            target_name=args.target, 
            mode=args.mode, 
            stochastic_window=args.stochastic,
            condition=args.condition
        )
