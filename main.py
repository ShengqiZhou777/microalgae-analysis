import argparse
from algae_fusion.engine.pipeline import run_pipeline, run_loo_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algae Fusion Modular Pipeline")
    parser.add_argument("--target", type=str, default="Dry_Weight", help="Target variable")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "cnn_only", "boost_only"])
    parser.add_argument("--cv_method", type=str, default="random")

    parser.add_argument("--max_folds", type=int, default=1)
    parser.add_argument("--loo", action="store_true", help="Run LOO experiment")
    parser.add_argument("--stochastic", action="store_true", help="Use individual image matching for sliding window")

    args = parser.parse_args()

    if args.loo:
        run_loo_experiment(target_name=args.target, stochastic_window=args.stochastic)
    else:
        mf = args.max_folds if args.max_folds > 0 else None
        run_pipeline(
            target_name=args.target, 
            mode=args.mode, 
            cv_method=args.cv_method,
            max_folds=mf,
            stochastic_window=args.stochastic
        )
