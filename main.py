import argparse
import sys
import os
from algae_fusion.engine.pipeline import run_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Algae Fusion Modular Pipeline")
    parser.add_argument("--target", type=str, default="Dry_Weight", help="Target variable")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "cnn_only", "boost_only", "ode","latent_ode"])
    parser.add_argument("--stochastic", action="store_true", help="Use individual image matching for sliding window")
    parser.add_argument("--condition", type=str, default="All", choices=["All", "Light", "Dark"], help="Filter by condition")
    parser.add_argument("--population-mean", action="store_true", help="Use population-mean time series for ODE mode")
    parser.add_argument("--ode-window-size", type=int, default=None, help="Force ODE to use fixed sliding window instead of full trajectory")
    args = parser.parse_args()
    
    # [LOGGING SETUP]
    # We set up logging based on args
    sys.path.append(os.path.dirname(os.path.abspath(__file__))) # Ensure local imports work
    from algae_fusion.utils.logger import setup_logger
    setup_logger(args.target, args.condition)

    run_pipeline(
        target_name=args.target, 
        mode=args.mode, 
        stochastic_window=args.stochastic,
        condition=args.condition,
        population_mean=args.population_mean,
        ode_window_size=args.ode_window_size
    )
