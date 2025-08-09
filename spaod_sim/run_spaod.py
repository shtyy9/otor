import argparse
from concurrent.futures import ProcessPoolExecutor


def _run_seed(seed: int, config: str, alg: str):
    """Placeholder for a single simulation run."""
    print(f"[Seed {seed}] config={config} alg={alg}")


def main():
    parser = argparse.ArgumentParser(description="SPAO-D simulation skeleton")
    parser.add_argument("--config", type=str, default="configs/starlink_1584.yaml")
    parser.add_argument(
        "--alg", type=str, default="spaod", choices=["spaod", "baseline_shortest", "baseline_random"]
    )
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()

    # Run each seed in its own process to better utilise CPU cores.
    seeds = list(range(args.seeds))
    if len(seeds) > 1:
        with ProcessPoolExecutor() as exe:
            exe.map(_run_seed, seeds, [args.config]*len(seeds), [args.alg]*len(seeds))
    else:
        _run_seed(seeds[0], args.config, args.alg)

if __name__ == "__main__":
    main()
