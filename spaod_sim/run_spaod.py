import argparse

def main():
    parser = argparse.ArgumentParser(description="SPAO-D simulation skeleton")
    parser.add_argument("--config", type=str, default="configs/starlink_1584.yaml")
    parser.add_argument("--alg", type=str, default="spaod", choices=["spaod","baseline_shortest","baseline_random"])
    parser.add_argument("--seeds", type=int, default=1)
    args = parser.parse_args()
    print("[Skeleton] config=", args.config, " alg=", args.alg, " seeds=", args.seeds)

if __name__ == "__main__":
    main()
