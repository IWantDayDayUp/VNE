from config import Config
from args import get_args


def run(config: Config):
    print(f"\n{'-' * 20}    Start    {'-' * 20}\n")

    print(f"\n{'-' * 20}    End    {'-' * 20}\n")
    pass


if __name__ == "__main__":
    # 1. Get the command
    args = get_args()

    # 2. Update config according to the given command
    config = Config(
        p_net_setting_path=args.p_net_setting_path,
        v_net_setting_path=args.v_net_setting_path,
    )
    config.update(args)

    # 3. Generate Dataset, if already generated, load from file

    # 4. Run with config
    run(config=config)
