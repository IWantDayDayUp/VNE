import config
from args import get_args


def run(config: config):
    print(f"\n{'-' * 20}    Start    {'-' * 20}\n")

    print(f"\n{'-' * 20}    End    {'-' * 20}\n")
    pass


if __name__ == "__main__":
    # 1. Get the command
    args = get_args()

    # 2. Update config according to the given command

    # 3. Generate Dataset, if already generated, load from file

    # 4. Run with config
    run(config=config)
