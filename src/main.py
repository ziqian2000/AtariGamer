import argparse

from src.DQN import Agent


def args_parse():
    parser = argparse.ArgumentParser(description="Atari Gamer")
    parser.add_argument("--env", help="\"NoFrameskip\" environment supported only.")
    parser.add_argument("--train", action="store_true", help="Train on a given environment.")
    parser.add_argument("--play", help="Play on a given environment. Followed by save path.")
    parser.add_argument("--debug", action="store_true", help="Some parameters will be very small.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parse()
    agent = Agent(args.env, args.debug)
    print("Environment:", args.env)

    if args.train:
        print("-----Start Training-----")
        agent.train()
    if args.play:
        print("-----Start Playing-----")
        agent.play(args.play, trials=5)
