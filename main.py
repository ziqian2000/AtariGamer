import argparse

from DQN.agent import Agent


def args_parse():
    parser = argparse.ArgumentParser(description="Atari Gamer")
    parser.add_argument("--env", help="\"NoFrameskip\" environment supported only.")
    parser.add_argument("--train", action="store_true", help="Train on a given environment.")
    parser.add_argument("--play", help="Play on a given environment. Followed by save path.")
    parser.add_argument("--local", action="store_true", help="The memory will be set small if running locally.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parse()
    agent = Agent(args.env, args.local)
    print("Environment:", args.env)

    if args.train:
        print("-----Start Training-----")
        agent.train()
    if args.play:
        print("-----Start Playing-----")
        agent.play(args.play, trials=5)
