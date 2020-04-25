import argparse

from DQN.agent import Agent


def args_parse():
    parser = argparse.ArgumentParser(description="Atari Gamer")
    parser.add_argument("--env", help="\"NoFrameskip\" environment supported only.")
    parser.add_argument("--train", action="store_true", help="Train on a given environment.")
    parser.add_argument("--play", help="Play on a given environment. Followed by save path.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parse()
    agent = Agent(args.env)
    print("Environment: ", args.env)

    if args.train:
        print("-----Start training-----")
        agent.train()
    if args.play:
        print("-----Start training-----")
        agent.play(args.play, trials=1)
