import argparse

from src.DQN import Agent


def args_parse():
    parser = argparse.ArgumentParser(description="Atari Gamer")

    # required argument
    parser.add_argument("--env", help="\"NoFrameskip\" environment supported only.")

    # optional argument
    parser.add_argument("--debug", action="store_true", help="Some parameters will be very small.")
    parser.add_argument("--DDQN", action="store_true", help="Use DDQN instead of natural DQN.")
    parser.add_argument("--mem", help="GPU memory (GB).")
    parser.add_argument("--render", action="store_true", help="GIF.")

    # mode -- choose 1 from 4
    parser.add_argument("--train", action="store_true", help="Train on a given environment.")
    parser.add_argument("--cont_train", help="Continue to train a given agent.")
    parser.add_argument("--play", help="Play on a given environment. Followed by save path.")

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    agent = Agent(args.env, args.debug, args.DDQN, args.mem, args.render)
    print("Environment:", args.env)

    if args.train:
        print("-----Start Training-----")
        agent.train()
    elif args.cont_train:
        print("-----Continue Training-----")
        agent.train(load_path=args.cont_train)
    elif args.play:
        print("-----Start Playing-----")
        agent.play(args.play, trials=5)
