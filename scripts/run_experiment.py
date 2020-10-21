import argparse

import multi_scale_search as mss

def main(args):
    if args.gui:
        print("Creating fancy GUI")

    agent = mss.agents.FlatAgent()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    main(args)
