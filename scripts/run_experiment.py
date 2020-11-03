import argparse
from src.low_fidelity_simulation.gui_low_fidelity_sim import GUI_LFS
#import multi_scale_search as mss
import config


def main(args):
    if args.gui:
        config.init('low_fidelity_simulation')
        gui = GUI_LFS()
        gui.run()


    #agent = mss.agents.FlatAgent()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    #args.gui = True
    main(args)
