import argparse
from low_fidelity_simulation.gui_low_fidelity_sim import GUI_LFS
import config
import logging

def main(args):
    logging.basicConfig(filename='infos.log', level=logging.INFO,
                        format='%(levelname)s - %(name)s - %(asctime)s - %(message)s')
    if args.gui:
        config.init('low_fidelity_simulation')
        gui = GUI_LFS()
        gui.run()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    #args.gui = True
    main(args)
