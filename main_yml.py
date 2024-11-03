from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main_yml():
    yml = r"/Users/qzhou/Documents/github/vidur/case/base.yml"
    config: SimulationConfig = SimulationConfig.create_from_yml(yml)
    config.write_config_to_file()

    set_seeds(config.seed)

    # simulator: Simulator = Simulator(config)


if __name__ == "__main__":
    main_yml()