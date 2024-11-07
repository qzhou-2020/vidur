from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main_yml(yml):
    config: SimulationConfig = SimulationConfig.create_from_yml(yml)
    config.write_config_to_file()

    set_seeds(config.seed)

    simulator: Simulator = Simulator(config)
    simulator.run()


def parallel_main_yml(case_name: str = 'vllm'):
    """
    This function is used to run the simulation in parallel.

    TODO:
        - error occurs when running in parallel
        - this is because the execution time predictor is not thread-safe (they share the same cache/)
    """
    from joblib import Parallel, delayed
    import os
    import glob
    
    __dirname__ = os.path.dirname(os.path.abspath(__file__))

    # Get all yaml files from the case directory
    yaml_files = glob.glob(os.path.join(__dirname__, "case", case_name, "*.yml"))
    
    Parallel(
        n_jobs=1
    )(delayed(main_yml)(yaml_file) for yaml_file in yaml_files)


if __name__ == "__main__":
    yml = r"/Users/qzhou/Documents/github/vidur/case/uniform/faster_transformer/T400.yml"
    main_yml(yml)
