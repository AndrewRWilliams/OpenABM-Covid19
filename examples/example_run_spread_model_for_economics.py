import functools
import multiprocessing
import os
import pickle
from typing import Mapping, Union

import numpy as np
import pandas as pd


import adapter_covid19
from adapter_covid19.datasources import Reader
from adapter_covid19.data_structures import Scenario
from adapter_covid19.enums import Age10Y, Region
from adapter_covid19.lockdown import get_lockdown_factor, get_working_factor
from COVID19.model import Parameters, OccupationNetworkEnum

import example_utils as utils


def set_occupation_params(params: Parameters, model, value: float):
    occupation_params = {
        f"lockdown_occupation_multiplier{on.name}": params.get_param(
            f"lockdown_occupation_multiplier{on.name}"
        )
        for on in OccupationNetworkEnum
    }
    if not all(np.isclose(v, value) for v in occupation_params.values()):
        for on in OccupationNetworkEnum:
            model.update_running_params(
                f"lockdown_occupation_multiplier{on.name}", value
            )


def run_worker(
    #mapping of age slices, but no region mentioned
    populations: Mapping[Age10Y, int],
    lockdown_start: int,
    lockdown_end: int,
    end: int,
    slow_unlock: bool,
    data_path: str,
    spread_model_params: Mapping[str, Union[str, float, int, bool]],
):
    population = 100_000
    #gets baseline_parameters.csv as Parameters class
    params = utils.get_baseline_parameters()
    params.set_param("n_total", population)

    #sets params for each each age group into the c_params file
    for k, v in populations.items():
        params.set_param(k.value, v)

    #returns Simulator class from OpenABM 
    sim = utils.get_simulation(params)

    #run the simulator UNTIL the lockdown starts
    sim.steps(lockdown_start)

    #update the parameters based on lockdown status
    for k, v in spread_model_params.items():
        sim.env.model.update_running_params(k, v)
    lockdown_factor = get_lockdown_factor(
        lockdown=True, slow_unlock=slow_unlock, lockdown_exit_time=0, time=sim.timestep
    )
    occupation_factor = get_working_factor(data_path, lockdown_factor)
    set_occupation_params(params, sim.env.model, occupation_factor)

    sim.env.model.update_running_params("lockdown_on", 1)
    sim.steps(lockdown_end - lockdown_start)

    if not slow_unlock:
        sim.env.model.update_running_params("lockdown_on", 0)
    while sim.timestep < end:
        lockdown_factor = get_lockdown_factor(
            lockdown=False,
            slow_unlock=slow_unlock,
            lockdown_exit_time=lockdown_end,
            time=sim.timestep,
        )
        occupation_factor = get_working_factor(data_path, lockdown_factor)
        set_occupation_params(params, sim.env.model, occupation_factor)
        sim.steps(min(end - sim.timestep, 10))

    timeseries = pd.DataFrame(sim.results)
    ill_ratio = timeseries["n_symptoms"] / population
    dead_ratio = timeseries["n_death"] / population
    quarantine_ratio = timeseries["n_quarantine"] / population
    data = {
        "ill_ratio": ill_ratio.to_dict(),
        "dead_ratio": dead_ratio.to_dict(),
        "quarantine_ratio": quarantine_ratio.to_dict(),
    }
    return data


def run(scenario: Scenario, data_path: str, reload: bool = False) -> None:
    reader = Reader(data_path)

    # population per region per age slice of 10 years
    populations_df = reader.load_csv("populations", orient="dataframe")

    # multidict with region, age group
    # e.g. populations_by_region[Region.C_NE][Age10Y.A30] 
    #      returns the int representing population from 30 to 39 in C_NE
    populations_by_region = {
        Region[k]: {Age10Y[kk]: vv for kk, vv in v.items()}
        for k, v in populations_df.set_index("region").T.to_dict().items()
    }

    # lockdown parameters
    lockdown_start, lockdown_end, end, slow_unlock = scenario.get_lockdown_info()

    #file_name = f"spread_model_cache_{lockdown_start}_{lockdown_end}_{end}_{self.slow_unlock}"
    # extension if spread model params are specifiec
    file_name = scenario.get_spread_model_filename()
    file_path = os.path.join(data_path, f"{file_name}.pkl")

    # define partial run_worker function
    # run worker function returns timeseries of ratios of ill, dead and quarantine
    if not os.path.exists(file_path) or reload:
        worker = functools.partial(
            run_worker,
            lockdown_start=lockdown_start,
            lockdown_end=lockdown_end,
            end=end,
            slow_unlock=slow_unlock,
            data_path=data_path,
            spread_model_params=scenario.spread_model_params,
        )
    
    # applies run_worker function to each dict of populations by region
        with multiprocessing.Pool() as pool:
            data = pool.map(worker, [populations_by_region[r] for r in Region])
        keys = data[0].keys()
        data = {
            k: {t: {r: data[i][k][t] for i, r in enumerate(Region)} for t in range(end)}
            for k in keys
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)


def get_spread_data(
    scenario: Scenario, data_path: str, reload: bool = False
) -> pd.DataFrame:
    file_name = scenario.get_spread_model_filename()
    file_path = os.path.join(data_path, f"{file_name}.pkl")

   
#   if the pickle file of ill, dead, quarantine timeseries doesn't exist yet, run the simulation to make it
    if not os.path.exists(file_path) or reload:
        run(scenario, data_path, reload)
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    total_population_by_region = pd.DataFrame(populations_by_region).sum()
    pct_population_by_region = (
        total_population_by_region / total_population_by_region.sum()
    )
    seriess = {
        k: (pd.DataFrame(v).T * pct_population_by_region).sum(axis=1)
        for k, v in data.items()
    }
    df = pd.concat(seriess, axis=1)
    return df


def plot_spread_data(scenario: Scenario, data_path: str, reload: bool = False):
    return df.plot(subplots=True, figsize=(12, 12))


if __name__ == "__main__":
    import sys
    from adapter_covid19.scenarios import SCENARIOS

    if len(sys.argv) < 2 or sys.argv[1].lower() not in SCENARIOS:
        valid = "|".join(SCENARIOS)
        print(f"Example usage: python {sys.argv[0]} <{valid}>")
    else:
        data_path = os.path.join(
            os.path.dirname(__file__), "../tests/adapter_covid19/data"
        )
        run(SCENARIOS[sys.argv[1].lower()], data_path)
