"""
Example of adding a simple arsenic contamination event to a scenario.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
from epyt_flow.data.benchmarks import load_leakdb_scenarios
from epyt_flow.simulation import ScenarioSimulator, EpanetConstants, ScenarioConfig
from epyt_flow.simulation.events import SpeciesInjectionEvent
from epyt_flow.utils import to_seconds
from epyt_flow.simulation.scada.scada_data_export import ScadaDataXlsxExport, ScadaDataNumpyExport

import matplotlib.pyplot as plt


if __name__ == "__main__":

    for x in range(1, 21):
        # Create a new scenario based on the first Net1 LeakDB scenario --
        # we add an additional EPANET-MSX configuration file
        config, = load_leakdb_scenarios(scenarios_id=[x], use_net1=True)
        config = ScenarioConfig(scenario_config=config,
                                f_msx_in="arsenic_contamination.msx")

        with ScenarioSimulator(scenario_config=config) as sim:
            # Set simulation duration to 21 days
            sim.set_general_parameters(simulation_duration=to_seconds(days=21))

            # Place some chlorine sensors and also keep track of the contaminant
            cl_sensor_locations = ["10", "11", "12", "13", "21", "22", "23", "31", "32"]
            all_nodes = sim.sensor_config.nodes
            sim.set_bulk_species_node_sensors({"Chlorine": cl_sensor_locations,
                                            # Also: Keep track of the contaminant
                                            "AsIII": all_nodes})   # Arsenite

            # Chlorine injection at node "10" -- i.e. a constant concentration source of 1mg/L
            sim.add_quality_source(node_id="10",
                        pattern=np.array([1.]),
                        source_type=EpanetConstants.EN_CONCEN)

            
            # Run simulation
            scada_data = sim.run_simulation()
            print(f"Simulating scenario {x}")  # epyt_flow.data.scada_data.ScadaData
            ScadaDataXlsxExport(f"./data/scada_data_no_cont{x}.xlsx", export_raw_data=False).export(scada_data)
            ScadaDataNumpyExport(f"./data/scada_data_no_cont{x}.npz", export_raw_data=False).export(scada_data)

            os.makedirs("./plots", exist_ok=True)

            # Inspect simulation results -- i.e. sensor readings over time
            scada_data.plot_bulk_species_node_concentration({"Chlorine": cl_sensor_locations})
            plt.savefig(f"./plots/chlorine_concentration_no_cont{x}.png")
            scada_data.plot_bulk_species_node_concentration({"AsIII": all_nodes})
            plt.savefig(f"./plots/arsenic_concentration__no_cont{x}.png")
