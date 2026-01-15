def new_unified_scenario(scenario_id: str, dataset_name: str) -> dict:
    return {
        "id": scenario_id,
        "dynamic_agents": {},
        "static_map_elements": {},
        "dynamic_map_elements": {},
        "metadata": {
            "dataset_name": dataset_name,
            "scenario_length": 0,
            "sdc_index": 0,
            "timesteps": [],
        },
    }
