from bin_factory import loader, transforms
from bin_factory.log_context import log


def prepare_scenario(py123d_data, config):
    scenario, extras = loader.extract_scenario(
        py123d_data, config.scenario_id_field, config.speed_limit_source, config.static_objects
    )
    # Store the nominal sampling step rather than py123d's measured mean.
    if scenario.metadata.scenario_length:
        if abs(scenario.metadata.dt - config.dt) > 0.05 * config.dt:
            raise ValueError(f"Scene step {scenario.metadata.dt:.4f} s is not the requested --dt {config.dt} s")
        scenario.metadata.dt = config.dt

    _validate(scenario, config.validate_level, "Validation failed", extras)
    transforms.run(scenario, extras, config)
    _validate(scenario, config.validate_level, "Post-transform validation failed")
    return scenario


def _validate(scenario, level, failure, extras=None):
    if level <= 0:
        return
    errors = loader.validate_scenario(scenario, extras=extras, level=level)
    for error in errors:
        log.error(f"{scenario.metadata.id}: {error}")
    if errors:
        raise loader.ValidationError(f"{failure} for scenario {scenario.metadata.id} with {len(errors)} errors")
