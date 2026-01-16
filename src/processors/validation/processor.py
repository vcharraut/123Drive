"""
Validation processor for PufferDrive dict format.

Validation is read-only - puffer_dict is returned unchanged.
"""

from src import logger_utils
from src.processors.validation import validate_puffer


logger = logger_utils.get_logger(__name__)


def validate_puffer_scenario(
    puffer_dict: dict,
    strict: bool = True,
    fail_on_error: bool = False,
    validation_level: int = 2,
    position_jump_threshold: float = 50.0,
    velocity_tolerance: float = 2.0,
    heading_tolerance_deg: float = 30.0,
    **kwargs,
) -> dict:
    """
    Validate PufferDrive dict structure and physics.

    Read-only processor - returns puffer_dict unchanged.

    Args:
        puffer_dict: The puffer dict to validate
        strict: If True, perform physics checks. If False, structure only.
        fail_on_error: If True, raise exception on validation failure
        validation_level: Strictness level for physics checks (1-4)
        position_jump_threshold: Max position jump between timesteps (meters)
        velocity_tolerance: Tolerance for velocity-position consistency (m/s)
        heading_tolerance_deg: Tolerance for heading-velocity alignment (degrees)

    Returns:
        The original puffer_dict unchanged

    Raises:
        ValueError: If fail_on_error=True and validation fails
    """
    scenario_id = puffer_dict.get("scenario_id", "unknown")

    # Soft validation (structure)
    is_valid, errors, warnings = validate_puffer.soft_validate(puffer_dict)

    # Strict validation (physics) if requested and soft passed
    if strict and is_valid:
        strict_valid, strict_errors, strict_warnings = validate_puffer.strict_validate(
            puffer_dict,
            validation_level=validation_level,
            position_jump_threshold=position_jump_threshold,
            velocity_tolerance=velocity_tolerance,
            heading_tolerance_deg=heading_tolerance_deg,
        )
        is_valid = strict_valid
        errors.extend(strict_errors)
        warnings.extend(strict_warnings)

    validation_type = "strict" if strict else "soft"

    if is_valid:
        logger.debug(f"Scenario {scenario_id}: {validation_type} validation passed")
    else:
        logger.warning(f"Scenario {scenario_id}: {validation_type} validation FAILED with {len(errors)} error(s)")
        for i, error in enumerate(errors[:10]):
            logger.warning(f"  Error: {error}")
        if len(errors) > 10:
            logger.warning(f"  ... and {len(errors) - 10} more errors")

    if warnings:
        logger.debug(f"Scenario {scenario_id}: {len(warnings)} warning(s)")
        for warning in warnings[:5]:
            logger.debug(f"  Warning: {warning}")

    if fail_on_error and not is_valid:
        raise ValueError(f"Scenario {scenario_id} validation failed: {errors[0] if errors else 'unknown'}")

    return puffer_dict
