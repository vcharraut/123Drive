"""
Validation module for PufferDrive dict format.

Provides:
- Soft validation: Structural checks (keys, types, shapes)
- Strict validation: Physics-based checks (trajectory coherence, constraints)
- Processor wrapper: Pipeline integration

Usage:
    from src.processors.validation import soft_validate, strict_validate
    is_valid, errors, warnings = soft_validate(puffer_dict)
    is_valid, errors, warnings = strict_validate(puffer_dict, validation_level=2)

    from src.processors.validation import validate_puffer_scenario
    puffer_dict = validate_puffer_scenario(puffer_dict, strict=True)
"""

from src.processors.validation.processor import validate_puffer_scenario
from src.processors.validation.validate_puffer import soft_validate, strict_validate


__all__ = [
    "soft_validate",
    "strict_validate",
    "validate_puffer_scenario",
]
