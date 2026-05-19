from .base import FormatSpec
from .full_length import FULL_LENGTH_SPEC
from .shorts import SHORTS_SPEC

FORMATS: dict[str, FormatSpec] = {
    "full_length": FULL_LENGTH_SPEC,
    "shorts": SHORTS_SPEC,
}


def get_format_spec(fmt: str) -> FormatSpec:
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format '{fmt}'. Must be one of: {list(FORMATS)}")
    return FORMATS[fmt]
