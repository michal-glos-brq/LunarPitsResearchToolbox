"""
This file implements pydantic models for mongo collections containing Lunar Pit Atlas data.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator


def safe_cast(value, cast_type, default=None):
    """
    Safely cast a value to a specified type.
    Returns the value cast to `cast_type` or `default` if casting fails.
    """
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return default


def custom_bool(value: Optional[str]) -> Optional[bool]:
    """
    Convert a string to a boolean. Returns None if conversion fails.
    """
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if value in {"y", "yes"}:
        return True
    elif value in {"n", "no"}:
        return False
    return None


class ImageMongoObject(BaseModel):
    emission_angle: Optional[float] = None
    image_path: str
    inc_angle: Optional[float] = None
    line: Optional[int] = None
    object: str
    resolution: Optional[float] = None
    sample: Optional[int] = None
    time: Optional[float] = None
    title: str

    @field_validator("emission_angle", "inc_angle", "resolution", "time", mode="before")
    def validate_float(cls, v):
        return safe_cast(v, float)

    @field_validator("line", "sample", mode="before")
    def validate_int(cls, v):
        return safe_cast(v, int)


class PitDetailsMongoObject(BaseModel):
    azimuth: Optional[int] = None
    beside_dome: Optional[bool] = None
    best_stereo_left: Optional[str] = None
    best_stereo_right: Optional[str] = None
    clipping_width: Optional[int] = None
    colinear_depressions: Optional[bool] = None
    colinear_fractures: Optional[bool] = None
    colinear_pits: Optional[bool] = None
    data_warning: Optional[str] = None
    depth: Optional[float] = None
    depth_1: Optional[float] = None
    depth_2: Optional[float] = None
    depth_3: Optional[float] = None
    desc: Optional[str] = None
    dtms: Optional[str] = None
    entrance_ramp: Optional[bool] = None
    funnel_max_diam: Optional[int] = None
    funnel_min_diam: Optional[int] = None
    host_feature: Optional[str] = None
    in_depression: Optional[bool] = None
    in_dome: Optional[bool] = None
    inner_max_diam: Optional[int] = None
    inner_min_diam: Optional[int] = None
    lat: Optional[float] = None
    long: Optional[float] = None
    name: str
    location: Optional[dict] = None  # GeoJSON Point
    nearby_depressions: Optional[bool] = Field(alias="nearby_depressions?")
    nearby_fractures: Optional[bool] = Field(alias="nearby_fractures?")
    nearby_pits: Optional[bool] = Field(alias="nearby_pits?")
    origin: Optional[str] = None
    other_imgs: Optional[str] = None
    overhang: Optional[bool] = None
    terrain: Optional[str] = None
    types: Optional[str] = None

    def __init__(self, **data):
        if data.get("lat") is not None and data.get("long") is not None:
            long = ((float(data.pop("long")) + 180) % 360) - 180
            data["location"] = {"type": "Point", "coordinates": [long, float(data.get("lat"))]}
        super().__init__(**data)

    # Field Validators
    @field_validator("lat", "long", "depth", "depth_1", "depth_2", "depth_3", mode="before")
    def validate_float(cls, v):
        return safe_cast(v, float)

    @field_validator(
        "clipping_width", "funnel_max_diam", "funnel_min_diam", "inner_max_diam", "inner_min_diam", mode="before"
    )
    def validate_int(cls, v):
        return safe_cast(v, int)

    @field_validator(
        "beside_dome",
        "colinear_depressions",
        "colinear_fractures",
        "colinear_pits",
        "entrance_ramp",
        "in_depression",
        "in_dome",
        "nearby_depressions",
        "nearby_fractures",
        "nearby_pits",
        "overhang",
        mode="before",
    )
    def validate_bool(cls, v):
        return custom_bool(v)


class PitsMongoObject(BaseModel):
    azimuth: Optional[int] = None
    depth: Optional[float] = None
    depth_sorting: Optional[float] = None
    funnel_max_diameter: Optional[int] = None
    funnel_min_diameter: Optional[int] = None
    hosting_feature: str
    inner_max_diameter: Optional[int] = None
    inner_max_diameter_sorting: Optional[int] = None
    inner_min_diameter: Optional[int] = None
    inner_min_diameter_sorting: Optional[int] = None
    latitude: Optional[float] = None
    link_suffix: Optional[str] = None
    longitude: Optional[float] = None
    name: str
    location: Optional[dict] = None  # GeoJSON Point

    def __init__(self, **data):
        if data.get("latitude") is not None and data.get("longitude") is not None:
            # Correct for nonstandard longitude values
            longitude = ((float(data.pop("longitude")) + 180) % 360) - 180
            data["location"] = {"type": "Point", "coordinates": [longitude, float(data.get("latitude"))]}
        super().__init__(**data)

    # Field Validators
    @field_validator("latitude", "longitude", "depth", "depth_sorting", mode="before")
    def validate_float(cls, v):
        return safe_cast(v, float)

    @field_validator(
        "azimuth",
        "inner_max_diameter",
        "inner_min_diameter",
        "inner_max_diameter_sorting",
        "inner_min_diameter_sorting",
        "funnel_max_diameter",
        "funnel_min_diameter",
        mode="before",
    )
    def validate_int(cls, v):
        return safe_cast(v, int)
