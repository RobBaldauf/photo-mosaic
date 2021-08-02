from pydantic import BaseModel


class Segment(BaseModel):
    """Model for storing detailed segment information, needed for segment sampling and filling"""

    id: str
    mosaic_id: str
    row_idx: int
    col_idx: int
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    brightness: int
    fillable: bool
    filled: bool
    is_start_segment: bool
