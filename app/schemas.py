from typing import Dict, Optional
from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Patient's description of symptoms")
    chief_complaint: Optional[str] = Field(
        default="", description="Short main complaint (optional, e.g. 'chest pain')"
    )


class TriageResponse(BaseModel):
    triage_level: str
    confidence: float
    probabilities: Dict[str, float]
    red_flag_override: bool
    recommendation: str
    disclaimer: str
