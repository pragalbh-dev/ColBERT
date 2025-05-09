from pydantic import BaseModel
from typing import List
class NegativeSamplingResponse(BaseModel):
    hard_negatives: List[str]
    soft_negatives: List[str]
