from pydantic import BaseModel
from typing import List

class NegativeSamplingResponse(BaseModel):
    hard_negatives: List[str]
    soft_negatives: List[str]

class SyntheticFactsheet(BaseModel):
    company_id: str
    factsheet: str

# If you want to support batch factsheet generation as a list of SyntheticFactsheet, you can use:
class SyntheticFactsheetBatch(BaseModel):
    factsheets: List[SyntheticFactsheet]
