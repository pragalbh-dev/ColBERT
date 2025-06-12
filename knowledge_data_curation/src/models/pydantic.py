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

# # New model for industry aspect extraction
# class IndustryAspects(BaseModel):
#     industry: List[str]
#     target_audience: List[str] 
#     technology_used: List[str]
#     products_solutions: List[str]
#     business_model: List[str]
#     revenue_model: List[str]


from pydantic import BaseModel, Field, ConfigDict
from typing import List

class IndustryAspects(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    industry: List[str] = Field(
        description="The industry or industries relevant to the business."
    )
    target_audience: List[str] = Field(
        description="The main target audiences for the business solutions."
    )
    technology_used: List[str] = Field(
        description="Technologies utilized by the business."
    )
    products_solutions: List[str] = Field(
        description="A list of products and solutions offered by the business."
    )
    business_model: List[str] = Field(
        description="The business models that the company operates under."
    )
    revenue_model: List[str] = Field(
        description="The various revenue models used by the business."
    )