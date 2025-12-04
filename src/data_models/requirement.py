# requirement.py
from pydantic import BaseModel, RootModel, Field, field_validator
from typing import Optional, List

class Requirement(BaseModel):
    """Information about a requirement. """
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    code: Optional[str] = Field(default=None, description="The code of the requirement")
    description: Optional[str] = Field(default=None, description="The description of the requirement")
"""
    @field_validator('code')
    def validate_code(cls, v):
        if not v.strip():
            raise ValueError("Code can't be empty")
        return v.strip()
    
    @field_validator('description')
    def validate_description(cls, v):
        if not v.strip():
            raise ValueError("Description can't be empty")
        return v.strip()
"""
class RequirementList(RootModel):
    root: List[Requirement]  
    def __iter__(self):
        #Allow iteration over requirements
        return iter(self.root)
    
    def __len__(self):
        #Get count of requirements
        return len(self.root)
    
    def __getitem__(self, index):
        #Allow indexing
        return self.root[index]
    
    def is_empty(self) -> bool:
        #Check is no requirements found
        return len(self.root) == 0
    
    def get_codes(self) -> List[str]:
        #Get all requirements codes
        return [req.code for req in self.root]
    
    def merge(self, other: 'RequirementList') -> 'RequirementList':
        # Merge one RequirementList with another
        combined_requirements = list(self.root) + list(other.root)
        return RequirementList(combined_requirements)
    
    """
    @field_validator('root')
    def validate_unique_codes(cls, v):
        # Ensure all requirement codes are unique
        codes = [req.code for req in v]
        if len(codes) != len(set(codes)):
            raise ValueError("Requirement codes must be unique")
        return v
"""  