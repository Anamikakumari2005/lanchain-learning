from operator import gt

from pydantic import BaseModel, Field
from typing import Optional

class student(BaseModel):
    
    name:str ='mika'
    age:Optional[int] = None
    cgpa : float = Field(gt=0,lt=10,default=5,description="CGPA must be between 0 and 10")
    
new_student={'age':90,'cgpa':9.5}   


ans = student(**new_student)
ans_dict = dict(ans) 

print(ans_dict['age'])