from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.core_output_parsers import PydanticOutputParsers
from pydantic import BaseModel,Field


load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)


model=ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name:str = Field(description='the name of a person')
    age:int=Field(gt=18,description='teh age of the person')

parser=PydanticOutputParsers(pydantic_object=Person)



template=PromptTemplate(
    template='write the name,age of the fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)

chain=template|model|parser
result=chain.invoke({'place':'Indian'})
print(result)
