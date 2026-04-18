from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParsers,ResponseSchemas

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

schema=[
    ResponseSchemas(name='fact1',description='the fact 1 of the topic'),
    ResponseSchemas(name='fact2',description='the fact 2 of the topic'),
    ResponseSchemas(name='fact3',description='the fact 3 of the topic')
]

parser=StructuredOutputParsers.from_respense_schemas(schema)

template=PromptTemplate(
    template='write down the facts about the {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}

)

chain=template|model|parser
result=chain.invoke({'topic':"black hole"})
print(result)
