import pandas
import numpy
from pypdf import PdfReader
import boto3
from langchain_aws import BedrockLLM
from langchain_community.chat_models import BedrockChat
from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_aws import ChatBedrock
from langchain_core.tools import tool

from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from langgraph.checkpoint import MemorySaver
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.chains import TransformChain



SKILLS = {}





# Amazon AWS LLMS

llm = BedrockLLM(
    model_id="mistral.mixtral-8x7b-instruct-v0:1",
    model_kwargs={"temperature": 0.3, "max_tokens":4076, "top_p":0.1,"top_k":1}
)



# Data Models
class Skill(BaseModel):
    skill: str = Field(..., description="Name of the skill")
    skill_explanation: str = Field(
        ..., 
        description="""explanation of skill demonstration using 
        context from the text and the context's contribution to the skill"""
        )

class Skills(BaseModel):
    skills: List[Skill] = Field(
        ..., 
        description="List of skills")



# Prompts
hard_skill_entity_extraction_prompt = """
You are a career counselor. Your task is to extract skill entities from the given text, which can be a resume or a job description.

Skill Entities:
Skills: Also called technical skills, these are job-specific and relevant to each position and seniority level. In other words, each position in every company will require unique hard skills.

TASK:
1. Perform a Part-of-Speech (POS) Tagging on the text.
2. Using the POS-tagged resume, perform Name Entity Recognition to identify all explicitly stated skill entities.
3. For each skill, provide an explanation of skill demonstration using context from the text and the context's contribution to the skill.
4. State the context as it appears in the resume, without extrapolating it with other experiences.

----------------------
Format Instruction:
{format_instructions}
----------------------

----------------------
Text: {text}
----------------------
"""

modify_skills_entity_extraction_prompt = """
TASK: 
1. Understand User Modified Skill Name,Modification justification and the modification action user wants to perform.
2. For add or modify Action , check if the skill_name already exists, if it does use the explanation to add to the current skill description.
2. For delete action, use the explanation to determine if the whole skills needs to be deleted or just parts of the current justification.

Only return the affected skill_name and skill_justification. 
If the whole skill is deleted , return an empty justification.
If the user input justification is not enough or enough infromation about the action is not provided, simply return an empty skill_name.

Only return the JSON. 
------------------------------- 
Format Instruction: 
{format_instructions}
------------------------------- 

------------------------------- 
user input skill: {skill_name}
user justification: {user_justification}
action: {action}
All Skills: {all_skills}
-------------------------------
"""



fix_format_instruction = """
--------------
{instructions}
--------------
Completion:
--------------
{completion}
--------------
Above, the Completion did not satisfy the constraints given in the Instructions.
Error:
--------------
{error}
--------------
Please try again. Please only respond with an answer that satisfies the constraints laid out in the Instructions.
Important:  Only correct the structural issues within the JSON format. Do not modify the existing data values themselves:
"""
def fix_chain_fun(inputs):    
    fix_prompt = PromptTemplate.from_template(fix_format_instruction)
    fix_prompt_str = fix_prompt.invoke({'instructions':inputs['instructions'],
                                        'completion':inputs['completion'],
                                        'error':inputs['error']}).text
    
    #print(fix_prompt_str)
    
    completion = llm.invoke(fix_prompt_str)

    # return {"completion": completion}
    
    return {"completion": completion}

fix_chain = TransformChain(
    input_variables = ["instructions", "completion", "error"],output_variables=["completion"], transform=fix_chain_fun
)
# TOOLS
# A tool consists of name of the tool, Description of the tool , A JSON schema defining the inputs to the tool and A function
@tool(parse_docstring=True)
def get_hard_skills(resume_text: str) ->  str:
    """Takes in User's Resume Text, extracts hard skills form resume , return a JSON of all extracted skills and their explanations using context from Resume.
    
    Args:
        resume_text: Resume String for skill extraction.
    """
    # Invoke the LLM
    parser = PydanticOutputParser(pydantic_object= Skills)
    
    fix_parser = OutputFixingParser(
        parser=parser,
        retry_chain=fix_chain,
        max_retries=2
    )

    prompt = PromptTemplate(
        template = hard_skill_entity_extraction_prompt, 
        input_variables=["text"],
        partial_variables= {"format_instructions": parser.get_format_instructions()})
    
    prompt_str = prompt.format(text=resume_text)

    print(prompt_str)
    
    response = llm.invoke(prompt_str)


    print(f"Response is : {response}")

    fixed_response = fix_parser.invoke(response).dict()

    for skill_set in fixed_response["skills"]:
        
        key = skill_set["skill"].lower()
        justification = skill_set["skill_explanation"]

        SKILLS[key] = justification

    return str(fixed_response) + "Skills were extracted successfully!"


@tool(parse_docstring=True)
def modify_skills(skill_name:str, user_explanation:str, action:str) ->  str:
    """Responsible for Adding, Modifying or Deleting skills from skill set extracted from the resume, only when explicitly asked by the user.Takes in a skill_name , user explanation for the action to be done on the skill and action.Actions can be add, delete or modify. 
    
    Args:
        skill_name: Name of the skill modification needs to be done on
        user_explanation: Explanation of the modification
        action: add or delete
    """
    
    # Invoke the LLM
    parser = PydanticOutputParser(pydantic_object= Skill)
    
    fix_parser = OutputFixingParser(
        parser=parser,
        retry_chain=fix_chain,
        max_retries=2
    )

    prompt = PromptTemplate(
        template = modify_skills_entity_extraction_prompt, 
        input_variables=["skill_name","user_justification", "all_skills"],
        partial_variables= {"format_instructions": parser.get_format_instructions()})
    
    prompt_str = prompt.format(skill_name=skill_name, user_justification=user_explanation, all_skills = SKILLS, action = action)

    print(prompt_str)
    
    response = llm.invoke(prompt_str)


    print(f"Response is : {response}")

    fixed_response = fix_parser.invoke(response).dict()


    if fixed_response["skill"] and fixed_response["skill_explanation"]:
        key = fixed_response["skill"].lower()
        justification = fixed_response["skill_explanation"]
        if key not in SKILLS:
            SKILLS[key] = justification
            return f"Skill: {skill_name} was Added Sucessfully."
        else:
            SKILLS[key] = justification
            return f"Skill: {skill_name} was Modified Sucessfully."
    if fixed_response["skill"] and not fixed_response["skill_explanation"]:
        key = fixed_response["skill"].lower()
        del SKILLS[key]
        return f"Skill: {skill_name} was Deleted Sucessfully"
    
        

    return "No Action was performed, please elaborate further."

get_hard_skills.args_schema.schema()
modify_skills.args_schema.schema()



# Agent Confirgutaiton
# Initialize Flask app


# Initialize memory and model
memory = MemorySaver()
model = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0", 
                    model_kwargs={"temperature": 0, "max_tokens": 6000, "top_p": 0.1, "top_k": 1})

# Initialize tools
tools = [get_hard_skills, modify_skills]

# Create agent
skills_extraction_agent = create_react_agent(model, tools, checkpointer=memory)

# Process the question with the agent
config = {"configurable": {"thread_id": "skills"}}

# Define Ask 
def ask_skills_extraction_agent(question):
    for chunk in skills_extraction_agent.stream({"messages": [HumanMessage(content=question)]}, config):
        answer = chunk
    return answer["agent"]["messages"][0].content


#Get Skills
def get_skills():
    return SKILLS

