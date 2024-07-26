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


ASPIRATION = ""
TARGET_SKILLS = {}
LEARNING_GOAL = {}
AWS_KB_ID = "OY4K3P2JUR"
client = boto3.client('bedrock-agent-runtime')


llm = BedrockLLM(
    model_id="mistral.mixtral-8x7b-instruct-v0:1",
    model_kwargs = {"temperature": 0, "max_tokens": 4000, "top_p": 0.1, "top_k": 1}
)


model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {"temperature": 0, "max_tokens": 4000, "top_p": 0.1, "top_k": 1}
chat = BedrockChat(model_id=model_id, model_kwargs=model_kwargs)

aspiration_extraction_prompt = """
You are a career consellor. 
You are translating client's aspiration into a SMART goal. 
Task: 
1. Understand SMART Goals' Definition and Criteria. 
2. Understand client Resume. 
3. Understand given example , with client aspiration translated into SMART goal. 
4. Translate client's aspiration into a SMART goal, explain the translation. 

Disclaimer: Translate the goals as it is , use the resume to infer information. 
------------------ 
Domain 
SMART Goal Defintion: Statements of the important results you are working to accomplish.Designed in a way to foster clear and mutual understanding of what constitutes expected levels of performance and successful professional development 
SMART Goal Ceriteria: 
    S-> Specific: What will be accomplished? What actions will you take? 
    M-> Measurable: What data will measure the goal? (How much? How well? 
    A->Achievable: Is the goal doable? Do you have the necessary skills and resources? 
    R->Relevant: How does the goal align with broader goals? Why is the result important? 
    T->Time-Bound: What is the time frame for accomplishing the goal? 

------------------
Resume: {resume}  
------------------ 

------------------
Example: 
Aspiration: I want to improve my performance  
GOAL: 
    Specific: I received low marks on my ability to use PowerPoint at my last performance review. Improving my skills requires that I learn how to use PowerPoint efficiently and practice using it by creating various presentations. I’d like to be more proficient using PowerPoint in time for my next review in six months. 
    Measurable: By the time of my next review, I should be able to create presentations that incorporate graphs, images, and other media in a couple of hours. I should also be able to efficiently use and create templates in PowerPoint that my coworkers can also use. 
    Achievable: Improving my PowerPoint skills is instrumental in moving forward in my career and receiving a better performance review. I can set time aside every week to watch PowerPoint tutorials and even enroll in an online class that can teach me new skills. I can also ask coworkers and my manager for PowerPoint tips. 
    Relevant: Working with PowerPoint is currently 25% of my job. As I move up in the company, I’ll need to spend 50% of my time creating PowerPoint presentations. I enjoy my career and want to continue to grow within this company. 
    Time-Bound: In six months, I should be proficient in PowerPoint ensuring it only occupies 25% of my workload instead of the nearly 40% of the time it occupies now. 

------------------ 

------------------ 
Aspiration: {aspiration}
------------------ 

------------------ 
format_instruction: {format_instructions} 
------------------

"""



skill_entity_extraction_prompt = """
You are a career counselor. Your task is to extract skill entities from the given text, which can be a job description.

Skill Entities:
Skills: Also called technical skills, these are job-specific and relevant to each position and seniority level. In other words, each position in every company will require unique hard skills.

TASK:
1. Perform a Part-of-Speech (POS) Tagging on the text.
2. Using the POS-tagged text, perform Name Entity Recognition to identify all explicitly stated skill entities.
3. For each skill, provide an explanation of skill demonstration using context from the text and the skill's contribution to the job description.
4. State the context as it appears in the text, without extrapolating it with other skills.

----------------------
Format Instruction:
{format_instructions}
----------------------

----------------------
Text: {text}
----------------------
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

# Data Models

class Goal(BaseModel):
    specific: str = Field(..., description="Specific: What will be accomplished? What actions will you take?")
    measurable: str = Field(..., description="Measurable: What data will measure the goal? (How much? How well?")
    achievable: str = Field(..., description="Achievable: Is the goal doable? Do you have the necessary skills and resources?")
    relevant: str = Field(..., description="Relevant: How does the goal align with broader goals? Why is the result important?")
    time_bound: str = Field(..., description="Time-Bound: What is the time frame for accomplishing the goal?")
    explanation: str = Field(..., description="Explanation of the translation of the aspiration into a SMART goal")

class Skill(BaseModel):
    skill: str = Field(..., description="Name of the skill")
    skill_explanation: str = Field(..., description="explanation of skill demonstration using context from the text and the context's contribution to the skill")#

class Skills(BaseModel):
    skills: List[Skill] = Field(..., description="List of skills")



#UTILS
def fix_chain_fun(inputs):    
    fix_prompt = PromptTemplate.from_template(fix_format_instruction)
    fix_prompt_str = fix_prompt.invoke({'instructions':inputs['instructions'],
                                        'completion':inputs['completion'],
                                        'error':inputs['error']}).text
    
    #print(fix_prompt_str)
    
    completion = chat.invoke(fix_prompt_str).content

    return {"completion": completion}

fix_chain = TransformChain(
    input_variables = ["instructions", "completion", "error"],output_variables=["completion"], transform=fix_chain_fun
)

def extract_skills_context(job_desc):
    parser = PydanticOutputParser(pydantic_object= Skills)
    
    fix_parser = OutputFixingParser(
        parser=parser,
        retry_chain=fix_chain,
        max_retries=2
    )

    prompt = PromptTemplate(
        template = skill_entity_extraction_prompt, 
        input_variables=["text"],
        partial_variables= {"format_instructions": parser.get_format_instructions()})
    
    prompt_str = prompt.format(text=job_desc)

    print(prompt_str)
    
    response = llm.invoke(prompt_str)


    print(f"Response is : {response}")

    fixed_response = fix_parser.invoke(response).dict()

    for skill_set in fixed_response["skills"]:
        
        key = skill_set["skill"].lower()
        justification = skill_set["skill_explanation"]

        TARGET_SKILLS[key] = justification

    return str(fixed_response) + "Skills were extracted successfully!"

#TOOLS
@tool(parse_docstring=True)
def extract_smart_goal(aspiration: str, resume: str) -> Goal:
    """Generates a SMART goal from the user's aspiration and resume.Takes User Aspiration and Resume as input and returns a SMART goal. 
       Do Not run the function if either resume or aspiration is not provided.Do not assume aspiration from resume.

    Args:
        aspiration: User's Aspiration
        resume: User's Resume
    """

    if aspiration == "" or resume == "":
        return "Please provide both an aspiration and a resume."
    parser = PydanticOutputParser(pydantic_object=Goal)
    
    fix_parser = OutputFixingParser(
        parser=parser,
        retry_chain=fix_chain,
        max_retries=2
    )

    prompt = PromptTemplate(
        template = aspiration_extraction_prompt, 
        input_variables=["aspiration", "resume"],
        partial_variables= {"format_instructions": parser.get_format_instructions()})
    
    prompt_str = prompt.format(aspiration=aspiration, resume=resume)

    print(prompt_str)
    
    response = chat.invoke(prompt_str).content
 

    print(f"Response is : {response}")

    fixed_response = fix_parser.invoke(response).dict()

    for key in fixed_response.keys():
        LEARNING_GOAL[key] = fixed_response[key]

    return fixed_response

# A tool consists of name of the tool, Description of the tool , A JSON schema defining the inputs to the tool and A function

@tool(parse_docstring=True)
def get_skills(specific_goal: str) ->  str:
    """Runs on User's explicit instruction to extracts skills from the learning goal.Take's in the specific goal and returns the extracted skills.
    
    Args:
        specific_goal: Resume String for skill extraction.
    """
    # Invoke the LLM
    context =  client.retrieve(
    knowledgeBaseId=AWS_KB_ID,
     retrievalQuery={
        'text': specific_goal,
    }
)
    responses = ""
    count = 0
    for result in context["retrievalResults"]:
        job_desc = result["content"]["text"]
        print(job_desc)
        responses += extract_skills_context(job_desc)
        count += 1
        if count >= 3:
            break
    
    return responses





memory = MemorySaver()
model = ChatBedrock(model_id ="anthropic.claude-3-haiku-20240307-v1:0", model_kwargs={"temperature": 0, "max_tokens":6000, "top_p":0.1,"top_k":1})
tools = [extract_smart_goal, get_skills]
aspiration_extraction_agent = create_react_agent(model, tools, checkpointer=memory)


config = {"configurable": {"thread_id": "aspiration"}}
# Define Ask 
def ask_aspiration_extraction_agent(question):
    for chunk in aspiration_extraction_agent.stream({"messages": [HumanMessage(content=question)]}, config):
        answer = chunk
    return answer["agent"]["messages"][0].content


# get Target Skills
def get_target_skills():
    return TARGET_SKILLS

# get Learning Goal
def get_learning_goal():
    return LEARNING_GOAL


