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


TRAINING_PLAN = {}


model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {"temperature": 0.2, "max_tokens": 4000,"top_p": 0.1, "top_k": 1}

llm = BedrockChat(
    model_id=model_id,
    model_kwargs=model_kwargs
)


# Data Models
from typing import List
from pydantic import BaseModel, Field

class ShortTermGoal(BaseModel):
    learning_outcomes: str = Field(..., description="Learning outcomes achieved at the end of the goal")
    specific: str = Field(..., description="Specific objective")
    measurable: str = Field(..., description="Measurable criteria to evaluate progress and success")
    achievable: str = Field(..., description="Consider the resources available and the learners’ current capabilities when setting objectives")
    relevant: str = Field(..., description="Align the learning objectives with the overall goals of the course and the individual learner’s needs")
    time_bound: str = Field(..., description="Set clear deadlines for each objective")

class LearningPlan(BaseModel):
    short_term_goals: List[ShortTermGoal] = Field(..., description="List of short-term goals")


# Prompts
learning_plan_creation_prompt = """
You are a Career Counselor. 
Your task is to generate a learning plan with atleast 5 or more short-term SMART goals pertaining to the client's long-term SMART goal. 

You will be given: 
1. Skills and their explanations Extacted from Client's Resume, these will be called client's skills. 
2. Skills and their explanations Extracted from Job Descriptions about the Client's long-term SMART goal, these will be called target skills. 
3. Client's long-term SMART goal. 

Considerations: 
1. All short-term goals should contribute to the long-term goal. 
2. If there exists a pre-requisite for short term goal y , either the pre-requisite should be filled by Client's skills or a previous goal x. 
3. All goals should be personalized to Client's skill set and gaps between client's skills and target skills. 
4. One short-term goal may encapsulate, one or more gaps between client's skills and target skills but the overall goal should be achievable in the time frame required by the short-term goal. 
5. Short-term goal cumulative timeline should sum up to the time given in the long-term goal. If some goals are parallel, that should be mentioned in the timeline.
6. It is okay to have many short-term goals as long as they are SMART and contribute to the long-term goal.
7. Mention the estimated weekly commitment required for each short-term goal. For Example,3 days per week etc.

Steps for short-term goal formulation, For Each goal: 
1. Indetify the desired learning outcomes achieved at the end of the goal. 
2. Be Specific, Use action verbs to precisely define what the learner will achieve. Clearly state the scope of the objective to eliminate ambiguity. 
3. Ensure Objective are measurable, Define clear, quantifiable criteria to evaluate progress and success.Be very transparent upfront about what constitutes an excellent, good, or acceptable “pass mark.” 
4. Ensure Objectives are Acheivable, Consider the resources available and the learners’ current capabilities when setting objectives. This ensures that the goals are challenging but within reach. 
5. Ensure Objectives are Relevant/Personalized, Align the learning objectives with the overall goals of the course and the individual learner’s needs. This ensures the learning process is valuable and beneficial for the learner. 
6. Establish Deadlines/Timeframes: Set clear deadlines for each objective. Balance the time constraints with the scope of the objective to ensure it’s feasible within the given timeframe. For Each goal, Return a brief explanation about it's contribution to the overall long-term goal and it's relation to with previously completed goals or the skills given in the resume. For Each goal, Return a brief explanation about it's contribution to the overall long-term goal . 

----------------------------- 
long-term SMART goal: {long_term_goal}
client skills: {client_skills} 
target skills: {target_skills}
-----------------------------

Only return the JSON. 
------------------------------- 
Format Instruction: 
{format_instructions}
------------------------------- 
"""

learning_plan_feedback_prompt = """
You are a Career Counselor. 
Your task is to generate a learning plan with atleast 5 or more short-term SMART goals pertaining to the client's long-term SMART goal. 

You will be given: 
1. Skills and their explanations Extacted from Client's Resume, these will be called client's skills. 
2. Skills and their explanations Extracted from Job Descriptions about the Client's long-term SMART goal, these will be called target skills. 
3. Client's long-term SMART goal. 
4. A JSON containing the previous learning plan created by the user.
5. Client's Feedback on the previous learning plan.

Considerations: 
1. All short-term goals should contribute to the long-term goal. 
2. If there exists a pre-requisite for short term goal y , either the pre-requisite should be filled by Client's skills or a previous goal x. 
3. All goals should be personalized to Client's skill set and gaps between client's skills and target skills. 
4. One short-term goal may encapsulate, one or more gaps between client's skills and target skills but the overall goal should be achievable in the time frame required by the short-term goal. 
5. Short-term goal cumulative timeline should sum up to the time given in the long-term goal. If some goals are parallel, that should be mentioned in the timeline.
6. It is okay to have many short-term goals as long as they are SMART and contribute to the long-term goal.
7. Mention the estimated weekly commitment required for each short-term goal. For Example,3 days per week etc.
8. Use Client's feedback to make necessary changes to the learning plan. return the updated learning plan.
9. If the feedback is not valid, return the original learning plan and request the user to make necessary changes to the feedback.

Steps for short-term goal formulation, For Each goal: 
1. Indetify the desired learning outcomes achieved at the end of the goal. 
2. Be Specific, Use action verbs to precisely define what the learner will achieve. Clearly state the scope of the objective to eliminate ambiguity. 
3. Ensure Objective are measurable, Define clear, quantifiable criteria to evaluate progress and success.Be very transparent upfront about what constitutes an excellent, good, or acceptable “pass mark.” 
4. Ensure Objectives are Acheivable, Consider the resources available and the learners’ current capabilities when setting objectives. This ensures that the goals are challenging but within reach. 
5. Ensure Objectives are Relevant/Personalized, Align the learning objectives with the overall goals of the course and the individual learner’s needs. This ensures the learning process is valuable and beneficial for the learner. 
6. Establish Deadlines/Timeframes: Set clear deadlines for each objective. Balance the time constraints with the scope of the objective to ensure it’s feasible within the given timeframe. For Each goal, Return a brief explanation about it's contribution to the overall long-term goal and it's relation to with previously completed goals or the skills given in the resume. For Each goal, Return a brief explanation about it's contribution to the overall long-term goal . 

----------------------------- 
long-term SMART goal: {long_term_goal}
client skills: {client_skills} 
target skills: {target_skills}
feedback: {feedback}
-----------------------------

Only return the JSON. 
------------------------------- 
Format Instruction: 
{format_instructions}
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


# UTILS
def fix_chain_fun(inputs):
    fix_prompt = PromptTemplate.from_template(fix_format_instruction)
    fix_prompt_str = fix_prompt.invoke({'instructions':inputs['instructions'],
                                        'completion':inputs['completion'],
                                        'error':inputs['error']}).text
    
    #print(fix_prompt_str)
    
    completion = llm.invoke(fix_prompt_str).content

    # return {"completion": completion}
    
    return {"completion": completion}

fix_chain = TransformChain(
    input_variables = ["instructions", "completion", "error"],output_variables=["completion"], transform=fix_chain_fun
)


# TOOLS

@tool(parse_docstring=True)
def generate_learning_plan(long_term_goal: str, client_skills: str, target_skills: str) -> str:
    """Takes a long-term goal, client's skills, and target skills and generates a learning plan with short-term SMART goals.
    
    Args:
        long_term_goal: The long-term SMART goal of the client.
        client_skills: ALL of the skills and their explanations extracted from the client's resume.
        target_skills: All of the target skills and their explanations extracted from job descriptions about the client's long-term SMART goal.
    """
    
    # Inputs
    inputs = ["long_term_goal", "client_skills", "target_skills"]
    # Define the parser
    parser = PydanticOutputParser(pydantic_object=LearningPlan)

    fix_parser = OutputFixingParser(parser=parser, retry_chain=fix_chain, max_retries=2)

    # Define the chain
    prompt = PromptTemplate(
        template=learning_plan_creation_prompt,
        inputs=inputs,
        partial_variables={"format_instructions":parser.get_format_instructions()}  
    )

    prompt_str = prompt.format(long_term_goal=long_term_goal, client_skills=client_skills, target_skills=target_skills)
    print(f"prompt is {prompt_str}")
    response = llm.invoke(prompt_str).content

    
    fixed_response = fix_parser.invoke(response).dict()

    
    for key, value in fixed_response.items():
        TRAINING_PLAN[key] = value

    return fixed_response

@tool(parse_docstring=True)
def edit_learning_plan(long_term_goal: str, client_skills: str, target_skills: str, learning_plan: str, feedback: str) -> str:
    """Responsible for editing the learning plan using user feedback or comments and previous learning plan. Takes a long-term goal, client's skills,target skills, previous learning plan and feedback and generates a new learning plan.
    Feedback can include actions. For Example "Change the plan" or "Add more goals" or "Remove goal 2" etc.
    
    Args:
        long_term_goal: The long-term SMART goal of the client.
        client_skills: ALL of the skills and their explanations extracted from the client's resume.
        target_skills: All of the target skills and their explanations extracted from job descriptions about the client's long-term SMART goal.
        learning_plan: The whole previous learning plan created by the user.
        feedback: The feedback given by the user on the previous learning plan.
    """
    
    # Inputs
    inputs = ["long_term_goal", "client_skills", "target_skills", "learning_plan", "feedback"]
    # Define the parser
    parser = PydanticOutputParser(pydantic_object=LearningPlan)

    fix_parser = OutputFixingParser(parser=parser, retry_chain=fix_chain, max_retries=2)

    # Define the chain
    prompt = PromptTemplate(
        template=learning_plan_creation_prompt,
        inputs=inputs,
        partial_variables={"format_instructions":parser.get_format_instructions()}  
    )

    prompt_str = prompt.format(long_term_goal=long_term_goal, client_skills=client_skills, target_skills=target_skills, learning_plan=learning_plan, feedback=feedback)
    print(f"prompt is {prompt_str}")
    response = llm.invoke(prompt_str).content

    print(response)
    fixed_response = fix_parser.invoke(response).dict()

    for key, value in fixed_response.items():
        TRAINING_PLAN[key] = value

    return fixed_response


generate_learning_plan.args_schema.schema()

learning_outcomes = {}
# memory = SqliteSaver.from_conn_string(":memory:")
memory = MemorySaver()
model = ChatBedrock(model_id ="anthropic.claude-3-haiku-20240307-v1:0", model_kwargs={"temperature": 0, "max_tokens":6000, "top_p":0.1,"top_k":1})
tools = [generate_learning_plan, edit_learning_plan]
agent_executor = create_react_agent(model, tools, checkpointer=memory)



config = {"configurable": {"thread_id": "training"}}
# Define Ask 
def ask_training_plan_generation_agent(question):
    for chunk in agent_executor.stream({"messages": [HumanMessage(content=question)]}, config):
        answer = chunk
    return answer["agent"]["messages"][0].content


#Get Skills
def get_training_plan():
    return TRAINING_PLAN



