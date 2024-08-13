# Import necessary libraries
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import frontend as st
import requests

# Global varibales

SKILLS_EXTRACTION_START = """
Extract Skills from my resume: 
{Ng Jia Yin
Email: ng.jiayin@outlook.com | Mobile: 90501881 linkedin.com/in/jia-yin-ng/ | github.com/ngjiayin
   EDUCATION
  National University of Singapore (NUS)
Bachelor of Science (Honours) in Data Science and Analytics
Aug 2021 – Present
• Expected Date of Graduation: Dec 2025
• Relevant Coursework: Data Visualisation, Decision Trees for Machine Learning, Artificial Intelligence:
Technology and Impact, Data Structures and Algorithms, Mathematical Statistics, Multivariable Calculus
WORK EXPERIENCE
Johnson & Johnson, Analytics Intern Jan 2024 – May 2024
•
•
• Utilised design thinking methodologies to develop a 90-second brand launch video for an in-house data analytics platform, enhancing brand engagement and visibility.
KPMG, Tax Processor Intern Jan 2021 – Jun 2021
• Prepared and filed over 100 personal income tax returns using Microsoft Excel and internal tax software over the span of 5 months.
• Communicated effectively with clients and team members to identify and obtain necessary financial records in compliance with IRAS regulations.
PROJECTS
Lego Data Visualisation Nov 2023
• Utilised dplyr and ggplot2 packages to generate 3 visualisations (line plot, bar graph, scatter plot) to analyse the evolution of Lego set colours from 1949 to 2022.
• Pre-processed and cleaned 5 csv files and presented the findings and discussion in a Rmd file.
Stock Price Prediction Oct 2022
• Implemented various regression methods in Python to predict stock prices, achieving a minimal mean- squared error (MSE) of 3.07%.
• Conducted feature engineering and employed Principal Component Analysis (PCA) for exploratory analysis.
TECHNICAL SKILLS
• Programming and Database Languages: Python, R, SQL and Java
• Python Libraries: NumPy, pandas (data manipulation), Matplotlib, seaborn (data visualization), scikit-learn,
TensorFlow, PyTorch (machine learning), OpenCV (image processing)
• R Libraries: tidyverse, dplyr, ggplot2
• Other Tools: Jupyter Notebooks, Tableau, Microsoft Excel
CO-CURRICULAR ACTIVITIES
Workshop Executive, NUS Statistics and Data Science Society Jul 2023 – Present
• Collaborate with team members to create engaging presentation slides and Python code for Data Cleaning, Statistical Testing, Image Processing and Machine Learning workshops.
• Deliver insightful data science content to more than 150 NUS students, effectively conveying complex concepts and fostering a conducive learning environment.
   Managed end-to-end financial forecasting for the Long Range Financial Plan (LRFP) spanning from 2024
to 2028, managing sales figures totalling more than $4 billion.
  Implemented data validation and cleaning processes using Microsoft Excel, resulting in a 15% reduction in
data errors.
    
"""

ASPIRATION_EXTRACTION_START = """
I wanna be a better Data Scientist. context: Ng Jia Yin
Email: ng.jiayin@outlook.com | Mobile: 90501881 linkedin.com/in/jia-yin-ng/ | github.com/ngjiayin
   EDUCATION
  National University of Singapore (NUS)
Bachelor of Science (Honours) in Data Science and Analytics
Aug 2021 – Present
• Expected Date of Graduation: Dec 2025
• Relevant Coursework: Data Visualisation, Decision Trees for Machine Learning, Artificial Intelligence:
Technology and Impact, Data Structures and Algorithms, Mathematical Statistics, Multivariable Calculus
WORK EXPERIENCE
Johnson & Johnson, Analytics Intern Jan 2024 – May 2024
•
•
• Utilised design thinking methodologies to develop a 90-second brand launch video for an in-house data analytics platform, enhancing brand engagement and visibility.
KPMG, Tax Processor Intern Jan 2021 – Jun 2021
• Prepared and filed over 100 personal income tax returns using Microsoft Excel and internal tax software over the span of 5 months.
• Communicated effectively with clients and team members to identify and obtain necessary financial records in compliance with IRAS regulations.
PROJECTS
Lego Data Visualisation Nov 2023
• Utilised dplyr and ggplot2 packages to generate 3 visualisations (line plot, bar graph, scatter plot) to analyse the evolution of Lego set colours from 1949 to 2022.
• Pre-processed and cleaned 5 csv files and presented the findings and discussion in a Rmd file.
Stock Price Prediction Oct 2022
• Implemented various regression methods in Python to predict stock prices, achieving a minimal mean- squared error (MSE) of 3.07%.
• Conducted feature engineering and employed Principal Component Analysis (PCA) for exploratory analysis.
TECHNICAL SKILLS
• Programming and Database Languages: Python, R, SQL and Java
• Python Libraries: NumPy, pandas (data manipulation), Matplotlib, seaborn (data visualization), scikit-learn,
TensorFlow, PyTorch (machine learning), OpenCV (image processing)
• R Libraries: tidyverse, dplyr, ggplot2
• Other Tools: Jupyter Notebooks, Tableau, Microsoft Excel
CO-CURRICULAR ACTIVITIES
Workshop Executive, NUS Statistics and Data Science Society Jul 2023 – Present
• Collaborate with team members to create engaging presentation slides and Python code for Data Cleaning, Statistical Testing, Image Processing and Machine Learning workshops.
• Deliver insightful data science content to more than 150 NUS students, effectively conveying complex concepts and fostering a conducive learning environment.
   Managed end-to-end financial forecasting for the Long Range Financial Plan (LRFP) spanning from 2024
to 2028, managing sales figures totalling more than $4 billion.
  Implemented data validation and cleaning processes using Microsoft Excel, resulting in a 15% reduction in
data errors.
       
"""
# Initialize session state if not already done
if "skills_extraction_messages" not in st.session_state:
    st.session_state.skills_extraction_messages = []

if "skills" not in st.session_state:
    st.session_state.skills = "No Skills Defined"

# Initialize session state if not already done
if "aspiration_extraction_messages" not in st.session_state:
    st.session_state.aspiration_extraction_messages = []

# Initialize Learning Goal in session state if not already done
if "learning_goal" not in st.session_state:
    st.session_state.learning_goal = "No Goal Defined"

if "target_skills" not in st.session_state:
    st.session_state.target_skills = "No Target Skills Defined"

# Initialize session state if not already done
if "training_plan_generation_messages" not in st.session_state:
    st.session_state.training_plan_generation_messages = []

# Initialize Learning Goal in session state if not already done
if "training_plan" not in st.session_state:
    st.session_state.training_plan = "No Training Plan Defined"





def format_learning_goal_to_markdown(learning_goal):
    specific = learning_goal.get('specific', 'No specific goal provided.')
    measurable = learning_goal.get('measurable', 'No measurable criteria provided.')
    achievable = learning_goal.get('achievable', 'No achievable criteria provided.')
    relevant = learning_goal.get('relevant', 'No relevance provided.')
    time_bound = learning_goal.get('time_bound', 'No time-bound criteria provided.')
    explanation = learning_goal.get('explanation', 'No explanation provided.')
    
    
    markdown_text = f"""
    ### Specific
    {specific}

    ### Measurable
    **{measurable}**

    ### Achievable

    **{achievable}**

    ### Relevant

    **{relevant}**

    ### Time-Bound

    **{time_bound}**
    """

    print(markdown_text)
    return markdown_text

def format_training_plan_to_markdown(training_plan_json):
    markdown_output = "## Short Term Goals\n\n"

    if not training_plan_json or "short_term_goals" not in training_plan_json:
        return st.session_state.training_plan

    for i, goal in enumerate(training_plan_json['short_term_goals'], start=1):
        markdown_output += f"### Goal {i}\n"
        markdown_output += f"- **Learning Outcomes:** {goal['learning_outcomes']}\n"
        markdown_output += f"- **Specific:** {goal['specific']}\n"
        markdown_output += f"- **Measurable:** {goal['measurable']}\n"
        markdown_output += f"- **Achievable:** {goal['achievable']}\n"
        markdown_output += f"- **Relevant:** {goal['relevant']}\n"
        markdown_output += f"- **Time-bound:** {goal['time_bound']}\n\n"
    
    return markdown_output

def format_skills_to_markdown(skills_dict):
    """
    Generate markdown for skills from a dictionary.

    Args:
        skills_dict (dict): A dictionary where keys are skill names and values are skill descriptions.

    Returns:
        str: A markdown-formatted string representing the skills and their descriptions.
    """
    if not skills_dict:
        return "No skills identified."

    markdown_output = ""

    for skill_name, skill_description in skills_dict.items():
        markdown_output += f"### {skill_name}\n"
        markdown_output += f"{skill_description}\n\n"

    return markdown_output

st.set_page_config(
    page_title="TPRS",  # Title displayed on the browser tab
)
# Add a sidebar for navigation
tab = st.sidebar.radio("Select a Tab", ["Skills Extraction Pipeline","Aspiration Extraction", "Training Plan Generation"])






if tab == "Skills Extraction Pipeline":
    st.title("Skills Extraction Pipeline")
    # Initialize Streamlit callback handler
    st_callback = StreamlitCallbackHandler(st.container())

    


    # Display chat messages and capture user input in the main container
    # Main chat container
    with st.container():
        # Display chat history
        st.subheader("Chat History")
        for message in st.session_state.skills_extraction_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Capture user input at the bottom of the chat
        question = st.chat_input("Ask me anything...")
        

        if question:
            # Add user message to chat history
            st.session_state.skills_extraction_messages.append({"role": "user", "content": question})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question)

            # Run agent and display assistant response
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())  # Streamlit container for output
                response = requests.post(
                    "http://127.0.0.1:5000/ask_skills_extraction",
                    json={"question": question},
                )

                final_answer = response.json()["answer"]
                skills = response.json()["skills"]
                st.session_state.skills_extraction_messages.append({"role": "assistant", "content": final_answer})

                st.session_state.skills = format_skills_to_markdown(skills)
                st.write(final_answer)
            

    # Display skills in the sidebar
    with st.sidebar:
        st.subheader("Skills")
        st.markdown(st.session_state.skills)
    
    with st.expander("Example Questions"):
        st.write(SKILLS_EXTRACTION_START)


elif tab == "Aspiration Extraction":
    # Set the title of the application
    st.title("Aspiration Extraction Pipeline")


    

    

    # Display chat messages and capture user input in the main container
    with st.container():
        st.subheader("Chat History")
        for message in st.session_state.aspiration_extraction_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Capture user input at the bottom of the chat
        question = st.chat_input("Ask me anything...")


        if question:
            # Add user message to chat history
            st.session_state.aspiration_extraction_messages.append({"role": "user", "content": question})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question)

            # Run agent and display assistant response
            with st.chat_message("assistant"):
                response = requests.post(
                    "http://127.0.0.1:5000/ask_aspiration_extraction",
                    json={"question": question},
                )

                final_answer = response.json()["answer"]
                target_skills = response.json()["target_skills"]
                learning_goal = response.json()["learning_goal"]
                st.session_state.aspiration_extraction_messages.append({"role": "assistant", "content": final_answer})

                st.session_state.learning_goal = format_learning_goal_to_markdown(learning_goal)
                st.session_state.target_skills = format_skills_to_markdown(target_skills)
                
                st.write(final_answer)


    # Display Learning Goal above the chat history
    
    # Create a sidebar for Target Skills
    with st.sidebar:
        st.header("SMART Goal")
        st.write(st.session_state.learning_goal)

        st.header("Target Skills")
        st.markdown(st.session_state.target_skills)
    
    with st.expander("Example Questions"):
        st.write(ASPIRATION_EXTRACTION_START)
    

elif tab == "Training Plan Generation":
    # Training Plan Generation Tab
    st.header("Training Plan Generation")
    

    # Display chat messages and capture user input in the main container
    with st.container():
        st.subheader("Chat History")
        for message in st.session_state.training_plan_generation_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Capture user input at the bottom of the chat
        question = st.chat_input("Ask me anything...")

        if question:
            # Add user message to chat history
            st.session_state.training_plan_generation_messages.append({"role": "user", "content": question})

            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question)

            # Run agent and display assistant response
            with st.chat_message("assistant"):
                response = requests.post(
                    "http://127.0.0.1:5000/ask_training_generation",
                    json={"question": question, 
                          "client_skills": st.session_state.skills,
                          "target_skills": st.session_state.target_skills,
                          "learning_goal": st.session_state.learning_goal,
                          "prev_training_plan": st.session_state.training_plan},
                )
            
                final_answer = response.json()["answer"]
                training_plan = response.json()["training_plan"]
                st.session_state.training_plan_generation_messages.append({"role": "assistant", "content": final_answer})

                if training_plan:
                    st.session_state.training_plan = format_training_plan_to_markdown(training_plan)
                st.write(final_answer)

    # Display Learning Goal in the sidebar
    with st.sidebar:
        st.header("Training Plan")
        st.markdown(st.session_state.training_plan)
