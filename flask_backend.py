# Initialize Flask app
from flask import Flask, request, jsonify
from resume_extraction import ask_skills_extraction_agent, get_skills
from aspiration_extraction import ask_aspiration_extraction_agent, get_learning_goal, get_target_skills
from training_plan_generation  import ask_training_plan_generation_agent, get_training_plan

app = Flask(__name__)
SKILLS = {}
LEARNING_GOAL = {}
TARGET_SKILLS = {}
TRAINING_PLAN = {}


# Define Flask route
@app.route('/ask_skills_extraction', methods=['POST'])
def ask_skills_extraction():
    # Get the question from the POST request
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400
    

    final_answer = ask_skills_extraction_agent(question)

    # Return the answer as a JSON response
    SKILLS = get_skills()
    return jsonify({'answer': final_answer, "skills": SKILLS})

# Define Flask route
@app.route('/ask_aspiration_extraction', methods=['POST'])
def ask_aspiration_extraction():
    # Get the question from the POST request
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400
    

    final_answer = ask_aspiration_extraction_agent(question)

    # Return the answer as a JSON response
    LEARNING_GOAL = get_learning_goal()
    TARGET_SKILLS = get_target_skills()

    print(f"Target Skill: {TARGET_SKILLS}")

    return jsonify({'answer': final_answer, "learning_goal": LEARNING_GOAL, "target_skills": TARGET_SKILLS})

@app.route('/ask_training_generation', methods=['POST'])
def training_plan_generation():
    # Get the question from the POST request
    data = request.get_json()
    client_skills = data.get('client_skills', '')
    target_skills = data.get('target_skills', '')
    learning_goal = data.get('learning_goal', '')
    prev_training_plan = data.get('prev_training_plan', '')
    question = data.get('question', '')

    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    print(prev_training_plan)
    if prev_training_plan != "No Training Plan Defined":
        final_answer = ask_training_plan_generation_agent(f"question: {question}")
    else:
        final_answer = ask_training_plan_generation_agent(f"question: {question}, client_skills: {client_skills[:min(len(client_skills), 500)]}, target_skills: {target_skills}, learning_goal: {learning_goal}")
    # Return the answer as a JSON response
    TRAINING_PLAN = get_training_plan()

    print(f"Training Plan: {TRAINING_PLAN}")

    return jsonify({'answer': final_answer, "training_plan": TRAINING_PLAN})


if __name__ == '__main__':
    app.run(debug=True)