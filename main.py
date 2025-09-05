# === IMPORTS =================================================================
# --- Standard Library Imports ---
import os  # Used for accessing environment variables and checking file paths.
import sys # Used to exit the script gracefully if a file is not found.
import uuid  # Used for generating unique IDs for interview sessions.
from datetime import datetime  # Used for timestamping database records.
from typing import List, Dict, Optional  # Provides type hints for clearer code.
import time  # Used for the `sleep()` function for better user experience.

# --- Third-Party Imports ---
from pydantic import BaseModel, Field  # For data validation and creating data schemas.
from pydantic_settings import BaseSettings  # For managing application settings from environment variables.
from dotenv import load_dotenv  # Used to load environment variables from the .env file.
from rich.console import Console  # A component from the Rich library for printing styled text.
from rich.panel import Panel  # A component from Rich to display text inside a bordered box.

# --- SQLAlchemy (Database) Imports ---
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON  # Core components for database interaction.
from sqlalchemy.orm import sessionmaker, declarative_base  # For creating database sessions and models.
from sqlalchemy.orm import Session  # The database session type hint.

# --- LangChain and Groq (AI) Imports ---
from langchain_core.prompts import ChatPromptTemplate  # For creating prompts to send to the AI.
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser  # For parsing the AI's response.
from langchain_groq import ChatGroq  # The LangChain integration for the Groq API.

# === INITIALIZATION ==========================================================

# --- Load Environment Variables ---
load_dotenv()  # This function finds and loads the variables from the .env file.

# --- Global Console Object ---
console = Console()  # Creates a global console object to use for all Rich printing.

# === CONFIGURATION AND SETUP =================================================

# --- 1. Application Configuration ---
class Settings(BaseSettings):
    # Defines application settings and reads them from the environment.
    GROQ_API_KEY: str  # Expects the Groq API key to be available as an environment variable.
    DATABASE_URL: str = "sqlite:///./interviews.db"  # Specifies the connection string for our SQLite database file.

settings = Settings()  # Creates an instance of the Settings class.

# --- 2. Database Setup (SQLite) ---
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})  # Creates the database engine for SQLite.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)  # Creates a configured "Session" class.
Base = declarative_base()  # Creates a base class for our database models to inherit from.

class InterviewSession(Base):
    # This class defines the structure of the 'interview_sessions' table in our database.
    __tablename__ = "interview_sessions"  # The name of the database table.
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))  # The unique ID for the session.
    status = Column(String, nullable=False, default='PENDING')  # The current status of the interview.
    job_description = Column(Text, nullable=False)  # The job description text.
    difficulty = Column(String, nullable=False)  # The difficulty level.
    questions_json = Column(JSON, nullable=True)  # A JSON field to store the list of generated questions.
    transcript_json = Column(JSON, nullable=True, default=[])  # A JSON field to store the Q&A transcript.
    report_text = Column(Text, nullable=True)  # A text field for the final evaluation report.
    created_at = Column(DateTime, default=datetime.utcnow)  # A timestamp for when the session was created.
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)  # A timestamp that updates on any change.

Base.metadata.create_all(bind=engine)  # This line creates the database table if it doesn't already exist.

# --- 3. Pydantic Schemas (for AI output validation) ---
class Skills(BaseModel):
    # Defines the expected JSON structure for skills extracted by the AI.
    skills: List[str] = Field(description="A list of key technical skills.")

# === BACKEND LOGIC FUNCTIONS =================================================

# NEW ROBUST VERSION - USE THIS
def parse_job_description(job_description: str) -> List[str]:
    # This function takes a job description and extracts skills using the AI.
    llm = ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="qwen/qwen3-32b")  # Initializes the Groq LLM.

    # This is the key change: we bind our Pydantic schema directly to the LLM.
    # This forces the model to return a valid JSON object matching the 'Skills' schema.
    structured_llm = llm.with_structured_output(Skills)

    prompt = ChatPromptTemplate.from_messages([  # Defines the prompt to guide the AI.
        ("system", "You are a senior technical recruiter. Extract the key technical skills from the job description."), # The prompt can be slightly simpler now.
        ("human", "{job_description}")
    ])
    
    # The chain no longer needs a separate JSON parser.
    chain = prompt | structured_llm
    
    # The result of the invoke will now be a Pydantic 'Skills' object directly.
    result_object = chain.invoke({"job_description": job_description})
    
    # We return the list of skills from the object.
    return result_object.skills

def generate_questions(skills: List[str], difficulty: str) -> List[str]:
    # This function takes a list of skills and generates interview questions.
    llm = ChatGroq(temperature=0.4, groq_api_key=settings.GROQ_API_KEY, model_name="groq/compound")  # Initializes the LLM.
    prompt = ChatPromptTemplate.from_messages([  # Defines the question generation prompt.
        ("system", "You are a technical hiring manager. Generate one concise, practical, {difficulty} level interview question for the following skill. Only provide the question text itself."),
        ("human", "Skill: {skill}")
    ])
    chain = prompt | llm | StrOutputParser()  # Chains the components. StrOutputParser returns a simple string.
    questions = []  # Initializes an empty list to hold the questions.
    for skill in skills:  # Loops through each skill.
        question = chain.invoke({"difficulty": difficulty, "skill": skill})  # Generates a question for the skill.
        questions.append(question.strip())  # Adds the cleaned-up question to the list.
    return questions  # Returns the final list of questions.

def evaluate_transcript(transcript: List[Dict]) -> str:
    # This function takes the interview transcript and generates a final evaluation.
    llm = ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="groq/compound")  # Initializes the LLM.
    prompt_text = """
    You are an expert technical interviewer providing feedback. Evaluate the candidate's responses from the interview transcript below.
    Provide a concise overall summary and a final score out of 10.
    Then, for each question, provide a score and 1-2 bullet points of constructive feedback.
    Format your response cleanly in Markdown.

    Transcript:
    {transcript_text}
    """  # Defines the detailed evaluation prompt.
    prompt = ChatPromptTemplate.from_template(prompt_text)  # Creates the prompt template.
    chain = prompt | llm | StrOutputParser()  # Chains the components.

    # Formats the transcript list into a readable string for the AI.
    formatted_transcript = ""  # Initializes an empty string.
    for i, item in enumerate(transcript):  # Loops through each question-answer pair.
        formatted_transcript += f"Question {i+1}: {item['question']}\nAnswer {i+1}: {item['answer']}\n\n"  # Appends formatted text.

    report = chain.invoke({"transcript_text": formatted_transcript})  # Executes the chain to get the report.
    return report  # Returns the final report text.

# === MAIN APPLICATION SCRIPT =================================================

def start_interview():
    # This is the main function that runs the entire interview flow.
    
    # --- HARDCODED VALUES ---
    jd_file_path = "jd.txt"  # The name of the job description file is now hardcoded.
    difficulty_level = "intermediate"  # The difficulty level is now hardcoded.

    console.print(Panel("AI Mock Interviewer", style="bold blue"))  # Prints a welcome banner.
    
    # --- Check if jd.txt exists before proceeding ---
    if not os.path.exists(jd_file_path): # Checks if the file exists in the current directory.
        console.print(f"[bold red]Error:[/bold red] The required file '{jd_file_path}' was not found.") # Prints an error message.
        console.print("Please create a 'jd.txt' file in this directory with the job description.") # Provides instructions.
        sys.exit(1) # Exits the script with an error code.

    with open(jd_file_path, 'r') as f:  # Opens the hardcoded job description file.
        jd_text = f.read()  # Reads the content of the file into a variable.

    db = SessionLocal()  # Creates a new database session for this entire run.
    try:  # A try...finally block ensures the database session is always closed at the end.
        # --- Step 1: Create and save the initial interview session ---
        session = InterviewSession(job_description=jd_text, difficulty=difficulty_level)  # Creates a new DB model instance using the hardcoded difficulty.
        db.add(session)  # Adds it to the database session.
        db.commit()  # Commits the transaction to save it to the database.
        db.refresh(session)  # Refreshes the instance to get the generated ID.
        console.print(f"[bold green]✓[/bold green] Interview session created ({session.session_id}).")  # Informs the user of success.
        
        # --- Step 2: Prepare questions (Synchronous blocking operation) ---
        with console.status("[bold yellow]Generating AI questions based on your JD...[/bold yellow]", spinner="dots") as status:  # Creates a spinner animation while waiting.
            try:
                skills = parse_job_description(session.job_description)  # Calls the parser function.
                questions = generate_questions(skills, session.difficulty)  # Calls the generator function.
                session.questions_json = questions  # Updates the session object with the questions.
                session.status = 'READY'  # Sets the status to ready.
                db.commit()  # Saves these changes to the database.
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Failed to generate AI questions: {e}")  # Prints an error message on failure.
                session.status = 'FAILED'  # Sets the status to failed.
                db.commit()  # Saves the failed status.
                return  # Exits the application.

        console.print("[bold green]✓[/bold green] Questions are ready. The interview will now begin.")
        time.sleep(1) # A brief pause for readability.
        
        # --- Step 3: Conduct the interview loop ---
        question_count = 1  # Initializes a counter for the questions.
        transcript = []  # Initializes a local list to hold the transcript.
        for question in session.questions_json:  # Loops through the generated questions.
            console.print(Panel(f"Question {question_count}/{len(session.questions_json)}:\n\n{question}", title="Question", border_style="cyan"))  # Displays the question.
            answer = console.input("[bold yellow]Your Answer:> [/bold yellow]")  # Prompts the user for their answer.
            
            transcript.append({"question": question, "answer": answer})  # Adds the Q&A pair to the local transcript.
            session.transcript_json = transcript  # Updates the session object.
            db.commit()  # Saves the latest answer to the database after each question.
            
            question_count += 1  # Increments the question counter.
            console.print("-" * 50)  # Prints a separator line.

        # --- Step 4: Finalize and get the evaluation report ---
        console.print("\n[bold blue]All questions have been answered. Evaluating your responses...[/bold blue]")
        with console.status("[bold yellow]Generating your final report...[/bold yellow]", spinner="dots") as status:  # Shows a spinner while waiting.
            try:
                report = evaluate_transcript(session.transcript_json)  # Calls the evaluation function.
                session.report_text = report  # Updates the session object with the report.
                session.status = 'COMPLETED'  # Sets the final status.
                db.commit()  # Saves the report to the database.
                console.print(Panel(report, title="Evaluation Report", border_style="bold green", expand=True))  # Prints the final report.
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Failed to generate evaluation: {e}") # Handles evaluation errors.
                session.status = 'FAILED'  # Sets status to failed.
                db.commit()  # Saves the status.

    finally:  # This block will always execute, even if an error occurs.
        db.close()  # Ensures the database session is closed properly.


if __name__ == '__main__':  # The standard entry point for a Python script.
    start_interview()  # Calls the main function to start the script.