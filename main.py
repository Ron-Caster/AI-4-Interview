# === IMPORTS =================================================================
import os
import sys
import uuid
import time
from datetime import datetime
from typing import List, Dict

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# === INITIALIZATION ==========================================================

load_dotenv()
console = Console()

# === CONFIGURATION AND SETUP =================================================

class Settings(BaseSettings):
    GROQ_API_KEY: str
    DATABASE_URL: str = "sqlite:///./interviews.db"

settings = Settings()

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, nullable=False, default='PENDING')
    job_description = Column(Text, nullable=False)
    difficulty = Column(String, nullable=False)
    questions_json = Column(JSON, nullable=True)
    transcript_json = Column(JSON, nullable=True, default=[])
    report_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# === Pydantic Schemas ========================================================

class Skills(BaseModel):
    skills: List[str] = Field(description="A list of key technical skills.")

# === BACKEND LOGIC FUNCTIONS =================================================

def parse_job_description(job_description: str) -> List[str]:
    llm = ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="qwen/qwen3-32b")
    structured_llm = llm.with_structured_output(Skills)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior technical recruiter. Extract the key technical skills from the job description."),
        ("human", "{job_description}")
    ])

    chain = prompt | structured_llm
    result_object = chain.invoke({"job_description": job_description})

    # Filter out overly generic skills
    blacklist = {"machine learning", "deep learning", "data science", "analytics"}
    filtered_skills = [s for s in result_object.skills if s.lower() not in blacklist]

    # Score and prioritize skills that appear in the JD more frequently
    jd_lower = job_description.lower()
    scored_skills = sorted(
        filtered_skills,
        key=lambda s: jd_lower.count(s.lower()),
        reverse=True
    )

    # Return only the top 8 most relevant skills
    return scored_skills[:8]

def generate_questions(skills: List[str], difficulty: str, max_questions: int = 8) -> List[str]:
    llm = ChatGroq(temperature=0.4, groq_api_key=settings.GROQ_API_KEY, model_name="groq/compound")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical hiring manager. Generate one concise, practical, {difficulty} level interview question for the following skill. Only provide the question text itself."),
        ("human", "Skill: {skill}")
    ])

    chain = prompt | llm | StrOutputParser()
    questions = []
    for skill in skills[:max_questions]:
        question = chain.invoke({"difficulty": difficulty, "skill": skill})
        questions.append(question.strip())
    return questions

def evaluate_transcript(transcript: List[Dict]) -> str:
    llm = ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name="groq/compound")
    prompt_text = """
    You are an expert technical interviewer providing feedback. Evaluate the candidate's responses from the interview transcript below.
    Provide a concise overall summary and a final score out of 10.
    Then, for each question, provide a score and 1-2 bullet points of constructive feedback.
    Format your response cleanly in Markdown.

    Transcript:
    {transcript_text}
    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()

    formatted_transcript = ""
    for i, item in enumerate(transcript):
        formatted_transcript += f"Question {i+1}: {item['question']}\nAnswer {i+1}: {item['answer']}\n\n"

    report = chain.invoke({"transcript_text": formatted_transcript})
    return report

def infer_difficulty(jd_text: str) -> str:
    jd_lower = jd_text.lower()
    if any(word in jd_lower for word in ["senior", "lead", "architect"]):
        return "advanced"
    elif any(word in jd_lower for word in ["junior", "entry", "associate"]):
        return "beginner"
    return "intermediate"

# === MAIN APPLICATION SCRIPT =================================================

def start_interview():
    jd_file_path = "jd.txt"

    console.print(Panel("AI Mock Interviewer", style="bold blue"))

    if not os.path.exists(jd_file_path):
        console.print(f"[bold red]Error:[/bold red] The required file '{jd_file_path}' was not found.")
        console.print("Please create a 'jd.txt' file in this directory with the job description.")
        sys.exit(1)

    with open(jd_file_path, 'r') as f:
        jd_text = f.read()

    difficulty_level = infer_difficulty(jd_text)
    db = SessionLocal()
    try:
        session = InterviewSession(job_description=jd_text, difficulty=difficulty_level)
        db.add(session)
        db.commit()
        db.refresh(session)
        console.print(f"[bold green]✓[/bold green] Interview session created ({session.session_id}).")

        with console.status("[bold yellow]Generating AI questions based on your JD...[/bold yellow]", spinner="dots"):
            try:
                skills = parse_job_description(session.job_description)
                questions = generate_questions(skills, session.difficulty)
                session.questions_json = questions
                session.status = 'READY'
                db.commit()
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Failed to generate AI questions: {e}")
                session.status = 'FAILED'
                db.commit()
                return

        console.print("[bold green]✓[/bold green] Questions are ready. The interview will now begin.")
        time.sleep(1)

        question_count = 1
        transcript = []
        for question in session.questions_json:
            console.print(Panel(f"Question {question_count}/{len(session.questions_json)}:\n\n{question}", title="Question", border_style="cyan"))
            answer = console.input("[bold yellow]Your Answer:> [/bold yellow]")

            transcript.append({"question": question, "answer": answer})
            session.transcript_json = transcript
            db.commit()

            question_count += 1
            console.print("-" * 50)

        console.print("\n[bold blue]All questions have been answered. Evaluating your responses...[/bold blue]")
        with console.status("[bold yellow]Generating your final report...[/bold yellow]", spinner="dots"):
            try:
                report = evaluate_transcript(session.transcript_json)
                session.report_text = report
                session.status = 'COMPLETED'
                db.commit()
                console.print(Panel(report, title="Evaluation Report", border_style="bold green", expand=True))
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] Failed to generate evaluation: {e}")
                session.status = 'FAILED'
                db.commit()
    finally:
        db.close()

if __name__ == '__main__':
    start_interview()
