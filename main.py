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
from rich.live import Live
from rich.align import Align

from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# === INITIALIZATION ==========================================================

load_dotenv()
console = Console()

# === CONFIGURATION AND SETUP =================================================

class Settings(BaseSettings):
    GROQ_API_KEY: str ="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"
    DATABASE_URL: str = "sqlite:///./interviews.db"

settings = Settings()

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_name = Column(String, nullable=False)   # NEW FIELD
    status = Column(String, nullable=False, default='PENDING')
    job_description = Column(Text, nullable=False)
    difficulty = Column(String, nullable=False)
    questions_json = Column(JSON, nullable=True)
    # NEW COLUMN (added after initial deployment) stores ideal answers
    answers_key_json = Column(JSON, nullable=True)
    transcript_json = Column(JSON, nullable=True, default=[])
    report_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# --- Lightweight migration to add missing columns if DB was created before new fields ---
def run_light_migrations():
    try:
        insp = inspect(engine)
        cols = [c['name'] for c in insp.get_columns('interview_sessions')]
        if 'answers_key_json' not in cols:
            # SQLite: JSON will be treated as TEXT if native JSON not supported
            with engine.begin() as conn:
                conn.execute(text('ALTER TABLE interview_sessions ADD COLUMN answers_key_json JSON'))
    except Exception as mig_err:
        console.print(f"[bold yellow]Migration warning:[/bold yellow] {mig_err}")

run_light_migrations()

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
    difficulty_map = {
        "basic": "Beginner-level",
        "intermediate": "Intermediate-level",
        "difficult": "Advanced-level"
    }
    difficulty_label = difficulty_map.get(difficulty.lower(), "Intermediate-level")

    llm = ChatGroq(temperature=0.4, groq_api_key=settings.GROQ_API_KEY, model_name="groq/compound")
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a technical hiring manager. Generate one concise, practical, {difficulty_label} interview question for the following skill. Only provide the question text itself."),
        ("human", "Skill: {skill}")
    ])

    chain = prompt | llm | StrOutputParser()
    questions = []
    for skill in skills[:max_questions]:
        question = chain.invoke({"difficulty": difficulty, "skill": skill})
        questions.append(question.strip())
    return questions
    

def evaluate_transcript(transcript: List[Dict], candidate_name: str) -> str:
    def _truncate(text: str, max_chars: int = 700) -> str:
        if not text:
            return ""
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    compact_lines = []
    for i, item in enumerate(transcript):
        q = _truncate(item.get('question', ''), 240)
        a = _truncate(item.get('answer', ''), 600)
        compact_lines.append(f"Q{i+1}: {q}\nA{i+1}: {a}")
    transcript_blob = "\n\n".join(compact_lines)

    prompt_text = f"""
You are an expert technical interviewer providing feedback for candidate **{{candidate_name}}**.
Provide a concise overall summary and a final score out of 10.
Then for each question provide a score and 1 bullet of constructive feedback.
Return clean Markdown.

Transcript (condensed):
{{transcript_text}}
"""
    models_to_try = [
        "groq/compound",
        "qwen/qwen3-32b",
        "llama-3.1-8b-instant"
    ]
    last_err = None
    for model_name in models_to_try:
        try:
            llm = ChatGroq(temperature=0, groq_api_key=settings.GROQ_API_KEY, model_name=model_name)
            chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()
            return chain.invoke({"transcript_text": transcript_blob, "candidate_name": candidate_name})
        except Exception as e:
            last_err = e
            if "rate limit" in str(e).lower() or "429" in str(e):
                continue
            continue
    raise last_err if last_err else RuntimeError("Evaluation failed with all models")

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

    console.print(Panel("Welcome Candidate!", style="bold magenta"))
    candidate_name = console.input("[bold blue]Enter your full name:> [/bold blue]").strip()
    if not candidate_name:
        console.print("[bold red]Name cannot be empty. Using 'Unknown Candidate'.[/bold red]")
        candidate_name = "Unknown Candidate"

    if not os.path.exists(jd_file_path):
        console.print(f"[bold red]Error:[/bold red] The required file '{jd_file_path}' was not found.")
        console.print("Please create a 'jd.txt' file in this directory with the job description.")
        sys.exit(1)

    with open(jd_file_path, 'r') as f:
        jd_text = f.read()

    # difficulty_level = infer_difficulty(jd_text)
    console.print(Panel("Select difficulty level: [bold green]basic[/bold green], [bold yellow]intermediate[/bold yellow], [bold red]difficult[/bold red]", style="cyan"))
    difficulty_level = console.input("[bold blue]Enter difficulty level:> [/bold blue]").strip().lower()

    if difficulty_level not in {"basic", "intermediate", "difficult"}:
        console.print("[bold red]Invalid choice! Defaulting to 'intermediate'.[/bold red]")
        difficulty_level = "intermediate"

    db = SessionLocal()
    try:
        session = InterviewSession(candidate_name=candidate_name, job_description=jd_text, difficulty=difficulty_level)
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

        time_limit = 45 * 60  # 45 minutes in seconds
        start_time = time.time()

        question_count = 1
        transcript = []

        for question in session.questions_json:
            elapsed = time.time() - start_time
            remaining = int(time_limit - elapsed)

            if remaining <= 0:
                console.print("[bold red]⏰ Time is up! The assessment has ended.[/bold red]")
                break

            # Format as MM:SS
            minutes, seconds = divmod(remaining, 60)
            time_display = f"{minutes:02d}:{seconds:02d}"

            console.print(Panel(
                f"Question {question_count}/{len(session.questions_json)} (⏳ Time left: {time_display})\n\n{question}",
                title="Question",
                border_style="cyan"
            ))

            answer = console.input("[bold yellow]Your Answer:> [/bold yellow]")

            transcript.append({"question": question, "answer": answer})
            session.transcript_json = transcript
            db.commit()

            question_count += 1
            console.print("-" * 50)


        # ✅ Always go to evaluation whether time ends or all questions answered
        console.print("\n[bold blue]Interview ended. Evaluating your responses...[/bold blue]")
        with console.status("[bold yellow]Generating your final report...[/bold yellow]", spinner="dots"):
            try:
                report = evaluate_transcript(session.transcript_json, session.candidate_name)
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
