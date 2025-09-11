import os
import uuid
import time
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# ================= Settings ======================
class Settings(BaseSettings):
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./interviews.db")

settings = Settings()

# ================= Database =====================
engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_name = Column(String, nullable=False)
    status = Column(String, nullable=False, default='PENDING')
    job_description = Column(Text, nullable=False)
    difficulty = Column(String, nullable=False)
    questions_json = Column(JSON, nullable=True)
    answers_key_json = Column(JSON, nullable=True)  # NEW for storing model answers
    transcript_json = Column(JSON, nullable=True, default=[])
    report_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# --- Lightweight migration (adds column if upgrading from older version) ---
def run_light_migrations():
    try:
        insp = inspect(engine)
        cols = [c['name'] for c in insp.get_columns('interview_sessions')]
        if 'answers_key_json' not in cols:
            with engine.begin() as conn:
                conn.execute(text('ALTER TABLE interview_sessions ADD COLUMN answers_key_json JSON'))
    except Exception as e:
        st.sidebar.warning(f"Migration issue: {e}")

run_light_migrations()

# ================= Pydantic =====================
class Skills(BaseModel):
    skills: List[str] = Field(description="A list of key technical skills.")

# ================= LLM Helpers ==================

def get_llm(model_name: str = "groq/compound", temperature: float = 0):
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise ValueError("GROQ_API_KEY not set. Provide it in .env or environment variable.")
    return ChatGroq(temperature=temperature, groq_api_key=api_key, model_name=model_name)

def parse_job_description(job_description: str) -> List[str]:
    llm = get_llm(model_name="qwen/qwen3-32b")
    structured_llm = llm.with_structured_output(Skills)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a senior technical recruiter. Extract the key technical skills from the job description."),
        ("human", "{job_description}")
    ])
    chain = prompt | structured_llm
    result_object = chain.invoke({"job_description": job_description})
    blacklist = {"machine learning", "deep learning", "data science", "analytics"}
    filtered_skills = [s for s in result_object.skills if s.lower() not in blacklist]
    jd_lower = job_description.lower()
    scored_skills = sorted(filtered_skills, key=lambda s: jd_lower.count(s.lower()), reverse=True)
    return scored_skills[:8]

def generate_questions_and_answers(skills: List[str], difficulty: str, max_items: int = 8):
    difficulty_map = {
        "basic": "Beginner-level",
        "intermediate": "Intermediate-level",
        "difficult": "Advanced-level"
    }
    difficulty_label = difficulty_map.get(difficulty.lower(), "Intermediate-level")

    # Question generation
    llm_q = get_llm(model_name="groq/compound", temperature=0.4)
    prompt_q = ChatPromptTemplate.from_messages([
        ("system", f"You are a technical hiring manager. Generate one concise, practical, {difficulty_label} interview question for the following skill. Only provide the question text itself."),
        ("human", "Skill: {skill}")
    ])
    chain_q = prompt_q | llm_q | StrOutputParser()

    # Answer key generation
    llm_a = get_llm(model_name="groq/compound", temperature=0)
    prompt_a = ChatPromptTemplate.from_messages([
        ("system", "You are an expert interviewer. Provide a concise, high-quality exemplary answer (3-6 sentences) to the following technical interview question. Avoid fluff."),
        ("human", "Question: {question}")
    ])
    chain_a = prompt_a | llm_a | StrOutputParser()

    questions = []
    answers = []
    for skill in skills[:max_items]:
        q = chain_q.invoke({"skill": skill}).strip()
        a = chain_a.invoke({"question": q}).strip()
        questions.append(q)
        answers.append(a)
    return questions, answers

def evaluate_transcript(transcript: List[Dict], candidate_name: str, answer_key: List[str]) -> str:
    llm = get_llm(model_name="groq/compound", temperature=0)
    prompt_text = """
You are an expert technical interviewer providing feedback for candidate {candidate_name}.
Evaluate the candidate's responses from the interview transcript below.
Provide a concise overall summary and a final score out of 10.
Then, for each question, show:
1. The Question
2. The Ideal Answer (from answer key)
3. Candidate Answer Summary (1-2 sentences)
4. Score (0-10) and 1 bullet of actionable feedback.
Return clean Markdown.

Transcript:
{transcript_text}
Answer Key:
{answer_key_text}
"""
    prompt = ChatPromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()

    formatted_transcript = []
    for i, item in enumerate(transcript):
        formatted_transcript.append(
            f"Question {i+1}: {item['question']}\nCandidate Answer: {item['answer']}\n"
        )
    transcript_blob = "\n".join(formatted_transcript)
    answer_blob = "\n".join([f"Q{i+1}: {a}" for i, a in enumerate(answer_key)])
    return chain.invoke({
        "candidate_name": candidate_name,
        "transcript_text": transcript_blob,
        "answer_key_text": answer_blob
    })

# ================= Utility ======================
TOTAL_TIME_SECONDS = 45 * 60  # 45 min

def time_left(start_ts: float) -> int:
    elapsed = int(time.time() - start_ts)
    return max(0, TOTAL_TIME_SECONDS - elapsed)

# ================= Streamlit UI =================
st.set_page_config(page_title="AI Interview", page_icon="🤖", layout="wide")
st.title("🤖 AI Technical Interview")

# Sidebar config
with st.sidebar:
    st.header("Session")
    api_key_input = st.text_input("GROQ API Key", type="password", help="Overrides environment variable.")
    if api_key_input:
        settings.GROQ_API_KEY = api_key_input
    jd_file = st.file_uploader("Upload Job Description (jd.txt)", type=["txt"])    
    difficulty = st.selectbox("Difficulty", ["basic", "intermediate", "difficult"], index=1)
    candidate_name = st.text_input("Candidate Full Name")
    start_btn = st.button("Start Interview", type="primary")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "questions" not in st.session_state:
    st.session_state.questions = []
if "answers_key" not in st.session_state:
    st.session_state.answers_key = []
if "responses" not in st.session_state:
    st.session_state.responses = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "completed" not in st.session_state:
    st.session_state.completed = False
if "report" not in st.session_state:
    st.session_state.report = None
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

def log(msg: str):
    ts = time.strftime('%H:%M:%S')
    st.session_state.debug_logs.append(f"[{ts}] {msg}")


status_placeholder = st.empty()
timer_placeholder = st.empty()

# Start interview
if start_btn and not st.session_state.session_id:
    if not candidate_name:
        st.warning("Please enter candidate name.")
    elif not jd_file:
        st.warning("Please upload job description.")
    else:
        jd_text = jd_file.read().decode("utf-8")
        db = SessionLocal()
        try:
            log("Starting job description parsing")
            with st.spinner("Parsing job description..."):
                skills = parse_job_description(jd_text)
            log(f"Extracted skills: {skills}")

            st.write("Generating questions and ideal answers:")
            progress = st.progress(0)
            questions = []
            answers_key = []
            skill_slice = skills[:8] if skills else []
            total = max(1, len(skill_slice))
            if not skill_slice:
                log("No skills extracted; using generic placeholders")
                skill_slice = ["problem solving", "system design", "python basics"]

            # Build chains once outside loop
            try:
                difficulty_map = {"basic": "Beginner-level", "intermediate": "Intermediate-level", "difficult": "Advanced-level"}
                difficulty_label = difficulty_map.get(difficulty.lower(), "Intermediate-level")
                llm_q = get_llm(model_name="groq/compound", temperature=0.4)
                prompt_q = ChatPromptTemplate.from_messages([
                    ("system", f"You are a technical hiring manager. Generate one concise, practical, {difficulty_label} interview question for the following skill. Only provide the question text itself."),
                    ("human", "Skill: {skill}")
                ])
                chain_q = prompt_q | llm_q | StrOutputParser()

                llm_a = get_llm(model_name="groq/compound", temperature=0)
                prompt_a = ChatPromptTemplate.from_messages([
                    ("system", "You are an expert interviewer. Provide a concise, high-quality exemplary answer (3-6 sentences) to the following technical interview question. Avoid fluff."),
                    ("human", "Question: {question}")
                ])
                chain_a = prompt_a | llm_a | StrOutputParser()
            except Exception as chain_err:
                log(f"Failed to initialize LLM chains: {chain_err}")
                chain_q = chain_a = None

            for i, skill in enumerate(skill_slice, start=1):
                try:
                    if chain_q and chain_a:
                        q = chain_q.invoke({"skill": skill}).strip()
                        a = chain_a.invoke({"question": q}).strip()
                    else:
                        q = f"Describe your experience with {skill}."
                        a = f"A strong answer would cover core concepts, concrete examples, measurable outcomes, and challenges overcome related to {skill}."
                    questions.append(q)
                    answers_key.append(a)
                    log(f"Generated Q{i}: {q[:70]}...")
                except Exception as gen_err:
                    fallback_q = f"Explain a key concept about {skill}."
                    fallback_a = f"An ideal answer clearly defines the concept, gives a short example, and notes a trade-off related to {skill}."
                    questions.append(fallback_q)
                    answers_key.append(fallback_a)
                    log(f"Error generating question for {skill}: {gen_err}; used fallback.")
                progress.progress(min(int(i/total*100), 100))
            progress.progress(100)

            session = InterviewSession(candidate_name=candidate_name, job_description=jd_text, difficulty=difficulty, questions_json=questions, answers_key_json=answers_key, status='READY')
            db.add(session)
            db.commit()
            db.refresh(session)
            st.session_state.session_id = session.session_id
            st.session_state.questions = questions
            st.session_state.answers_key = answers_key
            st.session_state.start_time = time.time()
            st.success(f"Session created: {session.session_id}")
            log("Interview session initialized successfully")
        except Exception as e:
            st.error(f"Failed to start interview: {e}")
            log(f"Startup failure: {e}")
        finally:
            db.close()

# Active interview logic
if st.session_state.session_id and not st.session_state.completed:
    # Static server-side remaining (authoritative)
    remaining = time_left(st.session_state.start_time)
    minutes, seconds = divmod(remaining, 60)
    # Client-side live countdown (no reruns) — purely visual
    components.html(
        f"""
<div id='timer-box' style='padding:8px 14px;border:2px solid #444;border-radius:8px;display:inline-block;font-family:monospace;font-size:20px;font-weight:600;background:#111;color:#0f0;'>
    Time Left: <span id='time-left'>{minutes:02d}:{seconds:02d}</span>
</div>
<script>
    let total = {remaining};
    function fmt(s){{ const m=Math.floor(s/60).toString().padStart(2,'0'); const c=(s%60).toString().padStart(2,'0'); return m+':'+c; }}
    const span=document.getElementById('time-left');
    function tick(){{ total -= 1; if(total < 0){{ return; }} span.textContent = fmt(total); }}
    setInterval(tick, 1000);
</script>
        """,
        height=60
    )
    if remaining == 0:
        st.warning("Time is up! Submit last answer and proceed to evaluation.")

    idx = st.session_state.current_index
    if idx < len(st.session_state.questions):
        st.subheader(f"Question {idx+1}/{len(st.session_state.questions)}")
        st.info(st.session_state.questions[idx])

        # Autosave answer in session_state using stable key
        answer_key = f"answer_{idx}"
        if answer_key not in st.session_state:
            # Preload from responses if returning to a previous question
            if idx < len(st.session_state.responses):
                st.session_state[answer_key] = st.session_state.responses[idx]["answer"]
            else:
                st.session_state[answer_key] = ""
        st.text_area("Your Answer", key=answer_key, height=180, placeholder="Type your answer here. It is autosaved.")

        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Save & Next", disabled=remaining==0):
                current_answer = st.session_state.get(answer_key, "")
                if idx < len(st.session_state.responses):
                    st.session_state.responses[idx]["answer"] = current_answer
                else:
                    st.session_state.responses.append({"question": st.session_state.questions[idx], "answer": current_answer})
                st.session_state.current_index += 1
        with col2:
            if st.button("Save (Stay)"):
                current_answer = st.session_state.get(answer_key, "")
                if idx < len(st.session_state.responses):
                    st.session_state.responses[idx]["answer"] = current_answer
                else:
                    st.session_state.responses.append({"question": st.session_state.questions[idx], "answer": current_answer})
                st.success("Saved")
        with col3:
            if st.button("Finish Now"):
                current_answer = st.session_state.get(answer_key, "")
                if idx < len(st.session_state.responses):
                    st.session_state.responses[idx]["answer"] = current_answer
                else:
                    st.session_state.responses.append({"question": st.session_state.questions[idx], "answer": current_answer})
                st.session_state.completed = True
    else:
        st.success("All questions answered. Click 'Generate Report' below to evaluate.")
        if st.button("Generate Report"):
            st.session_state.completed = True

# Evaluation
if st.session_state.completed and st.session_state.report is None:
    with st.spinner("Evaluating responses and generating feedback + answer key..."):
        try:
            report = evaluate_transcript(st.session_state.responses, candidate_name or "Candidate", st.session_state.answers_key)
            st.session_state.report = report
            # Persist transcript + report
            db = SessionLocal()
            try:
                session = db.query(InterviewSession).filter(InterviewSession.session_id == st.session_state.session_id).first()
                if session:
                    session.transcript_json = st.session_state.responses
                    session.report_text = report
                    session.status = 'COMPLETED'
                    db.commit()
            finally:
                db.close()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")

if st.session_state.report:
    st.markdown("## 📊 Evaluation Report")
    st.markdown(st.session_state.report)
    with st.expander("Show Answer Key"):
        for i, (q, a) in enumerate(zip(st.session_state.questions, st.session_state.answers_key), start=1):
            st.markdown(f"**Q{i}. {q}**\n\n> Ideal Answer: {a}")

    with st.expander("Debug Log"):
        if st.session_state.debug_logs:
            st.code("\n".join(st.session_state.debug_logs), language="text")
        else:
            st.write("No logs yet.")

    if st.button("Start New Interview"):
        for key in ["session_id", "start_time", "questions", "answers_key", "responses", "current_index", "completed", "report"]:
            st.session_state[key] = None if key in ["session_id", "start_time", "report"] else [] if key in ["questions", "answers_key", "responses"] else 0 if key=="current_index" else False
        st.experimental_rerun()

# Auto-refresh timer every second during active interview
# (Removed aggressive immediate rerun; using st_autorefresh instead)
