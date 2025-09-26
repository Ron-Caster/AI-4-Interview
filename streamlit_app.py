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
    return scored_skills[:10]

def generate_questions_and_answers(skills: List[str], difficulty: str, max_items: int = 10):
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
    def _truncate(text: str, max_chars: int = 700) -> str:
        if not text:
            return ""
        return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

    # Build compact prompt to reduce token usage
    compact_lines = []
    for i, item in enumerate(transcript):
        q = _truncate(item.get('question', ''), 240)
        a = _truncate(item.get('answer', ''), 600)
        
        # Build comprehensive detection info
        total_time = item.get('total_time', item.get('time_taken', 'N/A'))
        copy_score = item.get('copy_paste_score', 0)
        suspicion = item.get('copy_paste_suspicion', 'UNKNOWN')
        detection_reasons = item.get('detection_reasons', [])
        
        if isinstance(total_time, (int, float)):
            time_str = f" (Time: {total_time:.1f}s | Copy-Paste Score: {copy_score}/100 | Suspicion: {suspicion})"
            if detection_reasons:
                time_str += f"\nDetection Flags: {'; '.join(detection_reasons[:2])}"
        else:
            time_str = " (Time: N/A | Suspicion: UNKNOWN)"
            
        compact_lines.append(f"Q{i+1}: {q}{time_str}\nA{i+1}: {a}")
    transcript_blob = "\n\n".join(compact_lines)
    answer_blob = "\n".join([f"A{i+1}*: {_truncate(ans, 600)}" for i, ans in enumerate(answer_key)])

    prompt_text = """
You are an expert technical interviewer providing feedback for candidate {candidate_name}.
Evaluate the candidate and pay CLOSE ATTENTION to copy-paste detection scores and flags.

COPY-PASTE DETECTION SYSTEM:
Each answer has been analyzed with a comprehensive detection algorithm that checks:
- Completion speed vs answer complexity
- Character-per-second typing rates  
- Formatting patterns typical of prepared text
- Technical terminology density vs completion time
- Answer structure and completeness patterns

SUSPICION LEVELS:
- HIGH (Score 70-100): Very likely copy-paste - flag this prominently
- MEDIUM (Score 40-69): Suspicious patterns - investigate further
- LOW (Score 0-39): Appears to be genuine typing

IMPORTANT: If ANY answer shows HIGH suspicion, mention this prominently in your overall summary.
For MEDIUM/HIGH suspicion answers, be more critical of the content quality.

Output Format:
- **Overall Summary** (2-4 sentences) + **Final Score (/10)**
- **Copy-Paste Assessment**: Overall suspicion level across all answers
- For each question: **Q#**, **Time & Detection**, **Ideal Answer**, **Candidate Summary**, **Score (0-10)**, **Authenticity**, **Feedback**

Use Markdown formatting.

Transcript (with detection analysis):
{transcript_text}

Ideal Answers:
{answer_key_text}
"""

    models_to_try = [
        "groq/compound",            # primary (as configured elsewhere)
        "qwen/qwen3-32b",           # fallback 1
        "llama-3.1-8b-instant"      # fallback 2 (lighter)
    ]

    last_err = None
    for model_name in models_to_try:
        try:
            llm = get_llm(model_name=model_name, temperature=0)
            chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()
            return chain.invoke({
                "candidate_name": candidate_name,
                "transcript_text": transcript_blob,
                "answer_key_text": answer_blob
            })
        except Exception as e:
            last_err = e
            # If rate limit error, continue to next model
            if "rate limit" in str(e).lower() or "429" in str(e):
                continue
            # Other errors: try next model as well
            continue
    # If all models failed, raise the last error up to caller
    raise last_err if last_err else RuntimeError("Evaluation failed with all models")

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
if "question_start_time" not in st.session_state:
    st.session_state.question_start_time = None
if "typing_patterns" not in st.session_state:
    st.session_state.typing_patterns = []
if "first_keystroke_time" not in st.session_state:
    st.session_state.first_keystroke_time = None
if "paste_events" not in st.session_state:
    st.session_state.paste_events = []
if "keystroke_intervals" not in st.session_state:
    st.session_state.keystroke_intervals = []

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

            skill_slice = skills[:15] if skills else []
            if len(skill_slice) < 15:
                default_skills = ["problem solving", "system design", "python basics", "data structures", "algorithms", "software engineering", "database design", "web development", "cloud computing", "security", "testing", "version control", "api design", "scalability"]
                for i in range(15 - len(skill_slice)):
                    skill_slice.append(default_skills[i % len(default_skills)])

            st.write("Generating questions and ideal answers:")
            progress = st.progress(0)
            questions = []
            answers_key = []
            total = 15

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
            st.session_state.question_start_time = time.time()
            st.session_state.candidate_name = candidate_name
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

        # Question form ensures atomic capture of latest text on submit
        form_key = f"qa_form_{idx}"
        answer_key = f"answer_{idx}"
        with st.form(key=form_key, clear_on_submit=False):
            # Initialize answer state if needed
            if answer_key not in st.session_state:
                if idx < len(st.session_state.responses):
                    st.session_state[answer_key] = st.session_state.responses[idx]["answer"]
                else:
                    st.session_state[answer_key] = ""

            st.text_area(
                "Your Answer",
                key=answer_key,
                height=220,
                placeholder="Type your answer here."
            )
            
            # Advanced copy-paste detection JavaScript
            components.html(
                f"""
                <div id="detection-{idx}" style="display:none;"></div>
                <script>
                let questionIndex = {idx};
                let questionStartTime = {st.session_state.question_start_time};
                let firstKeystroke = false;
                let keystrokeData = [];
                let pasteDetected = false;
                let largeInputDetected = false;
                
                function detectCopyPaste() {{
                    const textarea = findTextarea();
                    if (!textarea) return;
                    
                    let lastLength = 0;
                    let lastTime = Date.now();
                    
                    // Detect paste events
                    textarea.addEventListener('paste', function(e) {{
                        pasteDetected = true;
                        const pasteTime = Date.now() / 1000;
                        const thinkTime = questionStartTime ? pasteTime - questionStartTime : 0;
                        sessionStorage.setItem('paste_detected_{idx}', 'true');
                        sessionStorage.setItem('paste_think_time_{idx}', thinkTime.toString());
                        console.log('PASTE DETECTED at question {idx}');
                    }});
                    
                    // Monitor typing patterns
                    textarea.addEventListener('input', function(e) {{
                        const now = Date.now();
                        const currentLength = textarea.value.length;
                        const lengthDelta = currentLength - lastLength;
                        const timeDelta = now - lastTime;
                        
                        if (!firstKeystroke && questionStartTime) {{
                            const thinkingTime = now/1000 - questionStartTime;
                            sessionStorage.setItem('thinking_time_{idx}', thinkingTime.toString());
                            firstKeystroke = true;
                        }}
                        
                        // Detect large sudden text additions (likely paste)
                        if (lengthDelta > 20 && timeDelta < 100) {{
                            largeInputDetected = true;
                            sessionStorage.setItem('large_input_{idx}', 'true');
                        }}
                        
                        // Track keystroke intervals for rhythm analysis
                        if (lengthDelta > 0 && timeDelta < 5000) {{
                            keystrokeData.push({{
                                'delta': lengthDelta,
                                'time': timeDelta,
                                'wpm': (lengthDelta / 5) / (timeDelta / 60000)
                            }});
                        }}
                        
                        lastLength = currentLength;
                        lastTime = now;
                        
                        // Store keystroke data
                        sessionStorage.setItem('keystroke_data_{idx}', JSON.stringify(keystrokeData));
                    }});
                    
                    // Monitor focus changes (switching to other apps)
                    let focusLost = false;
                    textarea.addEventListener('blur', function() {{
                        focusLost = true;
                        sessionStorage.setItem('focus_lost_{idx}', 'true');
                    }});
                }}
                
                function findTextarea() {{
                    const textareas = parent.document.querySelectorAll('textarea');
                    for (let ta of textareas) {{
                        if (ta.placeholder === 'Type your answer here.') return ta;
                    }}
                    return null;
                }}
                
                setTimeout(detectCopyPaste, 100);
                </script>
                """,
                height=1
            )

            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                save_next = st.form_submit_button("Save & Next", disabled=remaining==0, use_container_width=True)
            with c2:
                save_only = st.form_submit_button("Save (Stay)", use_container_width=True)
            with c3:
                finish_now = st.form_submit_button("Finish Now", use_container_width=True)

        # Handle form outcomes after rerun
        def _persist_current_answer(time_taken=None):
            current_answer = st.session_state.get(answer_key, "")
            response = {
                "question": st.session_state.questions[idx],
                "answer": current_answer
            }
            
            if time_taken is not None:
                response["total_time"] = time_taken
                
                # AGGRESSIVE COPY-PASTE DETECTION
                copy_paste_score = 0
                detection_reasons = []
                
                # 1. Check for extremely fast completion
                if time_taken < 15 and len(current_answer.strip()) > 100:
                    copy_paste_score += 40
                    detection_reasons.append("Extremely fast completion for long answer")
                
                # 2. Check character-per-second ratio
                if current_answer and time_taken > 0:
                    chars_per_sec = len(current_answer) / time_taken
                    if chars_per_sec > 8:  # Very fast typing (>8 chars/sec = ~96 WPM)
                        copy_paste_score += 30
                        detection_reasons.append(f"Suspiciously fast typing: {chars_per_sec:.1f} chars/sec")
                
                # 3. Check for perfect formatting patterns
                if current_answer:
                    formatting_indicators = [
                        '\n-' in current_answer,  # Bullet points
                        '\n1.' in current_answer,  # Numbered lists
                        current_answer.count('\n') > 3,  # Multiple paragraphs
                        len([w for w in current_answer.split() if len(w) > 8]) > len(current_answer.split()) * 0.3  # Complex words
                    ]
                    if sum(formatting_indicators) >= 2:
                        copy_paste_score += 25
                        detection_reasons.append("Complex formatting unlikely for live typing")
                
                # 4. Check for minimal thinking time with complex answer
                thinking_time = min(time_taken * 0.15, 10)  # Estimate
                if thinking_time < 5 and len(current_answer.split()) > 30:
                    copy_paste_score += 35
                    detection_reasons.append("Minimal thinking time for complex answer")
                
                # 5. Check answer structure and completeness
                if current_answer:
                    sentences = current_answer.split('. ')
                    if len(sentences) > 3 and all(len(s.split()) > 5 for s in sentences[:3]):
                        if time_taken < 30:
                            copy_paste_score += 20
                            detection_reasons.append("Well-structured answer completed too quickly")
                
                # 6. Check for technical accuracy vs speed mismatch
                technical_terms = ['algorithm', 'database', 'framework', 'implementation', 'architecture', 
                                 'optimization', 'scalability', 'performance', 'security', 'API']
                tech_term_count = sum(1 for term in technical_terms if term.lower() in current_answer.lower())
                if tech_term_count >= 3 and time_taken < 25:
                    copy_paste_score += 25
                    detection_reasons.append("Technical terminology used with suspiciously fast completion")
                
                response["thinking_time"] = thinking_time
                response["typing_time"] = time_taken - thinking_time
                response["copy_paste_score"] = copy_paste_score
                response["detection_reasons"] = detection_reasons
                
                # Determine suspicion level
                if copy_paste_score >= 70:
                    response["copy_paste_suspicion"] = "HIGH"
                elif copy_paste_score >= 40:
                    response["copy_paste_suspicion"] = "MEDIUM"
                else:
                    response["copy_paste_suspicion"] = "LOW"
                
                response["copy_paste_suspected"] = copy_paste_score >= 50
                
                # Calculate typing speed for display
                if current_answer and time_taken > 0:
                    words = len(current_answer.split())
                    response["typing_speed_wpm"] = (words / time_taken) * 60
                    
            if idx < len(st.session_state.responses):
                st.session_state.responses[idx].update(response)
            else:
                st.session_state.responses.append(response)

        if save_only:
            _persist_current_answer()
            st.success("Saved")
        if save_next:
            time_taken = time.time() - st.session_state.question_start_time
            _persist_current_answer(time_taken=time_taken)
            st.session_state.current_index += 1
            st.session_state.question_start_time = time.time()
            st.rerun()
        if finish_now:
            time_taken = time.time() - st.session_state.question_start_time
            _persist_current_answer(time_taken=time_taken)
            st.session_state.completed = True
            st.rerun()
    else:
        st.success("All questions answered. Click 'Generate Report' below to evaluate.")
        if st.button("Generate Report"):
            # Ensure last answer is saved if user reached end without pressing Save
            last_idx = len(st.session_state.questions) - 1
            if last_idx >= 0:
                last_key = f"answer_{last_idx}"
                if last_key in st.session_state:
                    if last_idx < len(st.session_state.responses):
                        st.session_state.responses[last_idx]["answer"] = st.session_state[last_key]
                    else:
                        st.session_state.responses.append({
                            "question": st.session_state.questions[last_idx],
                            "answer": st.session_state[last_key]
                        })
            st.session_state.completed = True

# Evaluation
if st.session_state.completed and st.session_state.report is None:
    with st.spinner("Evaluating responses and generating feedback + answer key..."):
        try:
            report = evaluate_transcript(
                st.session_state.responses,
                (st.session_state.candidate_name or candidate_name or "Candidate"),
                st.session_state.answers_key
            )
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
    # Export button to download report as .txt
    export_name = f"evaluation_report_{(st.session_state.candidate_name or 'candidate').replace(' ', '_')}.txt"
    st.download_button(
        label="Download Report (.txt)",
        data=st.session_state.report,
        file_name=export_name,
        mime="text/plain"
    )
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
        for key in ["session_id", "start_time", "questions", "answers_key", "responses", "current_index", "completed", "report", "candidate_name", "debug_logs", "question_start_time", "typing_patterns", "first_keystroke_time", "paste_events", "keystroke_intervals"]:
            if key in ["session_id", "start_time", "report", "question_start_time", "first_keystroke_time"]:
                st.session_state[key] = None
            elif key in ["questions", "answers_key", "responses", "debug_logs", "typing_patterns", "paste_events", "keystroke_intervals"]:
                st.session_state[key] = []
            elif key == "current_index":
                st.session_state[key] = 0
            elif key == "candidate_name":
                st.session_state[key] = ""
            else:
                st.session_state[key] = False
        st.rerun()

# Auto-refresh timer every second during active interview
# (Removed aggressive immediate rerun; using st_autorefresh instead)
