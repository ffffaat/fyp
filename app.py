import csv
import io
import os
import uuid
import json
import random

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from sqlalchemy.exc import IntegrityError
from model import db, Lecturer, Student, Assignment, Quiz, Question, QuizAttempt, AttemptAnswer, Submission, SubmissionCriterionScore
from nlg_quiz import QuizGenerationService, clean_text, generate_full_quiz_data, extract_key_entities, filter_valid_concepts
from config import Config
from datetime import datetime 
from nlp_assignment import evaluate_submission

# ----------------------------
# INITIAL SETUP
# ----------------------------
app = Flask(__name__)

# Initial configurations
app.config.from_object(Config)

# File Upload Configuration
UPLOAD_SCHEME_FOLDER = os.path.join("static", "uploads", "schemes")
app.config["UPLOAD_SCHEME_FOLDER"] = UPLOAD_SCHEME_FOLDER
os.makedirs(app.config["UPLOAD_SCHEME_FOLDER"], exist_ok=True)

UPLOAD_RUBRIC_FOLDER = os.path.join("static", "uploads", "rubrics")
app.config["UPLOAD_RUBRIC_FOLDER"] = UPLOAD_RUBRIC_FOLDER
os.makedirs(app.config["UPLOAD_RUBRIC_FOLDER"], exist_ok=True)

UPLOAD_ANSWER_SCHEME_FOLDER = os.path.join("static", "uploads", "answer_schemes")
app.config["UPLOAD_ANSWER_SCHEME_FOLDER"] = UPLOAD_ANSWER_SCHEME_FOLDER
os.makedirs(app.config["UPLOAD_ANSWER_SCHEME_FOLDER"], exist_ok=True)

UPLOAD_SUBMISSION_FOLDER = os.path.join("static", "uploads", "submissions")
app.config["UPLOAD_SUBMISSION_FOLDER"] = UPLOAD_SUBMISSION_FOLDER
os.makedirs(app.config["UPLOAD_SUBMISSION_FOLDER"], exist_ok=True)

# Initialize Extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ----------------------------
# HELPERS
# ----------------------------
# Ensure email is valid USM format based on role
def is_valid_usm_email(email: str, role: str) -> bool:
    if not email:
        return False
    email = email.strip().lower()
    if role == "Lecturer":
        return email.endswith("@usm.my")
    if role == "Student":
        return email.endswith("@student.usm.my")
    return False

@login_manager.user_loader
def load_user(user_id: str):
    """
    Expects user.get_id() format:
      Lecturer:<lecturer_id>
      Student:<student_id>
    """
    try:
        model_name, uid = user_id.split(":", 1)
        uid = int(uid)
        if model_name == "Lecturer":
            return db.session.get(Lecturer, uid)
        if model_name == "Student":
            return db.session.get(Student, uid)
    except Exception:
        return None
    return None

def _require_lecturer():
    if not isinstance(current_user, Lecturer):
        return False
    return True

def _require_student():
    if not isinstance(current_user, Student):
        return False
    return True

# ----------------------------
# ROUTES - AUTH
# ----------------------------
@app.route("/")
def home():
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        if not name or not email or not password:
            flash("Please fill in all fields.")
            return redirect(url_for("register"))

        if not is_valid_usm_email(email, "Lecturer"):
            flash("Invalid format. Lecturers must use name@usm.my.")
            return redirect(url_for("register"))

        if Lecturer.query.filter_by(email=email).first():
            flash("Email already registered!")
            return redirect(url_for("register"))

        new_lecturer = Lecturer(name=name, email=email)
        new_lecturer.set_password(password)

        db.session.add(new_lecturer)
        db.session.commit()

        login_user(new_lecturer)
        return redirect(url_for("lecturer_dashboard"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        user = Lecturer.query.filter_by(email=email).first() or Student.query.filter_by(email=email).first()

        if user and user.check_password(password):
            if isinstance(user, Student) and not user.is_active:
                flash("Please activate your student account first.")
                return redirect(url_for("activate"))

            login_user(user)
            return redirect(url_for("lecturer_dashboard" if isinstance(user, Lecturer) else "student_dashboard"))

        flash("Invalid email or password.")

    return render_template("login.html")

@app.route("/activate", methods=["GET", "POST"])
def activate():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""

        if not is_valid_usm_email(email, "Student"):
            flash("Invalid format. Students must use name@student.usm.my.")
            return redirect(url_for("activate"))

        if password != confirm:
            flash("Passwords do not match!")
            return redirect(url_for("activate"))

        student = Student.query.filter_by(email=email, is_active=False).first()
        if student:
            student.set_password(password)
            student.is_active = True
            db.session.commit()

            flash("Account activated! You can now log in.")
            return redirect(url_for("login"))

        flash("Invalid email or account already active.")
    return render_template("activate.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ----------------------------
# ROUTES - DASHBOARDS
# ----------------------------
@app.route("/lecturer/dashboard")
@login_required
def lecturer_dashboard():
    if not _require_lecturer():
        return redirect(url_for("login"))

    my_students = Student.query.filter_by(
        lecturer_id=current_user.lecturer_id
    ).all()

    active_count = sum(1 for s in my_students if s.is_active)
    total_count = len(my_students)

    # ----------------------------
    # Assignment data
    # ----------------------------
    my_assignments = Assignment.query.filter_by(
        lecturer_id=current_user.lecturer_id
    ).all()
    assignment_ids = [a.assignment_id for a in my_assignments]

    submissions = []
    if assignment_ids:
        submissions = Submission.query.filter(
            Submission.assignment_id.in_(assignment_ids)
        ).all()

    assignments_pending = sum(
        1 for s in submissions
        if (s.ai_status != "Evaluated" and not s.is_mark_released)
    )

    reviewed_assignments = sum(
        1 for s in submissions
        if s.ai_status == "Evaluated"
    )

    # Student assignment performance graph
    assignment_score_map = {}
    for s in submissions:
        if s.lecturer_final_score is not None:
            score = s.lecturer_final_score
        elif s.ai_total_score is not None:
            score = s.ai_total_score
        else:
            continue

        student_obj = s.student
        if student_obj:
            assignment_score_map.setdefault(student_obj.name, []).append(score)

    assignment_chart_labels = []
    assignment_chart_values = []

    for student_name, scores in assignment_score_map.items():
        if scores:
            assignment_chart_labels.append(student_name)
            assignment_chart_values.append(round(sum(scores) / len(scores), 2))

    # ----------------------------
    # Quiz data
    # ----------------------------
    my_quizzes = Quiz.query.filter_by(
        lecturer_id=current_user.lecturer_id
    ).all()

    published_quizzes = sum(1 for q in my_quizzes if q.is_published)

    quiz_ids = [q.quiz_id for q in my_quizzes]
    attempts = []
    if quiz_ids:
        attempts = QuizAttempt.query.filter(
            QuizAttempt.quiz_id.in_(quiz_ids)
        ).all()

    avg_quiz_score = None
    if attempts:
        valid_percents = [a.percent for a in attempts if a.percent is not None]
        if valid_percents:
            avg_quiz_score = round(sum(valid_percents) / len(valid_percents), 2)

    # Student quiz performance graph
    quiz_score_map = {}
    for a in attempts:
        if a.percent is None:
            continue
        student_obj = Student.query.get(a.student_id)
        if student_obj:
            quiz_score_map.setdefault(student_obj.name, []).append(a.percent)

    quiz_chart_labels = []
    quiz_chart_values = []

    for student_name, scores in quiz_score_map.items():
        if scores:
            quiz_chart_labels.append(student_name)
            quiz_chart_values.append(round(sum(scores) / len(scores), 2))

    return render_template(
        "lecturer_dashboard.html",
        name=current_user.name,
        total_count=total_count,
        active_count=active_count,
        students=my_students,
        assignments_pending=assignments_pending,
        reviewed_assignments=reviewed_assignments,
        published_quizzes=published_quizzes,
        avg_quiz_score=avg_quiz_score,
        assignment_chart_labels=assignment_chart_labels,
        assignment_chart_values=assignment_chart_values,
        quiz_chart_labels=quiz_chart_labels,
        quiz_chart_values=quiz_chart_values,
    )

@app.route("/student/dashboard")
@login_required
def student_dashboard():
    if not _require_student():
        return redirect(url_for("login"))

    # ----------------------------
    # Assignment data for this student
    # ----------------------------
    my_submissions = Submission.query.filter_by(
        student_id=current_user.student_id
    ).all()

    assignment_chart_labels = []
    assignment_chart_values = []

    for sub in my_submissions:
        assignment_obj = Assignment.query.get(sub.assignment_id)
        if not assignment_obj:
            continue

        if sub.lecturer_final_score is not None:
            score = sub.lecturer_final_score
        elif sub.ai_total_score is not None:
            score = sub.ai_total_score
        else:
            score = None

        if score is not None:
            assignment_chart_labels.append(assignment_obj.title)
            assignment_chart_values.append(round(score, 2))

    # ----------------------------
    # Quiz data for this student
    # ----------------------------
    my_attempts = QuizAttempt.query.filter_by(
        student_id=current_user.student_id
    ).all()

    quiz_chart_labels = []
    quiz_chart_values = []

    for attempt in my_attempts:
        quiz_obj = Quiz.query.get(attempt.quiz_id)
        if not quiz_obj or attempt.percent is None:
            continue

        quiz_chart_labels.append(quiz_obj.title)
        quiz_chart_values.append(round(attempt.percent, 2))

    # Stats
    total_assignments = len(my_submissions)
    completed_quizzes = len(my_attempts)

    avg_assignment_score = None
    if assignment_chart_values:
        avg_assignment_score = round(sum(assignment_chart_values) / len(assignment_chart_values), 2)

    avg_quiz_score = None
    if quiz_chart_values:
        avg_quiz_score = round(sum(quiz_chart_values) / len(quiz_chart_values), 2)

    return render_template(
        "student_dashboard.html",
        name=current_user.name,
        matrix_no=current_user.matrix_no,
        lecturer_name=current_user.owner_lecturer.name,
        total_assignments=total_assignments,
        completed_quizzes=completed_quizzes,
        avg_assignment_score=avg_assignment_score,
        avg_quiz_score=avg_quiz_score,
        assignment_chart_labels=assignment_chart_labels,
        assignment_chart_values=assignment_chart_values,
        quiz_chart_labels=quiz_chart_labels,
        quiz_chart_values=quiz_chart_values,
    )

# ----------------------------
# ROUTES - STUDENT MANAGEMENT
# ----------------------------
@app.route("/enroll_students", methods=["GET", "POST"])
@login_required
def enroll_students():
    if not _require_lecturer():
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename.endswith(".csv"):
            flash("Please upload a valid CSV file.")
            return redirect(url_for("enroll_students"))

        stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
        csv_input = csv.DictReader(stream)

        added = 0
        for row in csv_input:
            email = (row.get("email") or "").strip().lower()
            name = (row.get("name") or "").strip()
            matrix_no = (row.get("matrix_no") or "").strip()

            if not (email and name and matrix_no):
                continue

            if is_valid_usm_email(email, "Student") and not Student.query.filter_by(email=email).first():
                new_student = Student(
                    name=name,
                    email=email,
                    matrix_no=matrix_no,
                    is_active=False,
                    lecturer_id=current_user.lecturer_id,
                )
                db.session.add(new_student)
                added += 1

        db.session.commit()
        flash(f"Successfully enrolled {added} students.")

    enrolled_students = Student.query.filter_by(lecturer_id=current_user.lecturer_id).all()
    return render_template("enroll.html", students=enrolled_students)

@app.route("/delete_student/<int:student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    student = Student.query.filter_by(student_id=student_id, lecturer_id=current_user.lecturer_id).first()
    if student:
        db.session.delete(student)
        db.session.commit()
        flash("Student record removed.")
    else:
        flash("Student not found or you do not have permission.")
    return redirect(url_for("enroll_students"))

# ----------------------------
# ROUTES - QUIZ MANAGEMENT (NLG)
# ----------------------------
@app.route("/quiz_management")
@login_required
def quiz_management():
    if not _require_lecturer():
        return redirect(url_for("login"))

    my_quizzes = Quiz.query.filter_by(lecturer_id=current_user.lecturer_id)\
                           .order_by(Quiz.quiz_id.desc()).all()

    for qz in my_quizzes:
        try:
            qz.concepts_list = json.loads(qz.concepts) if getattr(qz, "concepts", None) else []
        except Exception:
            qz.concepts_list = []
        qz.question_count = len(qz.questions)

    return render_template("quiz_management.html", quizzes=my_quizzes)

@app.route("/upload_scheme", methods=["POST"])
@login_required
def upload_scheme():
    if not _require_lecturer():
        return redirect(url_for("login"))

    file = request.files.get("pdf_file")
    if not file or not file.filename.lower().endswith(".pdf"):
        flash("Please upload a valid PDF file.")
        return redirect(url_for("quiz_management"))

    safe_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = os.path.join(app.config["UPLOAD_SCHEME_FOLDER"], unique_name)
    file.save(save_path)

    try:
        reader = PdfReader(save_path)

        texts = []
        for i, page in enumerate(reader.pages):
            t = page.extract_text() or ""
            if i < 2:
                continue
            texts.append(t)

        full_text = " ".join(texts).strip()

        if not full_text:
            flash("Could not extract text from this PDF (may be scanned). Upload a text-based PDF.")
            return redirect(url_for("quiz_management"))

        full_text = clean_text(full_text)

        raw_concepts = extract_key_entities(full_text[:20000], top_k=20)
        concepts = filter_valid_concepts(raw_concepts, limit=8)

        if not concepts:
            flash("No valid key concepts detected. Try uploading more complete notes.")
            return redirect(url_for("quiz_management"))

        quiz_service = QuizGenerationService()
        seen_questions = set()
        seen_corrects = set()

        new_quiz = Quiz(
            title=f"AI Quiz: {safe_name}",
            lecturer_id=current_user.lecturer_id,
            concepts=json.dumps(concepts),
        )
        db.session.add(new_quiz)
        db.session.flush()

        created = 0

        for concept in concepts:
            quiz_bundle = quiz_service.generate_question_bundle(
                concept=concept,
                full_text=full_text,
                concept_pool=concepts
            )

            if not quiz_bundle:
                continue

            question_key = " ".join(quiz_bundle["question"].lower().split())
            correct_key = quiz_bundle["correct"].strip().lower()

            if question_key in seen_questions:
                continue
            if correct_key in seen_corrects:
                continue

            seen_questions.add(question_key)
            seen_corrects.add(correct_key)

            distractors = quiz_bundle.get("distractors", [])
            if len(distractors) < 3:
                continue

            new_question = Question(
                quiz_id=new_quiz.quiz_id,
                question_text=quiz_bundle["question"],
                correct_answer=quiz_bundle["correct"],
                distractor_1=distractors[0],
                distractor_2=distractors[1],
                distractor_3=distractors[2],
                explanation=quiz_bundle["explanation"],
            )
            db.session.add(new_question)
            created += 1

        if created == 0:
            db.session.rollback()
            flash("AI could not generate usable quiz questions from this PDF.")
            return redirect(url_for("quiz_management"))

        db.session.commit()
        flash(f"Success! AI generated {created} questions from {safe_name}.")

    except Exception as e:
        db.session.rollback()
        flash(f"AI Error: {str(e)}")

    return redirect(url_for("quiz_management"))

@app.route("/review_quiz/<int:quiz_id>")
@login_required
def review_quiz(quiz_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    return render_template("review_quiz.html", quiz=quiz)

@app.route("/quiz/<int:quiz_id>/save_review", methods=["POST"])
@login_required
def save_quiz_review(quiz_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    for q in quiz.questions:
        qt = " ".join((request.form.get(f"question_text_{q.question_id}") or "").split())
        ca = " ".join((request.form.get(f"correct_answer_{q.question_id}") or "").split())
        d1 = " ".join((request.form.get(f"distractor_1_{q.question_id}") or "").split())
        d2 = " ".join((request.form.get(f"distractor_2_{q.question_id}") or "").split())
        d3 = " ".join((request.form.get(f"distractor_3_{q.question_id}") or "").split())
        ex = " ".join((request.form.get(f"explanation_{q.question_id}") or "").split())

        if qt:
            q.question_text = qt

        if ca and d1 and d2 and d3:
            opts = [ca.lower(), d1.lower(), d2.lower(), d3.lower()]
            if len(set(opts)) < 4:
                flash(f"Question {q.question_id}: Options must be unique.")
                return redirect(url_for("review_quiz", quiz_id=quiz_id))

            q.correct_answer = ca
            q.distractor_1 = d1
            q.distractor_2 = d2
            q.distractor_3 = d3

        if ex:
            q.explanation = ex

    db.session.commit()
    flash("Saved changes to quiz questions.")
    return redirect(url_for("review_quiz", quiz_id=quiz_id))

@app.route("/quiz/<int:quiz_id>/question/<int:question_id>/delete", methods=["POST"])
@login_required
def delete_question(quiz_id, question_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    q = Question.query.filter_by(question_id=question_id, quiz_id=quiz.quiz_id).first_or_404()
    db.session.delete(q)
    db.session.commit()

    flash("Question deleted.")
    return redirect(url_for("review_quiz", quiz_id=quiz_id))

@app.route("/publish_quiz/<int:quiz_id>", methods=["POST"])
@login_required
def publish_quiz(quiz_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    if len(quiz.questions) == 0:
        flash("Cannot publish an empty quiz. Please generate or add questions first.")
        return redirect(url_for("review_quiz", quiz_id=quiz_id))

    quiz.is_published = True
    db.session.commit()

    flash(f"Quiz '{quiz.title}' is now live for your students!")
    return redirect(url_for("quiz_management"))

@app.route("/quiz/<int:quiz_id>/toggle_publish", methods=["POST"])
@login_required
def toggle_publish_quiz(quiz_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    if len(quiz.questions) == 0:
        flash("Cannot publish an empty quiz.")
        return redirect(url_for("quiz_management"))

    quiz.is_published = not quiz.is_published
    db.session.commit()

    flash("Quiz published." if quiz.is_published else "Quiz unpublished.")
    return redirect(url_for("quiz_management"))

@app.route("/delete_quiz/<int:quiz_id>", methods=["POST"])
@login_required
def delete_quiz(quiz_id):
    if not _require_lecturer():
        flash("Unauthorized action.")
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(quiz_id=quiz_id, lecturer_id=current_user.lecturer_id).first()
    if not quiz:
        flash("Quiz not found or you do not have permission to delete it.")
        return redirect(url_for("quiz_management"))

    try:
        quiz_title = quiz.title
        db.session.delete(quiz)
        db.session.commit()
        flash(f"Quiz '{quiz_title}' and all its AI questions have been deleted.")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting quiz: {str(e)}")

    return redirect(url_for("quiz_management"))

# ----------------------------
# ROUTES - QUIZ (STUDENT VIEW)
# ----------------------------
@app.route("/student/quizzes")
@login_required
def student_quiz_list():
    if not isinstance(current_user, Student):
        return redirect(url_for("login"))

    quizzes = (
        Quiz.query.filter_by(
            lecturer_id=current_user.lecturer_id,
            is_published=True
        )
        .order_by(Quiz.quiz_id.desc())
        .all()
    )

    attempts = QuizAttempt.query.filter_by(student_id=current_user.student_id).all()
    attempt_map = {a.quiz_id: a for a in attempts}

    for qz in quizzes:
        qz.question_count = len(qz.questions)
        qz.my_attempt = attempt_map.get(qz.quiz_id)

    return render_template("student_quiz_list.html", quizzes=quizzes)


@app.route("/student/quiz/<int:quiz_id>")
@login_required
def student_take_quiz(quiz_id):
    if not isinstance(current_user, Student):
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id,
        is_published=True
    ).first_or_404()

    existing = QuizAttempt.query.filter_by(
        student_id=current_user.student_id,
        quiz_id=quiz.quiz_id
    ).first()

    if existing:
        flash("You already submitted this quiz. You can review your answers.")
        return redirect(url_for("student_review_quiz", quiz_id=quiz.quiz_id))

    questions_payload = []
    for q in quiz.questions:
        options = [
            q.correct_answer,
            q.distractor_1,
            q.distractor_2,
            q.distractor_3
        ]
        random.shuffle(options)

        questions_payload.append({
            "question_id": q.question_id,
            "question_text": q.question_text,
            "options": options
        })

    return render_template(
        "student_take_quiz.html",
        quiz=quiz,
        questions=questions_payload
    )


@app.route("/student/quiz/<int:quiz_id>/submit", methods=["POST"])
@login_required
def student_submit_quiz(quiz_id):
    if not isinstance(current_user, Student):
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id,
        is_published=True
    ).first_or_404()

    existing = QuizAttempt.query.filter_by(
        student_id=current_user.student_id,
        quiz_id=quiz.quiz_id
    ).first()

    if existing:
        flash("You already submitted this quiz.")
        return redirect(url_for("student_review_quiz", quiz_id=quiz.quiz_id))

    total = len(quiz.questions)
    score = 0

    # First, calculate answers in memory
    answer_rows = []

    for q in quiz.questions:
        chosen = (request.form.get(f"q_{q.question_id}") or "").strip()
        is_correct = (chosen == q.correct_answer)

        if is_correct:
            score += 1

        answer_rows.append({
            "question_id": q.question_id,
            "chosen_answer": chosen[:255],
            "is_correct": is_correct
        })

    percent = round((score / total) * 100, 1) if total else 0.0

    attempt = QuizAttempt(
        student_id=current_user.student_id,
        quiz_id=quiz.quiz_id,
        score=score,
        total=total,
        percent=percent
    )

    try:
        db.session.add(attempt)
        db.session.flush()  # get attempt_id

        for row in answer_rows:
            db.session.add(AttemptAnswer(
                attempt_id=attempt.attempt_id,
                question_id=row["question_id"],
                chosen_answer=row["chosen_answer"],
                is_correct=row["is_correct"]
            ))

        db.session.commit()
        flash("Quiz submitted! You can review your answers.")

    except IntegrityError:
        db.session.rollback()
        flash("You already submitted this quiz.")
        return redirect(url_for("student_review_quiz", quiz_id=quiz.quiz_id))

    except Exception as e:
        db.session.rollback()
        flash(f"Error submitting quiz: {str(e)}")
        return redirect(url_for("student_take_quiz", quiz_id=quiz.quiz_id))

    return redirect(url_for("student_review_quiz", quiz_id=quiz.quiz_id))


@app.route("/student/quiz/<int:quiz_id>/review")
@login_required
def student_review_quiz(quiz_id):
    if not isinstance(current_user, Student):
        return redirect(url_for("login"))

    quiz = Quiz.query.filter_by(
        quiz_id=quiz_id,
        lecturer_id=current_user.lecturer_id,
        is_published=True
    ).first_or_404()

    attempt = QuizAttempt.query.filter_by(
        student_id=current_user.student_id,
        quiz_id=quiz.quiz_id
    ).first_or_404()

    answers = AttemptAnswer.query.filter_by(attempt_id=attempt.attempt_id).all()
    ans_map = {a.question_id: a for a in answers}

    results = []
    for q in quiz.questions:
        a = ans_map.get(q.question_id)
        results.append({
            "question_text": q.question_text,
            "chosen": a.chosen_answer if a else "",
            "is_correct": a.is_correct if a else False,
            "correct": q.correct_answer,
            "explanation": q.explanation
        })

    return render_template(
        "student_quiz_review.html",
        quiz=quiz,
        attempt=attempt,
        results=results
    )

# ----------------------------
# ROUTES - ASSIGNMENT MANAGEMENT
# ----------------------------
@app.route("/assignment_management", methods=["GET", "POST"])
@login_required
def assignment_management():
    if not _require_lecturer():
        return redirect(url_for("login"))

    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        instruction = (request.form.get("instruction") or "").strip()
        due_date_raw = (request.form.get("due_date") or "").strip()

        rubric_file = request.files.get("rubric_file")
        answer_scheme_file = request.files.get("answer_scheme_file")

        if not title or not due_date_raw:
            flash("Please fill in the assignment title and due date.")
            return redirect(url_for("assignment_management"))

        try:
            due_date = datetime.strptime(due_date_raw, "%Y-%m-%dT%H:%M")
        except ValueError:
            flash("Invalid due date format.")
            return redirect(url_for("assignment_management"))

        rubric_path = None
        if rubric_file and rubric_file.filename:
            if not (rubric_file.filename.lower().endswith(".xlsx") or rubric_file.filename.lower().endswith(".xls")):
                flash("Rubric file must be an Excel file (.xlsx or .xls).")
                return redirect(url_for("assignment_management"))

            safe_name = secure_filename(rubric_file.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            save_path = os.path.join(app.config["UPLOAD_RUBRIC_FOLDER"], unique_name)
            rubric_file.save(save_path)

            rubric_path = os.path.join("uploads", "rubrics", unique_name).replace("\\", "/")

        answer_scheme_path = None
        if answer_scheme_file and answer_scheme_file.filename:
            if not answer_scheme_file.filename.lower().endswith(".pdf"):
                flash("Answer scheme file must be a PDF.")
                return redirect(url_for("assignment_management"))

            safe_name = secure_filename(answer_scheme_file.filename)
            unique_name = f"{uuid.uuid4().hex}_{safe_name}"
            save_path = os.path.join(app.config["UPLOAD_ANSWER_SCHEME_FOLDER"], unique_name)
            answer_scheme_file.save(save_path)
            answer_scheme_path = os.path.join("uploads", "answer_schemes", unique_name).replace("\\", "/")

        new_assignment = Assignment(
            title=title,
            instruction=instruction,
            rubric_path=rubric_path,
            answer_scheme_path=answer_scheme_path,
            due_date=due_date,
            lecturer_id=current_user.lecturer_id
        )

        db.session.add(new_assignment)
        db.session.commit()

        flash("Assignment created successfully.")
        return redirect(url_for("assignment_management"))

    assignments = Assignment.query.filter_by(
        lecturer_id=current_user.lecturer_id
    ).order_by(Assignment.due_date.asc()).all()

    return render_template("assignment_management.html", assignments=assignments)

@app.route("/delete_assignment/<int:assignment_id>", methods=["POST"])
@login_required
def delete_assignment(assignment_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    assignment = Assignment.query.filter_by(
        assignment_id=assignment_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    try:
        db.session.delete(assignment)
        db.session.commit()
        flash("Assignment and related submissions deleted successfully.")
    except Exception as e:
        db.session.rollback()
        flash(f"Error deleting assignment: {str(e)}")

    return redirect(url_for("assignment_management"))

@app.route("/student_assignments", methods=["GET", "POST"])
@login_required
def student_assignments():
    if not _require_student():
        return redirect(url_for("login"))

    if request.method == "POST":
        assignment_id = request.form.get("assignment_id")
        file = request.files.get("submission_file")

        if not assignment_id:
            flash("Invalid assignment.")
            return redirect(url_for("student_assignments"))

        assignment = Assignment.query.filter_by(
            assignment_id=assignment_id,
            lecturer_id=current_user.lecturer_id
        ).first()

        if not assignment:
            flash("Assignment not found.")
            return redirect(url_for("student_assignments"))

        if assignment.due_date and datetime.utcnow() > assignment.due_date:
            flash("Submission deadline has passed.")
            return redirect(url_for("student_assignments"))

        if not file or not file.filename:
            flash("Please upload a submission file.")
            return redirect(url_for("student_assignments"))

        if not file.filename.lower().endswith(".pdf"):
            flash("Submission file must be a PDF.")
            return redirect(url_for("student_assignments"))

        safe_name = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = os.path.join(app.config["UPLOAD_SUBMISSION_FOLDER"], unique_name)
        file.save(save_path)

        file_path = os.path.join("uploads", "submissions", unique_name).replace("\\", "/")

        existing_submission = Submission.query.filter_by(
            assignment_id=assignment.assignment_id,
            student_id=current_user.student_id
        ).first()

        if existing_submission:
            existing_submission.file_path = file_path
            existing_submission.submitted_at = datetime.utcnow()
            flash("Submission updated successfully.")
        else:
            new_submission = Submission(
                assignment_id=assignment.assignment_id,
                student_id=current_user.student_id,
                file_path=file_path
            )
            db.session.add(new_submission)
            flash("Assignment submitted successfully.")

        db.session.commit()
        return redirect(url_for("student_assignments"))

    assignments = Assignment.query.filter_by(
        lecturer_id=current_user.lecturer_id
    ).order_by(Assignment.due_date.asc()).all()

    all_submissions = Submission.query.filter_by(
        student_id=current_user.student_id
    ).all()

    submission_map = {s.assignment_id: s for s in all_submissions}

    # Parse AI result JSON for criterion-level feedback
    ai_result_map = {}
    for sub in all_submissions:
        parsed_result = None
        if sub.ai_result_json:
            try:
                parsed_result = json.loads(sub.ai_result_json)
            except Exception:
                parsed_result = None
        ai_result_map[sub.assignment_id] = parsed_result

    return render_template(
        "student_assignments.html",
        assignments=assignments,
        submission_map=submission_map,
        ai_result_map=ai_result_map
    )

@app.route("/assignment/<int:assignment_id>/submissions")
@login_required
def view_submissions(assignment_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    assignment = Assignment.query.filter_by(
        assignment_id=assignment_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    submissions = (
        Submission.query
        .filter_by(assignment_id=assignment.assignment_id)
        .join(Student, Submission.student_id == Student.student_id)
        .order_by(Submission.submitted_at.desc())
        .all()
    )

    return render_template(
        "view_submissions.html",
        assignment=assignment,
        submissions=submissions
    )

@app.route("/submission/<int:submission_id>/evaluate", methods=["POST"])
@login_required
def evaluate_submission_route(submission_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.filter_by(
        assignment_id=submission.assignment_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    if not assignment.rubric_path:
        flash("No rubric file uploaded for this assignment.")
        return redirect(url_for("view_submissions", assignment_id=assignment.assignment_id))

    if not assignment.answer_scheme_path:
        flash("No answer scheme uploaded for this assignment.")
        return redirect(url_for("view_submissions", assignment_id=assignment.assignment_id))

    student_pdf_path = os.path.join("static", submission.file_path)
    rubric_xlsx_path = os.path.join("static", assignment.rubric_path)
    model_answer_pdf_path = os.path.join("static", assignment.answer_scheme_path)

    try:
        result = evaluate_submission(
            student_pdf_path=student_pdf_path,
            rubric_xlsx_path=rubric_xlsx_path,
            reference_pdf_paths=[model_answer_pdf_path],
        )

        SubmissionCriterionScore.query.filter_by(
            submission_id=submission.submission_id
        ).delete()

        for row in result["criteria_results"]:
            reference_chunks_flat = []

            for group in row.get("reference_chunks", []):
                if isinstance(group, list):
                    reference_chunks_flat.extend(group)
                elif isinstance(group, str):
                    reference_chunks_flat.append(group)

            db.session.add(SubmissionCriterionScore(
                submission_id=submission.submission_id,
                criterion_name=row.get("criterion_name", "Unknown Criterion"),
                section_name="Hybrid rubric semantic scoring",
                weight=row.get("weight", 0.0),
                max_scale=row.get("max_scale", 0.0),

                # Mapping new evaluator fields into existing DB columns
                semantic_similarity=row.get("best_reference_similarity", 0.0),
                keyword_coverage=row.get("optional_term_coverage", 0.0),
                structure_score=row.get("breadth_score", 0.0),

                raw_score=row.get("raw_score", 0.0),
                weighted_score=row.get("weighted_score", 0.0),
                feedback=row.get("feedback", ""),

                model_chunk_text="\n\n".join(reference_chunks_flat),
                student_chunk_text="\n\n".join(row.get("student_chunks", [])),
            ))

        submission.ai_total_score = result.get("final_score")
        submission.reference_quality_score = result.get("reference_quality_score")
        submission.ai_result_json = json.dumps(result)
        submission.ai_status = "Evaluated"
        submission.ai_feedback = (
            f"AI evaluated successfully. "
            f"Final={result.get('final_score', 0)}%, "
            f"Content={result.get('final_content_score', 0)}%, "
            f"Reference={result.get('reference_quality_score', 0)}%."
        )

        db.session.commit()
        flash("Submission evaluated successfully.")

    except Exception as e:
        import traceback
        traceback.print_exc()

        db.session.rollback()

        submission.ai_status = "Failed"
        submission.ai_feedback = f"Evaluation failed: {str(e)}"
        db.session.add(submission)
        db.session.commit()

        flash(f"Evaluation failed: {str(e)}")

    return redirect(url_for("view_submissions", assignment_id=assignment.assignment_id))

@app.route("/submission/<int:submission_id>/breakdown", methods=["GET", "POST"])
@login_required
def submission_breakdown(submission_id):
    if not _require_lecturer():
        return redirect(url_for("login"))

    submission = Submission.query.get_or_404(submission_id)
    assignment = Assignment.query.filter_by(
        assignment_id=submission.assignment_id,
        lecturer_id=current_user.lecturer_id
    ).first_or_404()

    if request.method == "POST":
        final_score_raw = (request.form.get("lecturer_final_score") or "").strip()
        lecturer_feedback = (request.form.get("lecturer_feedback") or "").strip()
        action = request.form.get("action")

        try:
            final_score = float(final_score_raw) if final_score_raw else None
        except ValueError:
            flash("Final score must be a valid number.")
            return redirect(url_for("submission_breakdown", submission_id=submission.submission_id))

        if final_score is not None and (final_score < 0 or final_score > 100):
            flash("Final score must be between 0 and 100.")
            return redirect(url_for("submission_breakdown", submission_id=submission.submission_id))

        submission.lecturer_final_score = final_score
        submission.lecturer_feedback = lecturer_feedback

        if action == "save":
            db.session.commit()
            flash("Lecturer review saved successfully.")

        elif action == "endorse":
            if final_score is None:
                flash("Please enter a final score before endorsement.")
                return redirect(url_for("submission_breakdown", submission_id=submission.submission_id))

            submission.is_mark_released = True
            submission.endorsed_at = datetime.utcnow()
            db.session.commit()
            flash("Mark endorsed and released to student.")

        elif action == "unrelease":
            submission.is_mark_released = False
            submission.endorsed_at = None
            db.session.commit()
            flash("Mark has been hidden from student.")

        return redirect(url_for("submission_breakdown", submission_id=submission.submission_id))

    scores = SubmissionCriterionScore.query.filter_by(
        submission_id=submission.submission_id
    ).all()
    
    ai_result = {}
    try:
        ai_result = json.loads(submission.ai_result_json) if submission.ai_result_json else {}
    except Exception:
        ai_result = {}

    return render_template(
        "submission_breakdown.html",
        submission=submission,
        assignment=assignment,
        scores=scores,
        ai_result=ai_result,
    )

@app.route("/student/submission/<int:assignment_id>/result")
@login_required
def student_submission_result(assignment_id):
    if not _require_student():
        return redirect(url_for("login"))

    submission = Submission.query.filter_by(
        assignment_id=assignment_id,
        student_id=current_user.student_id
    ).first_or_404()

    if not submission.is_mark_released:
        flash("Your mark has not been released by the lecturer yet.")
        return redirect(url_for("student_assignments"))

    return render_template(
        "student_submission_result.html",
        submission=submission
    )
# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)