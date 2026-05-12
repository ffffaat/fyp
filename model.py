from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Lecturer(db.Model, UserMixin):
    __tablename__ = 'lecturer'
    lecturer_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    students = db.relationship('Student', backref='owner_lecturer', lazy=True, cascade="all, delete-orphan")
    assignments = db.relationship('Assignment', backref='owner_lecturer', lazy=True, cascade="all, delete-orphan")
    quizzes = db.relationship('Quiz', backref='owner_lecturer', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return f"Lecturer:{self.lecturer_id}"

    def __repr__(self):
        return f"<Lecturer {self.lecturer_id} {self.email}>"


class Student(db.Model, UserMixin):
    __tablename__ = 'student'
    student_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    matrix_no = db.Column(db.String(20), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True)
    is_active = db.Column(db.Boolean, default=False)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturer.lecturer_id'), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password) if self.password_hash else False

    def get_id(self):
        return f"Student:{self.student_id}"

    def __repr__(self):
        return f"<Student {self.student_id} {self.email}>"


class Assignment(db.Model):
    __tablename__ = 'assignment'

    assignment_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    instruction = db.Column(db.Text, nullable=True)
    rubric_path = db.Column(db.String(255), nullable=True)
    answer_scheme_path = db.Column(db.String(255), nullable=True)
    due_date = db.Column(db.DateTime, nullable=False)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturer.lecturer_id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    submissions = db.relationship(
        "Submission",
        backref="assignment",
        lazy=True,
        cascade="all, delete-orphan"
    )


class Submission(db.Model):
    __tablename__ = "submission"

    submission_id = db.Column(db.Integer, primary_key=True)

    assignment_id = db.Column(
        db.Integer,
        db.ForeignKey("assignment.assignment_id", ondelete="CASCADE"),
        nullable=False
    )
    student_id = db.Column(
        db.Integer,
        db.ForeignKey("student.student_id", ondelete="CASCADE"),
        nullable=False
    )

    file_path = db.Column(db.String(255), nullable=False)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    ai_total_score = db.Column(db.Float, nullable=True)
    ai_status = db.Column(db.String(50), default="Not Evaluated")
    ai_feedback = db.Column(db.Text, nullable=True)

    # New fields for improved AI assignment marking
    reference_quality_score = db.Column(db.Float, nullable=True)
    ai_result_json = db.Column(db.Text, nullable=True)

    lecturer_final_score = db.Column(db.Float, nullable=True)
    lecturer_feedback = db.Column(db.Text, nullable=True)
    is_mark_released = db.Column(db.Boolean, default=False, nullable=False)
    endorsed_at = db.Column(db.DateTime, nullable=True)

    __table_args__ = (
        db.UniqueConstraint("assignment_id", "student_id", name="uq_assignment_student_submission"),
    )

    student = db.relationship("Student", backref="submissions")


class SubmissionCriterionScore(db.Model):
    __tablename__ = "submission_criterion_score"

    score_id = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(db.Integer, db.ForeignKey("submission.submission_id"), nullable=False)

    criterion_name = db.Column(db.String(255), nullable=False)
    section_name = db.Column(db.String(255), nullable=True)

    weight = db.Column(db.Float, nullable=False, default=0.0)
    max_scale = db.Column(db.Float, nullable=False, default=10.0)

    semantic_similarity = db.Column(db.Float, nullable=True)
    keyword_coverage = db.Column(db.Float, nullable=True)
    structure_score = db.Column(db.Float, nullable=True)

    raw_score = db.Column(db.Float, nullable=True)
    weighted_score = db.Column(db.Float, nullable=True)

    feedback = db.Column(db.Text, nullable=True)

    model_chunk_text = db.Column(db.Text, nullable=True)
    student_chunk_text = db.Column(db.Text, nullable=True)

    submission = db.relationship(
        "Submission",
        backref=db.backref("criterion_scores", lazy=True, cascade="all, delete-orphan")
    )


class Quiz(db.Model):
    __tablename__ = 'quiz'

    quiz_id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    is_published = db.Column(db.Boolean, default=False)
    lecturer_id = db.Column(db.Integer, db.ForeignKey('lecturer.lecturer_id'), nullable=False)
    concepts = db.Column(db.Text, nullable=True)
    questions = db.relationship(
        'Question',
        backref='parent_quiz',
        lazy=True,
        cascade='all, delete-orphan'
    )

    attempts = db.relationship(
        'QuizAttempt',
        backref='quiz',
        lazy=True,
        cascade='all, delete-orphan',
        passive_deletes=True
    )


class Question(db.Model):
    __tablename__ = 'question'
    question_id = db.Column(db.Integer, primary_key=True)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.quiz_id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)

    correct_answer = db.Column(db.String(255), nullable=False)
    distractor_1 = db.Column(db.String(255), nullable=False)
    distractor_2 = db.Column(db.String(255), nullable=False)
    distractor_3 = db.Column(db.String(255), nullable=False)

    explanation = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Question {self.question_id} quiz={self.quiz_id}>"


class QuizAttempt(db.Model):
    __tablename__ = 'quiz_attempt'

    attempt_id = db.Column(db.Integer, primary_key=True)

    student_id = db.Column(db.Integer, db.ForeignKey('student.student_id', ondelete='CASCADE'), nullable=False)
    quiz_id = db.Column(db.Integer, db.ForeignKey('quiz.quiz_id', ondelete='CASCADE'), nullable=False)

    score = db.Column(db.Float, nullable=True)
    total = db.Column(db.Integer, nullable=True)
    percent = db.Column(db.Float, nullable=True)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    answers = db.relationship(
        'AttemptAnswer',
        backref='attempt',
        lazy=True,
        cascade='all, delete-orphan',
        passive_deletes=True
    )


class AttemptAnswer(db.Model):
    __tablename__ = "attempt_answer"

    answer_id = db.Column(db.Integer, primary_key=True)

    attempt_id = db.Column(db.Integer, db.ForeignKey("quiz_attempt.attempt_id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("question.question_id"), nullable=False)

    chosen_answer = db.Column(db.String(255), nullable=False)
    is_correct = db.Column(db.Boolean, default=False, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("attempt_id", "question_id", name="uq_attempt_question_once"),
    )

    question = db.relationship("Question")