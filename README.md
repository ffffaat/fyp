# AI-Assisted Platform for Data Visualization and Visual Analytics Course

This project is an AI-powered EdTech platform developed specifically for the **Data Visualization and Visual Analytics** course. It is designed to support lecturers and students in assignment management, automated report evaluation, quiz generation, and performance monitoring.

The platform integrates **Natural Language Processing (NLP)** for AI-assisted assignment marking and feedback generation, and **Natural Language Generation (NLG)** for generating quiz questions from lecturer-uploaded notes.

---

## Project Overview

In courses that involve analytical reports and visualization-based assignments, lecturers often spend a significant amount of time marking student submissions, checking references, preparing quizzes, and monitoring student performance. This system aims to reduce repetitive academic workload while providing students with clearer feedback and better learning progress visibility.

For now, the platform is designed specifically for the **Data Visualization and Visual Analytics** course. However, it has the potential to be expanded to other courses involving report writing, project documentation, reflective writing, or analytical assessment.

---

## Main Features

### 1. AI Assignment Marking

The AI Assignment Marking feature helps lecturers evaluate student reports automatically. The system uses NLP to extract and analyze student submissions, compare them with the uploaded rubric and model answer, calculate rubric-based scores, and generate structured feedback.

Lecturers can review and endorse the AI-generated marks before releasing them to students.

### 2. Reference and In-text Citation Checker

This feature evaluates the quality and consistency of references in student reports. The system detects the reference section, identifies reference entries, checks in-text citations, and measures citation alignment.

It also evaluates source quality based on elements such as author, year, title, DOI or link, credible source indicators, and weak source indicators.

### 3. Quiz Generator

The Quiz Generator helps lecturers create quiz questions automatically from uploaded lecture notes. The system uses NLG to extract important concepts, retrieve relevant passages, and generate multiple-choice questions with correct answers, distractors, explanations, and source snippets.

Lecturers can review the generated questions before publishing the quiz to students.

### 4. Comprehensive Dashboard

The dashboard provides performance monitoring features for both lecturers and students.

Lecturers can view class-level performance, assignment progress, quiz results, and student achievement trends. Students can track their own marks, quiz scores, feedback, and learning progress.

---

## System Modules

The system consists of four main modules:

1. **Authentication and Profile Module**
   - Lecturer registration
   - Lecturer-controlled student enrolment
   - Student account activation
   - Role-based login and dashboard redirection

2. **Assignment Management Module**
   - Assignment creation
   - Rubric and model answer upload
   - Student assignment submission
   - NLP-based automated evaluation
   - Reference and citation checking
   - Lecturer mark and feedback endorsement
   - Student result viewing

3. **Quiz Module**
   - Lecturer notes upload
   - Automatic quiz generation
   - Question, answer, distractor, and explanation generation
   - Lecturer quiz review
   - Quiz publishing
   - Student quiz attempt and result viewing

4. **Dashboard Module**
   - Lecturer dashboard
   - Student dashboard
   - Assignment performance monitoring
   - Quiz performance monitoring
   - Feedback and progress tracking

---

## Technology Stack

### Backend
- Python
- Flask
- Flask API

### Frontend
- HTML
- CSS
- JavaScript

### Database
- PostgreSQL

### AI and Machine Learning
- Natural Language Processing
- Natural Language Generation
- SentenceTransformers
- `all-MiniLM-L6-v2`
- Hugging Face Transformers
- T5 / Flan-T5
- TF-IDF
- Cosine Similarity

### Python Libraries
- `pypdf`
- `openpyxl`
- `scikit-learn`
- `sentence-transformers`
- `transformers`
- `pandas`
- `numpy`

---

## AI Methods Used

### NLP-Based Assignment Evaluation

The assignment evaluation process includes:

1. Extracting text from student PDF reports.
2. Cleaning unnecessary text such as links, symbols, headers, footers, and repeated noise.
3. Parsing the uploaded rubric file.
4. Detecting relevant report sections.
5. Splitting selected sections into smaller chunks.
6. Retrieving the most relevant chunks for each rubric criterion.
7. Calculating semantic similarity using Sentence-BERT.
8. Falling back to TF-IDF cosine similarity if the transformer model is unavailable.
9. Evaluating concept coverage, breadth, evidence, and reference quality.
10. Generating rubric-based scores and feedback.

### NLG-Based Quiz Generation

The quiz generation process includes:

1. Cleaning uploaded lecturer notes.
2. Extracting important concepts using rule-based seed concepts and TF-IDF.
3. Retrieving relevant passages from the notes.
4. Generating multiple-choice questions using T5-based models.
5. Generating distractors and explanations.
6. Applying fallback rules if generated outputs are invalid.
7. Allowing lecturers to review and publish quizzes.

---

## Installation

1. Clone the Repository
git clone https://github.com/ffffaat/fyp.git
cd fyp
2. Create a Virtual Environment
python -m venv venv

Activate the virtual environment.

For Windows:

venv\Scripts\activate

For macOS/Linux:

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Set Up PostgreSQL Database

Create a PostgreSQL database and update the database configuration in the project.

Example environment configuration:

DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
DB_PORT=5432
5. Run the Application
python app.py

Then open the system in your browser:

http://localhost:5000
