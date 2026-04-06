import os
import re
import json
import random
from collections import OrderedDict
from typing import List, Dict, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

try:
    from sense2vec import Sense2Vec
except ImportError:
    Sense2Vec = None


# ============================================================
# MODEL LOADING
# ============================================================
_MODEL_NAME = os.getenv("NLG_MODEL_NAME", "google/flan-t5-base")
_HF_TOKEN = os.getenv("HF_TOKEN")

_tokenizer = None
_model = None


def _lazy_load_model():
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    kwargs = {}
    if _HF_TOKEN:
        kwargs["token"] = _HF_TOKEN

    _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, **kwargs)
    _model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME, **kwargs)


def _generate_text(prompt: str, max_new_tokens: int = 96) -> str:
    _lazy_load_model()

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=4,
        early_stopping=True
    )

    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


# ============================================================
# FILTERING / CLEANING
# ============================================================
BLACKLIST_NAME_FRAGMENTS = {
    "nur", "intan", "raihana", "ruhaiyem",
    "dr", "prof", "mr", "ms", "madam"
}

BLACKLIST_META_TERMS = {
    "email", "website", "http", "https", "www",
    "universiti", "sains", "malaysia", "school",
    "computer", "sciences", "cds505", "lecture",
    "slides", "slide", "copyright", "session",
    "semester", "page", "outline", "recap",
    "further", "reading", "table of contents"
}

GENERIC_BAD_CONCEPTS = {
    "data", "information", "model", "system", "process",
    "method", "approach", "framework", "level", "domain",
    "task", "design", "validation"
}


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.I)
    text = re.sub(r"\b\S+@\S+\b", " ", text)

    removable_patterns = [
        r"\bnur\s+intan\s+raihana\s+ruhaiyem\b",
        r"\braihana\s+ruhaiyem\b",
        r"\bnur\s+intan\b",
        r"\bruha?iyem\b",
        r"\bcds505\b",
        r"\buniversiti\s+sains\s+malaysia\b",
        r"\bschool\s+of\s+computer\s+sciences\b",
        r"\bw\s*e\s*l\s*e\s*a\s*d\b",
    ]
    for pat in removable_patterns:
        text = re.sub(pat, " ", text, flags=re.I)

    text = re.sub(r"[^A-Za-z0-9\s\-\(\)\.,:;'/]", " ", text)
    text = re.sub(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]?\b", " ", text)
    text = re.sub(r"\b\w{20,}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clean_option(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[A-Da-d][\)\.\:]\s*", "", text)
    text = re.sub(r"^\d+[\)\.\:]\s*", "", text)
    return text.strip(" -")


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _looks_like_person_name(term: str) -> bool:
    if not term:
        return True

    tokens = term.strip().split()
    low_tokens = [t.lower() for t in tokens]

    if any(tok in BLACKLIST_NAME_FRAGMENTS for tok in low_tokens):
        return True

    if 2 <= len(tokens) <= 4:
        title_case_count = sum(1 for t in tokens if t[:1].isupper() and t[1:].islower())
        if title_case_count >= max(2, len(tokens) - 1):
            return True

    return False


def _is_bad_concept(term: str) -> bool:
    if not term:
        return True

    low = term.strip().lower()
    tokens = low.split()

    if len(low) < 4:
        return True
    if low.isdigit():
        return True
    if _looks_like_person_name(term):
        return True
    if any(tok in BLACKLIST_NAME_FRAGMENTS for tok in tokens):
        return True
    if any(tok in BLACKLIST_META_TERMS for tok in tokens):
        return True
    if low in GENERIC_BAD_CONCEPTS:
        return True

    return False


def is_valid_question(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 5 or len(words) > 25:
        return False
    if any(meta in text.lower() for meta in BLACKLIST_META_TERMS):
        return False
    return text.endswith("?")


def is_valid_option(text: str) -> bool:
    if not text:
        return False
    text = _clean_option(text)
    words = text.split()
    if len(words) < 1 or len(words) > 12:
        return False
    if any(meta in text.lower() for meta in BLACKLIST_META_TERMS):
        return False
    return True


def is_valid_explanation(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 12 or len(words) > 80:
        return False
    if any(meta in text.lower() for meta in BLACKLIST_META_TERMS):
        return False
    return True


def _too_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    sa = set(_normalize_text(a).split())
    sb = set(_normalize_text(b).split())
    if not sa or not sb:
        return False
    overlap = len(sa & sb) / max(len(sa), len(sb))
    return overlap >= threshold


# ============================================================
# CONCEPT EXTRACTION
# ============================================================
def extract_key_entities(text: str, top_k: int = 12) -> List[str]:
    cleaned = clean_text(text)
    if len(cleaned) < 150:
        return []

    idiom_matches = re.findall(
        r"(?:Idiom|Idioms)\s*:\s*([A-Za-z][A-Za-z ,\-\(\)]+)",
        cleaned,
        flags=re.I
    )

    concepts = []
    seen = set()

    for match in idiom_matches:
        parts = re.split(r",| and ", match)
        for part in parts:
            concept = part.strip(" -").title()
            if concept and not _is_bad_concept(concept):
                key = concept.lower()
                if key not in seen:
                    concepts.append(concept)
                    seen.add(key)

    if len(concepts) >= top_k:
        return concepts[:top_k]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        max_features=4000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-']+\b",
    )

    X = vectorizer.fit_transform([cleaned])
    terms = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    ranked = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)

    for term, score in ranked:
        if score <= 0:
            continue
        concept = term.strip().title()
        key = concept.lower()
        if key in seen:
            continue
        if _is_bad_concept(concept):
            continue
        concepts.append(concept)
        seen.add(key)
        if len(concepts) >= top_k:
            break

    return concepts


def filter_valid_concepts(concepts: List[str], limit: int = 8) -> List[str]:
    valid = []
    seen = set()

    for concept in concepts:
        concept = (concept or "").strip()
        if not concept or _is_bad_concept(concept):
            continue
        key = concept.lower()
        if key in seen:
            continue
        valid.append(concept)
        seen.add(key)
        if len(valid) >= limit:
            break

    return valid


# ============================================================
# PASSAGE RETRIEVAL
# ============================================================
def _split_into_passages(text: str, window: int = 3, step: int = 2) -> List[str]:
    text = clean_text(text)
    sentences = re.split(r"(?<=[\.\?!])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    passages = []
    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i:i + window]).strip()
        if len(chunk) >= 80:
            passages.append(chunk)

    return passages[:300]


def get_relevant_passage(text: str, query: str, top_k: int = 2) -> str:
    passages = _split_into_passages(text)
    if not passages:
        return clean_text(text)[:1500]

    docs = [query] + passages
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(docs)
    sims = cosine_similarity(X[0:1], X[1:])[0]
    ranked = sims.argsort()[::-1][:top_k]

    selected = [passages[i] for i in ranked if sims[i] > 0]
    return " ".join(selected)[:1500] if selected else clean_text(text)[:1500]


# ============================================================
# OPTIONAL SENSE2VEC
# ============================================================
class Sense2VecDistractorGeneration:
    def __init__(self, model_path: str):
        self.s2v = None

        if Sense2Vec is None:
            return

        try:
            self.s2v = Sense2Vec().from_disk(model_path)
        except Exception:
            self.s2v = None

    def generate(self, answer: str, desired_count: int) -> List[str]:
        if self.s2v is None:
            return []

        distractors = []
        answer_norm = answer.lower().replace(" ", "_")
        sense = self.s2v.get_best_sense(answer_norm)

        if not sense:
            return []

        most_similar = self.s2v.most_similar(sense, n=desired_count)

        for phrase in most_similar:
            normalized_phrase = phrase[0].split("|")[0].replace("_", " ").lower()
            if normalized_phrase != answer_norm:
                distractors.append(normalized_phrase.capitalize())

        return list(OrderedDict.fromkeys(distractors))


# ============================================================
# GENERATION LOGIC
# ============================================================
def generate_question(concept: str, passage: str) -> str:
    prompt = f"""
You are creating a university multiple-choice quiz for a Data Visualization course.

Write one clear question based on the notes.

Rules:
- The correct answer must be: {concept}
- Ask about meaning, purpose, or usage
- Maximum 18 words
- End with a question mark
- No lecturer names, no course codes, no metadata

Notes:
{passage}

Question:
"""
    text = _generate_text(prompt, max_new_tokens=48)
    text = _clean_option(text)

    if not text.endswith("?"):
        text += "?"

    if is_valid_question(text):
        return text

    return f"Which option best describes {concept}?"


def generate_distractors(concept: str, passage: str, concept_pool: List[str], desired_count: int = 3) -> List[str]:
    distractors = []

    for other in concept_pool:
        if other.lower() == concept.lower():
            continue
        if _too_similar(other, concept):
            continue
        if other.lower() in [d.lower() for d in distractors]:
            continue
        distractors.append(other)
        if len(distractors) >= desired_count:
            return distractors

    prompt = f"""
You are creating wrong answer options for a university multiple-choice quiz.

Task:
Generate {desired_count} plausible distractors for the correct answer.

Rules:
- correct answer: {concept}
- distractors must be related to data visualization
- each distractor should be short, 1 to 4 words
- do not repeat the correct answer
- return one distractor per line
- no numbering

Notes:
{passage}
"""
    text = _generate_text(prompt, max_new_tokens=64)

    for line in text.splitlines():
        line = _clean_option(line)
        if not is_valid_option(line):
            continue
        if line.lower() == concept.lower():
            continue
        if _too_similar(line, concept):
            continue
        if line.lower() in [d.lower() for d in distractors]:
            continue
        distractors.append(line)
        if len(distractors) >= desired_count:
            break

    fallback_pool = [
        "Scatter Plot",
        "Heat Map",
        "Bar Chart",
        "Line Graph",
        "Treemap",
        "Node Link Diagram",
        "Table View",
    ]

    for item in fallback_pool:
        if len(distractors) >= desired_count:
            break
        if item.lower() == concept.lower():
            continue
        if item.lower() in [d.lower() for d in distractors]:
            continue
        distractors.append(item)

    return distractors[:desired_count]


def generate_explanation(concept: str, question: str, correct: str, distractors: List[str], passage: str) -> str:
    prompt = f"""
You are giving quiz feedback to a student.

Write a short explanation.

Rules:
- 2 to 3 sentences
- explain why the correct answer is correct
- briefly explain why the other options are less suitable
- simple, meaningful academic English
- no names, no websites, no course codes

Question: {question}
Correct answer: {correct}
Other options: {", ".join(distractors)}

Notes:
{passage}
"""
    text = _generate_text(prompt, max_new_tokens=100)
    text = " ".join(text.split())

    if is_valid_explanation(text):
        return text

    return (
        f"The correct answer is {correct} because it best matches the concept described in the notes. "
        f"The other options are less suitable because they refer to different visualization methods or ideas."
    )


# ============================================================
# SERVICE
# ============================================================
class QuizGenerationService:
    def __init__(self):
        self.s2v_generator = Sense2VecDistractorGeneration(
            model_path="app/ml_models/sense2vec_distractor_generation/data/s2v_old"
        )

    def generate_question_bundle(self, concept: str, full_text: str, concept_pool: Optional[List[str]] = None) -> Optional[Dict[str, object]]:
        if not concept or _is_bad_concept(concept):
            return None

        full_text = clean_text(full_text)
        concept_pool = concept_pool or [concept]

        passage = get_relevant_passage(full_text, concept)
        question = generate_question(concept, passage)
        correct = concept

        distractors = generate_distractors(
            concept=concept,
            passage=passage,
            concept_pool=concept_pool,
            desired_count=3
        )

        if len(distractors) < 3:
            s2v_items = self.s2v_generator.generate(answer=concept, desired_count=6)
            for item in s2v_items:
                if len(distractors) >= 3:
                    break
                if item.lower() == concept.lower():
                    continue
                if item.lower() in [d.lower() for d in distractors]:
                    continue
                distractors.append(item)

        if len(distractors) < 3:
            return None

        options = [correct] + distractors
        if len(set(_normalize_text(x) for x in options)) < 4:
            return None

        explanation = generate_explanation(concept, question, correct, distractors, passage)

        return {
            "question": question[:600],
            "correct": correct[:255],
            "distractors": [d[:255] for d in distractors[:3]],
            "explanation": explanation[:1200],
            "source_snippet": passage[:1000]
        }


# ============================================================
# BACKWARD-COMPATIBLE WRAPPER
# ============================================================
def generate_full_quiz_data(context: str, concept: str):
    service = QuizGenerationService()
    return service.generate_question_bundle(concept=concept, full_text=context, concept_pool=[concept])