import os
import re
from typing import List, Dict, Tuple

from pypdf import PdfReader
from openpyxl import load_workbook
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
try:
    from sentence_transformers import SentenceTransformer
    _sbert = SentenceTransformer("all-MiniLM-L6-v2")
except Exception:
    _sbert = None


# -------------------------------------------------
# BASIC TEXT HELPERS
# -------------------------------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b\S+@\S+\b", " ", text)
    text = re.sub(r"[•●■◆▶►▪]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_for_matching(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_front_matter_noise(text: str) -> str:
    """
    Remove common cover-page / table-of-contents noise.
    Try to keep content from the first real section onward.
    """
    raw = text

    raw = re.sub(
        r"school of computer sciences.*?(?=1\.0\s+|\b1\s*\.\s*|\bintroduction\b)",
        " ",
        raw,
        flags=re.I | re.S
    )
    raw = re.sub(
        r"academic session.*?(?=1\.0\s+|\b1\s*\.\s*|\bintroduction\b)",
        " ",
        raw,
        flags=re.I | re.S
    )
    raw = re.sub(
        r"table of contents.*?(?=1\.0\s+|\b1\s*\.\s*|\bintroduction\b)",
        " ",
        raw,
        flags=re.I | re.S
    )

    m = re.search(
        r"(1\.0\s+[A-Za-z][A-Za-z\s&\-]+|1\s*\.\s*[A-Za-z][A-Za-z\s&\-]+|\bIntroduction\b)",
        raw,
        flags=re.I
    )
    if m:
        raw = raw[m.start():]

    return clean_text(raw)


def is_toc_like_chunk(chunk: str) -> bool:
    """
    Detect table-of-contents / heading-list chunks.
    """
    low = chunk.lower()

    if "table of contents" in low:
        return True

    dotted = len(re.findall(r"\.{4,}", chunk))
    many_numbers = len(re.findall(r"\b\d+\b", chunk))
    heading_like = len(re.findall(r"\b\d+(\.\d+)*\s+[A-Za-z]", chunk))
    words = chunk.split()

    if len(words) > 0:
        heading_ratio = heading_like / max(1, len(words) / 20)
    else:
        heading_ratio = 0

    if dotted >= 2:
        return True
    if heading_like >= 5 and heading_ratio > 1.0:
        return True
    if many_numbers >= 12 and len(words) < 180:
        return True

    return False


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path, strict=False)
    texts = []

    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            texts.append(t)

    full_text = clean_text(" ".join(texts))
    return full_text


# -------------------------------------------------
# RUBRIC PARSING
# -------------------------------------------------
def parse_rubric_xlsx(rubric_path: str) -> List[Dict]:
    if not rubric_path.lower().endswith((".xlsx", ".xlsm", ".xltx", ".xltm")):
        raise ValueError("Rubric file must be Excel (.xlsx/.xlsm/.xltx/.xltm), not PDF.")

    wb = load_workbook(rubric_path, data_only=True)
    ws = wb.active

    criteria = []

    for row in range(1, ws.max_row + 1):
        criterion = ws[f"B{row}"].value
        weight = ws[f"G{row}"].value
        max_scale = ws[f"H{row}"].value

        if criterion and isinstance(criterion, str):
            crit = criterion.strip()

            if crit.lower() in {
                "criteria / scale",
                "criteria",
                "scale",
                "individual",
                "criterion"
            }:
                continue

            if weight is not None and max_scale is not None:
                try:
                    weight_val = float(weight)
                    max_scale_val = float(max_scale)
                except Exception:
                    continue

                criteria.append({
                    "criterion_name": crit,
                    "weight": weight_val,
                    "max_scale": max_scale_val,
                })

    if not criteria:
        raise ValueError("No rubric criteria could be parsed from the Excel file.")

    return criteria


# -------------------------------------------------
# CRITERION QUERY EXPANSION
# -------------------------------------------------
CRITERION_QUERY_HINTS = {
    "Abstract, Intro & Background": [
        "abstract", "introduction", "background", "big picture",
        "real world problem", "problem context", "dataset description",
        "dataset preparation", "target audience", "motivation"
    ],
    "Data & Task Abstraction": [
        "data abstraction", "dataset types", "data items", "data attribute types",
        "derived attributes", "task abstraction", "task type", "what why how"
    ],
    "Methods & Tools for Data Storytelling": [
        "methods", "tools", "data storytelling", "tableau", "interactive dashboard",
        "story", "visualisation method", "implementation"
    ],
    "Methods & Tools": [
        "methods", "tools", "software", "implementation", "tableau",
        "visual encoding", "marks and channels", "interaction design"
    ],
    "Visual Analytics": [
        "visual analytics", "analysis", "design justification", "final visualisations",
        "dashboard", "charts", "interaction", "encoding"
    ],
    "Related Work": [
        "related work", "references", "literature", "previous studies", "citation"
    ],
    "Discussion and Conclusion": [
        "discussion", "conclusion", "summary", "finding", "overall", "in summary"
    ],
    "References": [
        "references", "bibliography", "citation"
    ],
    "Vis Tool Project File & Datasets": [
        "dataset", "data source", "preparation", "visualisation tool", "tableau", "dashboard"
    ],
}


def build_query_for_criterion(criterion_name: str) -> str:
    base = criterion_name.strip()
    hints = CRITERION_QUERY_HINTS.get(criterion_name, [])
    query = " ".join([base] + hints)
    return clean_text(query)


# -------------------------------------------------
# CHUNKING
# -------------------------------------------------
def split_into_paragraph_chunks(text: str, min_words: int = 40, max_words: int = 220) -> List[str]:
    """
    Split document into paragraph-like chunks while filtering TOC/front matter.
    """
    text = remove_front_matter_noise(text)
    text = text.replace("\r", "\n")

    raw_parts = re.split(r"\n\s*\n+", text)

    cleaned_parts = []
    for part in raw_parts:
        p = clean_text(part)
        if not p:
            continue
        if is_toc_like_chunk(p):
            continue
        cleaned_parts.append(p)

    if len(cleaned_parts) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
        cleaned_parts = []
        current = []

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            current.append(sent)
            if len(" ".join(current).split()) >= max_words:
                chunk = " ".join(current).strip()
                if not is_toc_like_chunk(chunk):
                    cleaned_parts.append(chunk)
                current = []

        if current:
            chunk = " ".join(current).strip()
            if not is_toc_like_chunk(chunk):
                cleaned_parts.append(chunk)

    merged_chunks = []
    buffer = ""

    for part in cleaned_parts:
        candidate = (buffer + " " + part).strip() if buffer else part
        if len(candidate.split()) < min_words:
            buffer = candidate
        else:
            merged_chunks.append(candidate)
            buffer = ""

    if buffer:
        if merged_chunks:
            merged_chunks[-1] = (merged_chunks[-1] + " " + buffer).strip()
        else:
            merged_chunks.append(buffer)

    final_chunks = []
    for chunk in merged_chunks:
        words = chunk.split()
        if len(words) <= max_words:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(words), max_words):
                piece = " ".join(words[i:i + max_words]).strip()
                if piece and not is_toc_like_chunk(piece):
                    final_chunks.append(piece)

    return final_chunks


# -------------------------------------------------
# SEMANTIC REPRESENTATION
# -------------------------------------------------
def compute_semantic_similarity(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0

    if _sbert is not None:
        embeddings = _sbert.encode([text_a[:3000], text_b[:3000]])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return max(0.0, float(score))

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=1000)
    X = vectorizer.fit_transform([text_a, text_b])
    score = cosine_similarity(X[0:1], X[1:2])[0][0]
    return max(0.0, float(score))


def heading_bonus(query: str, chunk: str) -> float:
    q = normalize_text_for_matching(query)
    c = normalize_text_for_matching(chunk[:500])

    bonus = 0.0

    heading_terms = q.split()
    matched = sum(1 for term in heading_terms if len(term) > 3 and term in c)

    if matched >= 1:
        bonus += 0.03 * matched

    if any(term in c for term in [
        "introduction", "background", "big picture", "target audience",
        "data abstraction", "task abstraction", "design justification",
        "validation", "references", "conclusion", "discussion"
    ]):
        bonus += 0.05

    return min(bonus, 0.15)


def chunk_relevance_scores(query: str, chunks: List[str]) -> List[Tuple[str, float]]:
    scored = []

    for chunk in chunks:
        base_score = compute_semantic_similarity(query, chunk)
        bonus = heading_bonus(query, chunk)

        penalty = 0.0
        low = chunk.lower()

        if "matric no" in low or "usm email" in low or "name matric" in low:
            penalty += 0.20
        if "table of contents" in low:
            penalty += 0.40

        final_score = max(0.0, base_score + bonus - penalty)
        scored.append((chunk, final_score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def retrieve_top_chunks(query: str, chunks: List[str], top_k: int = 2) -> List[str]:
    if not chunks:
        return []

    ranked = chunk_relevance_scores(query, chunks)
    return [chunk for chunk, _ in ranked[:top_k]]


def get_relevant_text_for_criterion(criterion_name: str, text: str, top_k: int = 2) -> Tuple[str, List[str]]:
    chunks = split_into_paragraph_chunks(text)
    query = build_query_for_criterion(criterion_name)

    selected_chunks = retrieve_top_chunks(query, chunks, top_k=top_k)

    if not selected_chunks and chunks:
        selected_chunks = chunks[:top_k]

    merged = " ".join(selected_chunks).strip()
    return merged, selected_chunks


# -------------------------------------------------
# FEATURE COMPUTATION
# -------------------------------------------------
def compute_keyword_coverage(student_text: str, model_text: str, max_keywords: int = 25) -> float:
    student_text = clean_text(student_text).lower()
    model_text = clean_text(model_text).lower()

    if not model_text:
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_keywords
        )
        X = vectorizer.fit_transform([model_text])
        keywords = vectorizer.get_feature_names_out()
    except Exception:
        return 0.0

    if len(keywords) == 0:
        return 0.0

    hits = 0
    for kw in keywords:
        if kw in student_text:
            hits += 1

    return hits / len(keywords)


def compute_structure_score(student_text: str, model_text: str) -> float:
    student_len = len(student_text.split())
    model_len = len(model_text.split())

    if model_len == 0:
        return 0.0

    ratio = student_len / model_len

    if 0.8 <= ratio <= 1.2:
        return 1.0
    elif 0.6 <= ratio < 0.8 or 1.2 < ratio <= 1.5:
        return 0.75
    elif 0.4 <= ratio < 0.6 or 1.5 < ratio <= 2.0:
        return 0.45
    else:
        return 0.2


def criterion_score(similarity: float, keyword_cov: float, structure_score: float, max_scale: float) -> float:
    blended = (0.60 * similarity) + (0.25 * keyword_cov) + (0.15 * structure_score)
    return round(blended * max_scale, 2)


def build_feedback(
    criterion_name: str,
    similarity: float,
    keyword_cov: float,
    structure_score: float,
    raw_score: float,
    max_scale: float
) -> str:
    if raw_score >= max_scale * 0.85:
        strength = "Strong"
    elif raw_score >= max_scale * 0.60:
        strength = "Moderate"
    else:
        strength = "Weak"

    return (
        f"{strength} performance for '{criterion_name}'. "
        f"Semantic alignment={similarity:.2f}, keyword coverage={keyword_cov:.2f}, "
        f"structure match={structure_score:.2f}, score={raw_score:.2f}/{max_scale:.2f}."
    )


# -------------------------------------------------
# MAIN EVALUATION PIPELINE
# -------------------------------------------------
def evaluate_submission(student_pdf_path: str, model_answer_pdf_path: str, rubric_xlsx_path: str) -> Dict:
    if not os.path.exists(student_pdf_path):
        raise FileNotFoundError(f"Student submission not found: {student_pdf_path}")

    if not os.path.exists(model_answer_pdf_path):
        raise FileNotFoundError(f"Model answer PDF not found: {model_answer_pdf_path}")

    if not os.path.exists(rubric_xlsx_path):
        raise FileNotFoundError(f"Rubric Excel not found: {rubric_xlsx_path}")

    student_text = extract_pdf_text(student_pdf_path)
    model_text = extract_pdf_text(model_answer_pdf_path)
    rubric_items = parse_rubric_xlsx(rubric_xlsx_path)

    if not student_text.strip():
        raise ValueError("Student PDF text extraction failed or returned empty text.")

    if not model_text.strip():
        raise ValueError("Model answer PDF text extraction failed or returned empty text.")

    results = []
    total_weight = sum(item["weight"] for item in rubric_items) or 1.0
    total_weighted_score = 0.0

    for item in rubric_items:
        criterion_name = item["criterion_name"]
        weight = item["weight"]
        max_scale = item["max_scale"]

        model_relevant_text, model_chunks = get_relevant_text_for_criterion(
            criterion_name=criterion_name,
            text=model_text,
            top_k=2
        )

        student_relevant_text, student_chunks = get_relevant_text_for_criterion(
            criterion_name=criterion_name,
            text=student_text,
            top_k=2
        )

        similarity = compute_semantic_similarity(student_relevant_text, model_relevant_text)
        keyword_cov = compute_keyword_coverage(student_relevant_text, model_relevant_text)
        structure = compute_structure_score(student_relevant_text, model_relevant_text)

        raw_score = criterion_score(similarity, keyword_cov, structure, max_scale)
        weighted_score = round((raw_score / max_scale) * weight, 2) if max_scale else 0.0

        feedback = build_feedback(
            criterion_name=criterion_name,
            similarity=similarity,
            keyword_cov=keyword_cov,
            structure_score=structure,
            raw_score=raw_score,
            max_scale=max_scale
        )

        total_weighted_score += weighted_score

        results.append({
            "criterion_name": criterion_name,
            "section_name": "Retrieved semantic chunks",
            "weight": weight,
            "max_scale": max_scale,
            "semantic_similarity": round(similarity, 4),
            "keyword_coverage": round(keyword_cov, 4),
            "structure_score": round(structure, 4),
            "raw_score": raw_score,
            "weighted_score": weighted_score,
            "feedback": feedback,
            "model_chunks": model_chunks,
            "student_chunks": student_chunks,
        })

    final_score = round((total_weighted_score / total_weight) * 100, 2)

    return {
        "final_score": final_score,
        "criteria_results": results,
        "student_text": student_text,
        "model_text": model_text,
    }