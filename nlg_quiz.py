import os
import re
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
# MODEL PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

QG_MODEL_DIR = os.path.join(BASE_DIR, "app", "ml_models", "qg_final")
DG_MODEL_DIR = os.path.join(BASE_DIR, "app", "ml_models", "dg_final")
EXPL_MODEL_DIR = os.path.join(BASE_DIR, "app", "ml_models", "expl_final")
S2V_MODEL_DIR = os.path.join(
    BASE_DIR, "app", "ml_models", "sense2vec_distractor_generation", "data", "s2v_old"
)

FALLBACK_MODEL_NAME = os.getenv("NLG_MODEL_NAME", "google/flan-t5-base")
_HF_TOKEN = os.getenv("HF_TOKEN")


# ============================================================
# MODEL WRAPPER
# ============================================================
class LocalSeq2SeqGenerator:
    def __init__(self, model_dir: str, fallback_model_name: Optional[str] = None):
        self.model_dir = model_dir
        self.fallback_model_name = fallback_model_name
        self.tokenizer = None
        self.model = None
        self.loaded_source = None
        self._load()

    def _load(self):
        kwargs = {}
        if _HF_TOKEN:
            kwargs["token"] = _HF_TOKEN

        if os.path.exists(self.model_dir):
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, **kwargs)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir, **kwargs)
            self.loaded_source = self.model_dir
        elif self.fallback_model_name:
            self.tokenizer = AutoTokenizer.from_pretrained(self.fallback_model_name, **kwargs)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.fallback_model_name, **kwargs)
            self.loaded_source = self.fallback_model_name
        else:
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

    def generate(self, prompt: str, max_new_tokens: int = 96, num_beams: int = 4) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            early_stopping=True
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()


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
    "further", "reading", "table", "contents"
}

GENERIC_BAD_CONCEPTS = {
    "data", "information", "model", "system", "process",
    "method", "approach", "framework", "level", "domain",
    "task", "design", "validation"
}

BAD_SINGLE_WORD_CONCEPTS = {
    "visual", "channel", "channels", "encoding", "attributes", "attribute",
    "marks", "mark", "effectiveness", "choice", "target", "actions",
    "using", "context", "focus", "information", "item", "items",
    "mapping", "mappings", "representation", "representations",
    "difference", "example", "examples", "type", "types", "step", "steps",
    "part", "parts", "view", "views", "detail", "details",
    "region", "regions", "question", "answer", "answers",
    "note", "notes", "figure", "figures", "table", "tables"
}

MULTIWORD_BAD_PHRASES = {
    "visual channel",
    "visual channels",
    "data item",
    "data items",
    "choice target",
}

ROMAN_NUMERAL_PATTERN = re.compile(r"^\(?[ivxlcdm]+\)?$", re.I)
NOISE_PATTERN = re.compile(r"^(sep|sep>|<sep>|>sep|ii|iii|iv|vi|vii|viii|ix|x)$", re.I)


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
    text = re.sub(r"</?sep>", " ", text, flags=re.I)
    text = re.sub(r"\bsep\b", " ", text, flags=re.I)
    text = re.sub(r">sep|sep>|<sep>|</sep>", " ", text, flags=re.I)
    text = re.sub(r"^\([a-z0-9ivxlcdm]+\)\s*", "", text, flags=re.I)
    text = re.sub(r"[<>|_/\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -:;,.")
    return text


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

    raw = _clean_option(term)
    low = raw.lower()
    tokens = low.split()

    if not low:
        return True
    if len(low) < 4:
        return True
    if low.isdigit():
        return True
    if ROMAN_NUMERAL_PATTERN.match(low):
        return True
    if NOISE_PATTERN.match(low):
        return True
    if _looks_like_person_name(raw):
        return True
    if any(tok in BLACKLIST_NAME_FRAGMENTS for tok in tokens):
        return True
    if any(tok in BLACKLIST_META_TERMS for tok in tokens):
        return True
    if low in GENERIC_BAD_CONCEPTS:
        return True
    if low in MULTIWORD_BAD_PHRASES:
        return True
    if len(tokens) == 1 and low in BAD_SINGLE_WORD_CONCEPTS:
        return True
    if re.match(r"^\([a-z0-9ivxlcdm]+\)$", low, re.I):
        return True
    if "sep" in low:
        return True

    alpha_count = sum(ch.isalpha() for ch in raw)
    if alpha_count < 3:
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
    low = text.lower()
    words = text.split()

    if len(words) < 1 or len(words) > 6:
        return False
    if len(text) < 3:
        return False
    if any(meta in low for meta in BLACKLIST_META_TERMS):
        return False
    if "sep" in low:
        return False
    if ROMAN_NUMERAL_PATTERN.match(low):
        return False
    if NOISE_PATTERN.match(low):
        return False
    if re.match(r"^\([a-z0-9ivxlcdm]+\)$", low, re.I):
        return False
    if _is_bad_concept(text):
        return False

    alpha_count = sum(ch.isalpha() for ch in text)
    if alpha_count < 3:
        return False

    return True


def is_valid_explanation(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 12 or len(words) > 100:
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


def deduplicate_concepts_by_phrase(concepts: List[str]) -> List[str]:
    # prefer longer multi-word phrases over shorter subfragments
    concepts = [_clean_option(c) for c in concepts if _clean_option(c)]
    concepts = sorted(concepts, key=lambda x: (-len(x.split()), -len(x), x.lower()))

    kept = []
    for concept in concepts:
        low = concept.lower()
        if any(low == k.lower() for k in kept):
            continue
        if any(low in k.lower().split() for k in kept):
            continue
        if any(low == k.lower() for k in kept):
            continue
        kept.append(concept)

    # preserve readable order after filtering
    return kept


# ============================================================
# CONCEPT EXTRACTION
# ============================================================
def extract_key_entities(text: str, top_k: int = 20) -> List[str]:
    cleaned = clean_text(text)
    if len(cleaned) < 150:
        return []

    concepts = []
    seen = set()

    idiom_matches = re.findall(
        r"(?:Idiom|Idioms)\s*:\s*([A-Za-z][A-Za-z ,\-\(\)]+)",
        cleaned,
        flags=re.I
    )

    for match in idiom_matches:
        parts = re.split(r",| and ", match)
        for part in parts:
            concept = _clean_option(part.title())
            key = concept.lower()
            if concept and key not in seen and not _is_bad_concept(concept):
                concepts.append(concept)
                seen.add(key)

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
        concept = _clean_option(term.strip().title())
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
    concepts = deduplicate_concepts_by_phrase(concepts)

    valid = []
    seen = set()

    for concept in concepts:
        concept = _clean_option(concept)
        key = concept.lower()

        if not concept or _is_bad_concept(concept):
            continue
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
            cleaned = _clean_option(normalized_phrase.capitalize())
            if cleaned.lower() != answer.lower() and is_valid_option(cleaned):
                distractors.append(cleaned)

        return list(OrderedDict.fromkeys(distractors))


# ============================================================
# MAIN SERVICE
# ============================================================
class QuizGenerationService:
    def __init__(self):
        self.qg_model = LocalSeq2SeqGenerator(QG_MODEL_DIR, fallback_model_name=FALLBACK_MODEL_NAME)
        self.dg_model = LocalSeq2SeqGenerator(DG_MODEL_DIR, fallback_model_name=FALLBACK_MODEL_NAME)
        self.expl_model = LocalSeq2SeqGenerator(EXPL_MODEL_DIR, fallback_model_name=FALLBACK_MODEL_NAME)
        self.s2v_generator = Sense2VecDistractorGeneration(model_path=S2V_MODEL_DIR)

    def _is_question_aligned(self, question: str, concept: str) -> bool:
        q = question.strip().lower()
        c = concept.strip().lower()

        bad_starts = [
            "what is ",
            "why is ",
            "how is ",
            "explain ",
            "describe ",
            "what does ",
            "what are ",
            "how does ",
            "what is the difference",
            "why does ",
        ]
        if any(q.startswith(x) for x in bad_starts):
            return False

        if c in q:
            return False

        good_starts = [
            "which concept",
            "which term",
            "which method",
            "which technique",
            "which visualization",
            "which principle",
            "which approach",
            "which structure",
            "which option",
        ]
        return any(q.startswith(x) for x in good_starts)

    def _is_generic_fallback_question(self, question: str) -> bool:
        q = question.strip().lower()
        return q in {
            "which concept best describes the idea presented in the notes?",
            "which concept best describes the idea in the notes?",
            "which concept best describes the notes?",
        }

    def _fallback_question(self, concept: str, passage: str) -> str:
        low = concept.lower()

        if "visual encoding" in low or low == "encoding":
            return "Which concept refers to mapping data attributes to visual channels?"
        if "tree" in low and "treemap" not in low:
            return "Which structure is commonly used to represent hierarchical relationships?"
        if "treemap" in low:
            return "Which visualization uses nested rectangles to show hierarchical proportions?"
        if "abstraction" in low:
            return "Which concept reduces detail to focus on essential features of the data?"
        if "filter" in low:
            return "Which interaction technique lets users focus on a subset of data?"
        if "brushing" in low:
            return "Which interaction technique highlights selected items across views?"
        if "position" in low:
            return "Which visual channel is typically most accurate for quantitative comparison?"
        if "marks" in low:
            return "Which term refers to basic graphical elements such as points, lines, and areas?"

        return "Which concept best describes the idea presented in the notes?"

    def generate_question(self, concept: str, passage: str) -> str:
        prompt = f"""
You are generating a university multiple-choice question.

Rules:
- The correct answer must be exactly: {concept}
- The answer will be a concept term, not a sentence
- Write a concept-identification question
- Good styles:
  - Which concept refers to ...
  - Which term best describes ...
  - Which method is used for ...
  - Which visualization is suitable for ...
- Do NOT write:
  - What is {concept}
  - Why is {concept} useful
  - Explain {concept}
  - What is the purpose of ...
- Maximum 18 words
- End with a question mark

Context:
{passage}

Question:
"""
        text = self.qg_model.generate(prompt, max_new_tokens=64, num_beams=4)
        text = _clean_option(text)

        if not text.endswith("?"):
            text += "?"

        if self._is_question_aligned(text, concept) and is_valid_question(text):
            return text

        return self._fallback_question(concept, passage)

    def generate_distractors(
        self,
        concept: str,
        question: str,
        passage: str,
        concept_pool: List[str],
        desired_count: int = 3
    ) -> List[str]:
        prompt = f"generate distractors: {concept} <sep> {question} <sep> {passage}"
        text = self.dg_model.generate(prompt, max_new_tokens=96, num_beams=4)

        distractors = []

        parts = re.split(r"<sep>|</sep>|>sep|sep>|[\n\r]+", text, flags=re.I)
        candidates = [_clean_option(x) for x in parts if _clean_option(x)]

        for line in candidates:
            low = line.lower()

            if not is_valid_option(line):
                continue
            if _is_bad_concept(line):
                continue
            if low == concept.lower():
                continue
            if _too_similar(line, concept):
                continue
            if low in [d.lower() for d in distractors]:
                continue

            distractors.append(line)
            if len(distractors) >= desired_count:
                break

        # fallback to good concept pool
        if len(distractors) < desired_count:
            for other in concept_pool:
                other = _clean_option(other)
                if not is_valid_option(other):
                    continue
                if _is_bad_concept(other):
                    continue
                if other.lower() == concept.lower():
                    continue
                if _too_similar(other, concept):
                    continue
                if other.lower() in [d.lower() for d in distractors]:
                    continue

                distractors.append(other)
                if len(distractors) >= desired_count:
                    break

        # fallback to sense2vec
        if len(distractors) < desired_count:
            s2v_items = self.s2v_generator.generate(answer=concept, desired_count=6)
            for item in s2v_items:
                item = _clean_option(item)
                if not is_valid_option(item):
                    continue
                if item.lower() == concept.lower():
                    continue
                if item.lower() in [d.lower() for d in distractors]:
                    continue

                distractors.append(item)
                if len(distractors) >= desired_count:
                    break

        # hard fallback
        fallback_pool = [
            "Treemap",
            "Scatter Plot",
            "Bar Chart",
            "Line Chart",
            "Histogram",
            "Box Plot",
            "Node Link Diagram",
            "Adjacency Matrix",
        ]

        for item in fallback_pool:
            if len(distractors) >= desired_count:
                break
            if item.lower() == concept.lower():
                continue
            if item.lower() in [d.lower() for d in distractors]:
                continue
            if _too_similar(item, concept):
                continue

            distractors.append(item)

        return distractors[:desired_count]

    def generate_explanation(
        self,
        concept: str,
        question: str,
        distractors: List[str],
        passage: str
    ) -> str:
        while len(distractors) < 3:
            distractors.append("Other concept")

        prompt = (
            f"generate explanation: {concept} <sep> {question} <sep> "
            f"{distractors[0]} <sep> {distractors[1]} <sep> {distractors[2]} <sep> {passage}"
        )

        text = self.expl_model.generate(prompt, max_new_tokens=160, num_beams=4)
        text = " ".join(text.split())

        if is_valid_explanation(text):
            return text

        return (
            f"The correct answer is {concept} because it best matches the concept described in the notes. "
            f"The other options are less suitable because they refer to different visualization methods or ideas."
        )

    def generate_question_bundle(
        self,
        concept: str,
        full_text: str,
        concept_pool: Optional[List[str]] = None
    ) -> Optional[Dict[str, object]]:
        concept = _clean_option(concept)

        if not concept or _is_bad_concept(concept):
            return None

        full_text = clean_text(full_text)
        concept_pool = concept_pool or [concept]
        concept_pool = filter_valid_concepts(concept_pool, limit=50)

        passage = get_relevant_passage(full_text, concept)
        question = self.generate_question(concept, passage)
        correct = concept

        if not self._is_question_aligned(question, correct):
            return None
        if self._is_generic_fallback_question(question):
            return None
        if _is_bad_concept(correct):
            return None

        distractors = self.generate_distractors(
            concept=concept,
            question=question,
            passage=passage,
            concept_pool=concept_pool,
            desired_count=3
        )

        if len(distractors) < 3:
            return None

        options = [correct] + distractors
        if len(set(_normalize_text(x) for x in options)) < 4:
            return None

        explanation = self.generate_explanation(
            concept=concept,
            question=question,
            distractors=distractors,
            passage=passage
        )

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
    return service.generate_question_bundle(
        concept=concept,
        full_text=context,
        concept_pool=[concept]
    )