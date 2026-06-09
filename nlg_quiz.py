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
# Loads a local seq2seq model or fallback Hugging Face model for text generation.
class LocalSeq2SeqGenerator:
    # Initializes model paths and loads tokenizer/model.
    def __init__(self, model_dir: str, fallback_model_name: Optional[str] = None):
        self.model_dir = model_dir
        self.fallback_model_name = fallback_model_name
        self.tokenizer = None
        self.model = None
        self.loaded_source = None
        self._load()

    # Loads the model from local directory or fallback model name.
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

        try:
            if hasattr(self.model, "config"):
                self.model.config.tie_word_embeddings = False
            if hasattr(self.model, "generation_config"):
                self.model.generation_config.max_length = None
        except Exception:
            pass

    # Generates text output from a given prompt.
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
    "note", "notes", "figure", "figures", "table", "tables",
    "completely", "single", "dynamic", "global", "local",
    "radial", "multiple", "focus", "context", "distortion",
    "perspective", "superimpose", "aggregation", "interaction",
    "filtering"
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

# Cleans raw extracted PDF text by removing noise and extra spacing.
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

# Cleans generated answer options or concepts.
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

# Normalizes text for comparison and duplicate checking.
def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Detects whether a term looks like a person name.
def _looks_like_person_name(term: str) -> bool:
    if not term:
        return True

    tokens = term.strip().split()
    low_tokens = [t.lower() for t in tokens]

    if any(tok in BLACKLIST_NAME_FRAGMENTS for tok in low_tokens):
        return True

    if "," in term and len(tokens) <= 5:
        return True

    if re.search(r"\b[A-Z]\.\s*[A-Z]?\.", term):
        return True

    return False

# Checks whether a concept is unsuitable for quiz generation.
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

# Validates whether generated text is a proper question.
def is_valid_question(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 5 or len(words) > 25:
        return False
    if any(meta in text.lower() for meta in BLACKLIST_META_TERMS):
        return False
    return text.endswith("?")

# Validates whether a generated answer or distractor option is acceptable.
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

# Checks whether the generated explanation is usable.
def is_valid_explanation(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if len(words) < 12 or len(words) > 100:
        return False
    if any(meta in text.lower() for meta in BLACKLIST_META_TERMS):
        return False
    return True

# Checks whether two options are too similar.
def _too_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    sa = set(_normalize_text(a).split())
    sb = set(_normalize_text(b).split())
    if not sa or not sb:
        return False
    overlap = len(sa & sb) / max(len(sa), len(sb))
    return overlap >= threshold

# Removes duplicate concepts while keeping the best phrase form.
def deduplicate_concepts_by_phrase(concepts: List[str]) -> List[str]:
    concepts = [_clean_option(c) for c in concepts if _clean_option(c)]
    concepts = sorted(concepts, key=lambda x: (-len(x.split()), -len(x), x.lower()))

    kept = []
    for concept in concepts:
        low = concept.lower()
        if any(low == k.lower() for k in kept):
            continue
        kept.append(concept)
    return kept

# ============================================================
# CONCEPT EXTRACTION
# ============================================================
# Extracts possible key concepts from text using patterns and TF-IDF.
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

# Detects known lecture concepts from the uploaded notes.
def get_seed_concepts(text: str) -> List[str]:
    normalized = text.lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"\s+", " ", normalized)

    seed_rules = [
        ("Data Abstraction", ["data abstraction"]),
        ("Task Abstraction", ["task abstraction"]),
        ("Items And Attributes", ["items and attributes", "items attributes"]),
        ("Dataset Types", ["dataset types"]),
        ("Data Types", ["data types"]),
        ("Attribute Types", ["attribute types"]),
        ("Categorical Attribute", ["categorical attribute", "nominal attribute"]),
        ("Ordinal Attribute", ["ordinal attribute", "ordered attribute"]),
        ("Quantitative Attribute", ["quantitative attribute", "numerical attribute"]),
        ("Derived Attributes", ["derived attributes", "derived attribute"]),
        ("Analysis Framework", ["analysis framework", "what why how"]),
        ("Nested Model", ["nested model"]),
        ("Validation Approaches", ["validation approaches", "validation approach"]),
        ("Tables", ["tables", "tabular data"]),
        ("Networks", ["networks", "network data", "node link"]),
        ("Trees", ["trees", "tree data", "hierarchical data"]),
        ("Fields", ["fields", "field data"]),
        ("Spatial Fields", ["spatial fields", "spatial field"]),
        ("Geometry", ["geometry", "geometric data"]),
        ("Clusters", ["clusters", "cluster analysis"]),
        ("Marks", ["marks", "mark types", "points lines areas"]),
        ("Points", ["points", "point marks"]),
        ("Lines", ["lines", "line marks"]),
        ("Areas", ["areas", "area marks"]),
        ("Visual Channels", ["visual channels", "channels"]),
        ("Visual Encoding", ["visual encoding", "encode data", "encoding data"]),
        ("Position Channel", ["position channel", "spatial position"]),
        ("Color Channel", ["color channel", "color hue", "color saturation", "color luminance"]),
        ("Size Channel", ["size channel", "size encoding"]),
        ("Shape Channel", ["shape channel", "shape encoding"]),
        ("Length Channel", ["length channel", "bar length"]),
        ("Angle Channel", ["angle channel", "angle encoding"]),
        ("Area Channel", ["area channel", "area encoding"]),
        ("Retinal Channels", ["retinal channels"]),
        ("Magnitude Channels", ["magnitude channels"]),
        ("Identity Channels", ["identity channels"]),
        ("Bar Chart", ["bar chart", "bar graph"]),
        ("Line Chart", ["line chart", "line graph"]),
        ("Pie Chart", ["pie chart"]),
        ("Scatter Plot", ["scatter plot", "scatterplot"]),
        ("Histogram", ["histogram"]),
        ("Box Plot", ["box plot", "boxplot"]),
        ("Heatmap", ["heatmap", "heat map"]),
        ("Treemap", ["treemap", "tree map"]),
        ("Radar Plot", ["radar plot", "radar chart", "radar plots"]),
        ("Scatterplot Matrix", ["scatterplot matrix"]),
        ("Parallel Coordinates", ["parallel coordinates"]),
        ("Normalized Stacked Bar Chart", ["normalized stacked bar chart", "100 percent stacked bar"]),
        ("Filtering", ["filtering", "filter interaction"]),
        ("Brushing", ["brushing", "brushing and linking"]),
        ("Linking", ["linking", "linked views"]),
        ("Zooming", ["zooming", "zoom interaction"]),
        ("Panning", ["panning", "pan interaction"]),
        ("Details On Demand", ["details on demand", "detail on demand"]),
        ("Overview First", ["overview first"]),
        ("Focus And Context", ["focus and context"]),
        ("Dynamic Query", ["dynamic query", "dynamic queries"]),
        ("Effectiveness Principle", ["effectiveness principle", "visual effectiveness"]),
        ("Expressiveness Principle", ["expressiveness principle", "visual expressiveness"]),
        ("Chart Junk", ["chart junk", "chartjunk"]),
        ("Data Ink Ratio", ["data ink ratio"]),
        ("Cognitive Load", ["cognitive load"]),
        ("Perspective Distortion", ["perspective distortion"]),
        ("Wrong Abstraction", ["wrong abstraction"]),
        ("Wrong Idiom", ["wrong idiom"]),
        ("Interpolation", ["interpolation"]),
    ]

    found = []
    for canonical, patterns in seed_rules:
        if any(p in normalized for p in patterns):
            found.append(canonical)

    return found

# Filters extracted concepts into valid quiz concepts.
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

        words = concept.split()

        if len(words) == 1:
            continue

        if len(words) > 5:
            continue

        valid.append(concept)
        seen.add(key)

        if len(valid) >= limit:
            break

    return valid

# Builds the final concept pool used for quiz generation.
def build_concept_pool(full_text: str, limit: int = 20) -> List[str]:
    text = clean_text(full_text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\s+", " ", text)

    seed_rules = [
        ("Data Abstraction", ["data abstraction"]),
        ("Task Abstraction", ["task abstraction"]),
        ("Items And Attributes", ["items and attributes", "items attributes"]),
        ("Dataset Types", ["dataset types"]),
        ("Spatial Fields", ["spatial fields", "spatial field"]),
        ("Derived Attributes", ["derived attributes", "derived attribute"]),
        ("Analysis Framework", ["analysis framework"]),
        ("Nested Model", ["nested model"]),
        ("Validation Approaches", ["validation approaches", "validation approach"]),
        ("Wrong Abstraction", ["wrong abstraction"]),
        ("Wrong Idiom", ["wrong idiom"]),
        ("Interpolation", ["interpolation"]),
        ("Perspective Distortion", ["perspective distortion"]),
        ("Bar Chart", ["bar chart"]),
        ("Pie Chart", ["pie chart"]),
        ("Radar Plots", ["radar plots", "radar plot"]),
        ("Heatmap", ["heatmap"]),
        ("Scatterplot Matrix", ["scatterplot matrix"]),
        ("Parallel Coordinates", ["parallel coordinates"]),
        ("Normalized Stacked Bar Chart", ["normalized stacked bar chart"]),
        ("Marks", ["marks", "mark types"]),
        ("Channels", ["channels", "visual channels"]),
        ("Visual Encoding", ["visual encoding", "encode data"]),
        ("Position Channel", ["position channel", "spatial position"]),
        ("Color Channel", ["color channel", "color hue", "color saturation"]),
        ("Size Channel", ["size channel", "size encoding"]),
        ("Shape Channel", ["shape channel", "shape encoding"]),
        ("Length Channel", ["length channel", "bar length"]),
        ("Angle Channel", ["angle channel", "angle encoding"]),
        ("Area Channel", ["area channel", "area encoding"]),
        ("Categorical Attribute", ["categorical attribute", "nominal attribute"]),
        ("Ordered Attribute", ["ordered attribute", "ordinal attribute"]),
        ("Quantitative Attribute", ["quantitative attribute", "numerical attribute"]),
        ("Retinal Channels", ["retinal channels"]),
        ("Magnitude Channels", ["magnitude channels"]),
        ("Identity Channels", ["identity channels"]),
    ]

    rule_concepts = []

    for canonical, patterns in seed_rules:
        if any(p in text for p in patterns):
            rule_concepts.append(canonical)

    tfidf_concepts = extract_key_entities(full_text, top_k=20)
    tfidf_concepts = filter_valid_concepts(tfidf_concepts, limit=8)

    combined = rule_concepts + tfidf_concepts

    seen = set()
    final = []

    for c in combined:
        c = _clean_option(c)
        if not c or _is_bad_concept(c):
            continue
        if c.lower() in seen:
            continue

        seen.add(c.lower())
        final.append(c)

    return final[:limit]

# ============================================================
# PASSAGE RETRIEVAL
# ============================================================
# Splits long text into smaller passages for retrieval.
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

# Retrieves the most relevant passage for a selected concept.
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
# Loads optional Sense2Vec model for distractor generation.
class Sense2VecDistractorGeneration:
    # Initializes the Sense2Vec distractor generator.
    def __init__(self, model_path: str):
        self.s2v = None

        if Sense2Vec is None:
            return

        try:
            self.s2v = Sense2Vec().from_disk(model_path)
        except Exception:
            self.s2v = None

    # Generates related distractor options using Sense2Vec.
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
# CONCEPT EXPLANATIONS
# ============================================================
CONCEPT_EXPLANATIONS = {
    "Data Abstraction": "Data abstraction converts domain-specific information into general data and attribute types that can be used for visualization design.",
    "Task Abstraction": "Task abstraction explains what the user wants to achieve with the data, such as comparing, identifying trends, or finding outliers.",
    "Visual Encoding": "Visual encoding maps data attributes to visual marks and channels so that information can be represented visually.",
    "Marks": "Marks are the basic graphical elements in a visualization, such as points, lines, and areas.",
    "Visual Channels": "Visual channels are visual properties such as position, color, size, and shape that encode data attributes.",
    "Position Channel": "Position is often one of the most effective channels for comparing quantitative values accurately.",
    "Color Channel": "Color can be used to distinguish categories or show magnitude, depending on whether hue, saturation, or luminance is used.",
    "Bar Chart": "A bar chart uses bar length to compare values across categories.",
    "Heatmap": "A heatmap uses color intensity in a grid to represent values across two dimensions.",
    "Parallel Coordinates": "Parallel coordinates show multivariate data by mapping variables onto parallel axes.",
    "Scatter Plot": "A scatter plot shows the relationship between two quantitative variables.",
    "Treemap": "A treemap uses nested rectangles to show hierarchical data and proportional values.",
}
# ============================================================
# MAIN SERVICE
# ============================================================
# Main service for generating quiz questions, distractors, and explanations.
class QuizGenerationService:
    # Loads all models needed for quiz generation.
    def __init__(self):
        self.qg_model = LocalSeq2SeqGenerator(QG_MODEL_DIR, fallback_model_name=FALLBACK_MODEL_NAME)
        self.dg_model = LocalSeq2SeqGenerator(DG_MODEL_DIR, fallback_model_name=FALLBACK_MODEL_NAME)
        self.expl_model = LocalSeq2SeqGenerator(EXPL_MODEL_DIR, fallback_model_name=FALLBACK_MODEL_NAME)
        self.s2v_generator = Sense2VecDistractorGeneration(model_path=S2V_MODEL_DIR)

    # Checks whether a generated question matches the intended concept.
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

    # Detects generic fallback questions that should be rejected.
    def _is_generic_fallback_question(self, question: str) -> bool:
        q = question.strip().lower()
        return q in {
            "which concept best describes the idea presented in the notes?",
            "which concept best describes the idea in the notes?",
            "which concept best describes the notes?",
        }

    # Provides rule-based fallback questions for known concepts.
    def _fallback_question(self, concept: str, passage: str) -> str:
        low = concept.lower()

        if "treemap" in low:
            return "Which visualization uses nested rectangles to show hierarchical proportions?"
        if "scatter plot" in low:
            return "Which visualization is commonly used to show relationships between two quantitative variables?"
        if "bar chart" in low:
            return "Which visualization is commonly used to compare values across categories?"
        if "line chart" in low:
            return "Which visualization is commonly used to show trends over time?"
        if "node link diagram" in low:
            return "Which visualization represents nodes connected by links?"
        if "adjacency matrix" in low:
            return "Which visualization represents relationships in a matrix form?"
        if "brushing" in low:
            return "Which interaction technique highlights selected items across linked views?"
        if "filtering" in low:
            return "Which interaction technique allows users to focus on a subset of data?"
        if "data abstraction" in low:
            return "Which concept translates domain-specific information into generic visualization language?"
        if "items and attributes" in low:
            return "Which concept distinguishes individual entities from their measured properties?"
        if "dataset types" in low:
            return "Which concept classifies tables, networks, fields, and geometry?"
        if "spatial fields" in low:
            return "Which dataset type represents values sampled over a continuous domain?"
        if "derived attributes" in low:
            return "Which concept refers to values computed from original attributes?"
        if "analysis framework" in low:
            return "Which framework asks what, why, and how in visualization design?"
        if "nested model" in low:
            return "Which model organizes visualization design into four interdependent levels?"
        if "validation approaches" in low:
            return "Which concept compares immediate and downstream validation methods?"
        if "wrong abstraction" in low:
            return "Which design error means the user is shown the wrong thing?"
        if "wrong idiom" in low:
            return "Which design error means the chosen representation does not work?"
        if "interpolation" in low:
            return "Which process estimates values between sampled data points?"
        if "task abstraction" in low:
            return "Which concept explains why the user is looking at the data?"
        if "perspective distortion" in low:
            return "Which issue occurs when 3D perspective makes visual comparison misleading?"
        if "heatmap" in low:
            return "Which visualization uses color in a grid to show values across two dimensions?"
        if "scatterplot matrix" in low:
            return "Which visualization compares pairwise relationships across many variables?"
        if "parallel coordinates" in low:
            return "Which visualization represents multivariate data using parallel axes?"
        if "normalized stacked bar chart" in low:
            return "Which chart is used for part-to-whole comparison after normalizing to 100 percent?"
        if "pie chart" in low:
            return "Which chart uses angle to represent part-to-whole proportions?"
        if "bar chart" in low:
            return "Which chart compares values across categories using bar length?"
        if "radar plots" in low or "radar plot" in low:
            return "Which visualization places multiple attributes on axes radiating from a center point?"
        if "marks" == low:
            return "Which concept refers to the basic graphical elements used to represent data?"
        if "channels" == low:
            return "Which concept refers to visual properties used to encode data attributes?"
        if "visual encoding" in low:
            return "Which process maps data attributes to marks and channels?"
        if "position channel" in low:
            return "Which channel is most effective for accurately comparing quantitative values?"
        if "color channel" in low:
            return "Which channel uses hue, saturation, or luminance to encode information?"
        if "size channel" in low:
            return "Which channel changes the size of marks to represent data values?"
        if "shape channel" in low:
            return "Which channel uses different forms to distinguish categories?"
        if "length channel" in low:
            return "Which channel is commonly used in bar charts for value comparison?"
        if "angle channel" in low:
            return "Which channel is commonly used in pie charts to show proportions?"
        if "area channel" in low:
            return "Which channel uses region size to encode quantitative values?"
        if "categorical attribute" in low:
            return "Which attribute type represents distinct groups without natural order?"
        if "ordered attribute" in low:
            return "Which attribute type has a meaningful sequence or ranking?"
        if "quantitative attribute" in low:
            return "Which attribute type represents numerical measurable values?"
        if "retinal channels" in low:
            return "Which channels are perceived visually without changing spatial position?"
        if "magnitude channels" in low:
            return "Which channels are suitable for showing ordered or numerical magnitude?"
        if "identity channels" in low:
            return "Which channels are suitable for distinguishing categories?"
        if "histogram" in low:
            return "Which chart shows the distribution of numerical values using bins?"
        if "box plot" in low:
            return "Which chart summarizes distribution using median, quartiles, and outliers?"
        if "scatter plot" in low:
            return "Which chart is used to examine relationships between two quantitative variables?"
        if "visual channels" in low:
            return "Which concept refers to visual properties used to encode data attributes?"
        if "marks" in low:
            return "Which concept refers to graphical elements such as points, lines, and areas?"
        if "expressiveness principle" in low:
            return "Which principle states that a visualization should show all and only the relevant information?"
        if "effectiveness principle" in low:
            return "Which principle focuses on choosing the most accurate visual encoding for the task?"
        if "data ink ratio" in low:
            return "Which design idea encourages maximizing useful ink and minimizing unnecessary decoration?"
        if "details on demand" in low:
            return "Which interaction technique allows users to request more information about selected items?"
        if "overview first" in low:
            return "Which interaction principle suggests showing the big picture before details?"

        return ""
    
    # Builds a complete rule-based quiz item if model generation fails.
    def build_rule_based_bundle(self, concept: str, full_text: str, concept_pool: List[str]) -> Optional[Dict[str, object]]:
        concept = _clean_option(concept)
        if not concept:
            return None

        passage = get_relevant_passage(full_text, concept)
        question = self._fallback_question(concept, passage)

        if not question:
            return None

        distractors = []
        for other in concept_pool:
            other = _clean_option(other)
            if not other or other.lower() == concept.lower():
                continue
            if other.lower() in [d.lower() for d in distractors]:
                continue
            distractors.append(other)
            if len(distractors) >= 3:
                break

        if len(distractors) < 3:
            fallback_pool = [
                "Data Abstraction",
                "Task Abstraction",
                "Items And Attributes",
                "Dataset Types",
                "Spatial Fields",
                "Derived Attributes",
                "Analysis Framework",
                "Nested Model",
                "Validation Approaches",
                "Interpolation",
                "Perspective Distortion",
                "Bar Chart",
                "Pie Chart",
                "Heatmap",
            ]
            for item in fallback_pool:
                if len(distractors) >= 3:
                    break
                if item.lower() == concept.lower():
                    continue
                if item.lower() in [d.lower() for d in distractors]:
                    continue
                distractors.append(item)

        if len(distractors) < 3:
            return None

        return {
            "question": question[:600],
            "correct": concept[:255],
            "distractors": [d[:255] for d in distractors[:3]],
            "explanation": (
                f"The correct answer is {concept}. "
                f"{CONCEPT_EXPLANATIONS.get(concept, 'This concept best matches the meaning described in the source material.')} "
                f"The distractors are related concepts, but they do not fit the question as accurately."
            ),
            "source_snippet": passage[:1000]
        }

    # Generates a question for a selected concept.
    def generate_question(self, concept: str, passage: str) -> str:
        prompt = f"""
You are generating one university multiple-choice question.

Rules:
- The correct answer must be exactly: {concept}
- Write a concept-identification question
- Ask about the function, purpose, or meaning of the concept
- Maximum 18 words
- End with a question mark
- Do not mention the answer term in the question
- Do not use vague wording like "uses" unless clearly meaningful

Context:
{passage}

Question:
"""
        text = self.qg_model.generate(prompt, max_new_tokens=64, num_beams=4)
        text = _clean_option(text)

        if text and not text.endswith("?"):
            text += "?"

        if self._is_question_aligned(text, concept) and is_valid_question(text):
            return text

        return self._fallback_question(concept, passage)

    @staticmethod
    # Checks whether answer and distractor options are similar in type.
    def same_option_type(answer: str, candidate: str) -> bool:
        a_words = answer.split()
        c_words = candidate.split()

        if abs(len(a_words) - len(c_words)) > 3:
            return False

        return True

    # Generates distractor options for a quiz question.
    def generate_distractors( self, concept: str, question: str, passage: str, concept_pool: List[str], desired_count: int = 3) -> List[str]:
        prompt = f"generate distractors: {concept} <sep> {question} <sep> {passage}"
        text = self.dg_model.generate(prompt, max_new_tokens=96, num_beams=4)

        distractors = []
        parts = re.split(r"<sep>|</sep>|>sep|sep>|[\n\r]+", text, flags=re.I)
        candidates = [_clean_option(x) for x in parts if _clean_option(x)]

        for line in candidates:
            low = line.lower()

            if not is_valid_option(line):
                continue
            if not self.same_option_type(concept, line):
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

        if len(distractors) < desired_count:
            for other in concept_pool:
                other = _clean_option(other)
                if not is_valid_option(other):
                    continue
                if not self.same_option_type(concept, other):
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

        if len(distractors) < desired_count:
            s2v_items = self.s2v_generator.generate(answer=concept, desired_count=6)
            for item in s2v_items:
                item = _clean_option(item)
                if not is_valid_option(item):
                    continue
                if not self.same_option_type(concept, item):
                    continue
                if item.lower() == concept.lower():
                    continue
                if item.lower() in [d.lower() for d in distractors]:
                    continue

                distractors.append(item)
                if len(distractors) >= desired_count:
                    break

        fallback_pool = [
            "Data Abstraction",
            "Items And Attributes",
            "Dataset Types",
            "Spatial Fields",
            "Derived Attributes",
            "Analysis Framework",
            "Nested Model",
            "Validation Approaches",
        ]

        for item in fallback_pool:
            if len(distractors) >= desired_count:
                break
            if item.lower() == concept.lower():
                continue
            if item.lower() in [d.lower() for d in distractors]:
                continue
            if not self.same_option_type(concept, item):
                continue
            if _too_similar(item, concept):
                continue

            distractors.append(item)

        return distractors[:desired_count]

    # Generates a short explanation for the correct answer.
    def generate_explanation(self, concept: str, question: str, distractors: List[str], passage: str ) -> str:
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

        concept_key = _clean_option(concept)

        if concept_key in CONCEPT_EXPLANATIONS:
            return (
                f"The correct answer is {concept_key}. "
                f"{CONCEPT_EXPLANATIONS[concept_key]} "
                f"This matches the question because the source material describes this concept more directly than the other options."
            )

        return (
            f"The correct answer is {concept_key} because it best matches the concept described in the notes. "
            f"The other options are less suitable because they refer to different visualization methods or ideas."
        )

    # Generates one complete quiz question bundle.
    def generate_question_bundle( self, concept: str, full_text: str, concept_pool: Optional[List[str]] = None) -> Optional[Dict[str, object]]:
        concept = _clean_option(concept)

        ALLOWED_SINGLE_WORD_CONCEPTS = {
            "brushing",
            "filtering",
            "interpolation",
            "marks",
            "channels",
            "histogram",
            "treemap",
            "zooming",
            "panning",
            "linking",
            "heatmap"
        }

        if len(concept.split()) == 1 and concept.lower() not in ALLOWED_SINGLE_WORD_CONCEPTS:
            return None

        if not concept or _is_bad_concept(concept):
            return None

        full_text = clean_text(full_text)
        concept_pool = concept_pool or build_concept_pool(full_text)

        if not concept_pool:
            concept_pool = build_concept_pool(full_text)

        passage = get_relevant_passage(full_text, concept)
        question = self.generate_question(concept, passage)
        correct = concept

        if not question:
            return None
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

# =========================================================
# MAIN PIPELINE
# =========================================================
    # Generates a full quiz set from uploaded PDF text.
    def generate_quiz_from_text(self, full_text: str, max_questions: int = 10) -> List[Dict[str, object]]:
        full_text = clean_text(full_text)
        concept_pool = build_concept_pool(full_text)

        bundles = []
        seen_questions = set()
        seen_corrects = set()

        for concept in concept_pool:
            bundle = self.generate_question_bundle(
                concept=concept,
                full_text=full_text,
                concept_pool=concept_pool
            )

            if not bundle:
                bundle = self.build_rule_based_bundle(
                    concept=concept,
                    full_text=full_text,
                    concept_pool=concept_pool
                )

            if not bundle:
                continue

            q_key = _normalize_text(bundle["question"])
            c_key = _normalize_text(bundle["correct"])

            if q_key in seen_questions or c_key in seen_corrects:
                continue

            seen_questions.add(q_key)
            seen_corrects.add(c_key)
            bundles.append(bundle)

            if len(bundles) >= max_questions:
                break

        return bundles

# ============================================================
# BACKWARD-COMPATIBLE WRAPPERS
# ============================================================
# Backward-compatible wrapper for generating one quiz item.
def generate_full_quiz_data(context: str, concept: str):
    service = QuizGenerationService()
    return service.generate_question_bundle(
        concept=concept,
        full_text=context,
        concept_pool=build_concept_pool(context)
    )

# Backward-compatible wrapper for generating a full quiz from text.
def generate_quiz_from_text(context: str, max_questions: int = 10):
    service = QuizGenerationService()
    return service.generate_quiz_from_text(
        full_text=context,
        max_questions=max_questions
    )