"""Skill vocabulary: canonical names, aliases, normalization, and a fast text scanner.

Everything that compares skills (profile vs job) must go through this module, otherwise
"React", "ReactJS" and "React.js" count as three different skills and scoring breaks.
Grow `_SKILLS` as you notice misses: it is a plain dict on purpose.
"""
from __future__ import annotations

import re

# canonical name -> extra aliases. The canonical name itself is always an alias too.
_SKILLS: dict[str, list[str]] = {
    # languages
    "Python": ["python3", "py"],
    "JavaScript": ["js", "java script", "ecmascript", "es6"],
    "TypeScript": ["ts"],
    "Java": [],
    "Kotlin": [],
    "Scala": [],
    "Go": ["golang"],
    "Rust": [],
    "C++": ["cpp", "c plus plus"],
    "C#": ["csharp", "c sharp"],
    "C": [],
    "Ruby": [],
    "PHP": [],
    "Swift": [],
    "Objective-C": ["objc", "objective c"],
    "R": [],
    "SQL": [],
    "Bash": ["shell scripting", "shell"],
    "PowerShell": [],
    "Perl": [],
    "Elixir": [],
    "Haskell": [],
    "Dart": [],
    "Julia": [],
    "MATLAB": [],
    # frontend / mobile
    "React": ["reactjs", "react.js", "react js"],
    "React Native": ["react-native"],
    "Angular": ["angularjs", "angular.js"],
    "Vue": ["vue.js", "vuejs", "vue js"],
    "Svelte": [],
    "Next.js": ["nextjs", "next js"],
    "Redux": [],
    "HTML": ["html5"],
    "CSS": ["css3"],
    "Sass": ["scss"],
    "Tailwind CSS": ["tailwind", "tailwindcss"],
    "Webpack": [],
    "Vite": [],
    "jQuery": [],
    "Flutter": [],
    "iOS": [],
    "Android": [],
    # backend / APIs
    "Node.js": ["node", "nodejs", "node js"],
    "Express.js": ["express", "expressjs", "express js"],
    "Django": [],
    "Flask": [],
    "FastAPI": ["fast api"],
    "Spring Boot": ["springboot", "spring", "spring framework"],
    ".NET": ["dotnet", ".net core", "asp.net", "asp.net core"],
    "Ruby on Rails": ["rails", "ror"],
    "Laravel": [],
    "GraphQL": [],
    "gRPC": [],
    "REST APIs": ["rest", "restful", "rest api", "restful api", "restful apis", "restful services"],
    "Microservices": ["microservice", "micro-services", "micro services"],
    "Kafka": ["apache kafka"],
    "RabbitMQ": [],
    "Celery": [],
    "OAuth": ["oauth2"],
    # data
    "PostgreSQL": ["postgres", "psql"],
    "MySQL": [],
    "MongoDB": ["mongo"],
    "SQLite": [],
    "Redis": [],
    "Elasticsearch": ["elastic search"],
    "DynamoDB": ["dynamo db"],
    "Cassandra": [],
    "Snowflake": [],
    "BigQuery": ["big query"],
    "Redshift": [],
    "Databricks": [],
    "Apache Spark": ["spark", "pyspark"],
    "Airflow": ["apache airflow"],
    "dbt": [],
    "Pandas": [],
    "NumPy": [],
    "Hadoop": [],
    "ETL": [],
    "Tableau": [],
    "Power BI": ["powerbi"],
    "Looker": [],
    # cloud / devops
    "AWS": ["amazon web services"],
    "Azure": ["microsoft azure"],
    "GCP": ["google cloud", "google cloud platform"],
    "Docker": [],
    "Kubernetes": ["k8s"],
    "Terraform": [],
    "Ansible": [],
    "Puppet": [],
    "Chef": [],
    "Helm": [],
    "Jenkins": [],
    "GitHub Actions": [],
    "CI/CD": ["cicd", "continuous integration"],
    "Linux": [],
    "Git": [],
    "Nginx": [],
    "Serverless": [],
    "Prometheus": [],
    "Grafana": [],
    "Datadog": [],
    # AI / ML
    "Machine Learning": ["ml"],
    "Deep Learning": [],
    "NLP": ["natural language processing"],
    "Computer Vision": [],
    "PyTorch": ["torch"],
    "TensorFlow": [],
    "Keras": [],
    "scikit-learn": ["sklearn", "scikit learn"],
    "XGBoost": [],
    "LLMs": ["llm", "large language models", "large language model"],
    "RAG": ["retrieval augmented generation", "retrieval-augmented generation"],
    "Generative AI": ["genai", "gen ai"],
    "Hugging Face": ["huggingface"],
    "LangChain": [],
    "MLOps": [],
    # testing / practices
    "Jest": [],
    "Pytest": [],
    "Selenium": [],
    "Cypress": [],
    "Playwright": [],
    "TDD": ["test-driven development"],
    "Unit Testing": [],
    "Agile": [],
    "Scrum": [],
    "System Design": [],
    "Distributed Systems": [],
    "OOP": ["object-oriented programming", "object oriented programming"],
}

# Aliases that are too ambiguous to find in free text ("go to market", "the rest of the team",
# "a node in the cluster"). They still normalize when a user or the LLM supplies them as a skill.
NO_SCAN = {
    "py", "js", "ts", "go", "r", "c", "shell", "rest", "node", "express",
    "spring", "ror", "ml", "torch",
}

# Words that are also common English: only match the capitalized form ("Swift", "RAG"),
# never "swift response" or "a rag".
CASE_SENSITIVE = {
    "swift", "rust", "ruby", "spark", "flask", "dart", "chef", "puppet",
    "julia", "rails", "rag", "helm",
}


def _build_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in _SKILLS.items():
        for alias in (canonical, *aliases):
            key = alias.lower()
            if lookup.setdefault(key, canonical) != canonical:
                raise ValueError(f"alias {alias!r} maps to both {lookup[key]!r} and {canonical!r}")
    return lookup


_ALIAS_TO_CANONICAL = _build_lookup()


def _surface_form(alias: str) -> str:
    return alias.upper() if len(alias) <= 3 else alias.capitalize()


_ci_forms: dict[str, str] = {}
_cs_forms: dict[str, str] = {}
for _alias, _canonical in _ALIAS_TO_CANONICAL.items():
    if _alias in NO_SCAN:
        continue
    if _alias in CASE_SENSITIVE:
        _cs_forms[_surface_form(_alias)] = _canonical
    else:
        _ci_forms[_alias] = _canonical


def _compile(forms, flags: int = 0) -> re.Pattern[str]:
    # longest first, so "react native" wins over "react"; spaces match any whitespace run
    parts = [re.escape(f).replace("\\ ", r"\s+") for f in sorted(forms, key=len, reverse=True)]
    # Not preceded/followed by a letter, digit, +, or #  ->  "Java" does not match inside
    # "JavaScript", and "SQL" does not match inside "NoSQL".
    return re.compile(rf"(?<![A-Za-z0-9+#.])(?:{'|'.join(parts)})(?![A-Za-z0-9+#])", flags)


_CI_RE = _compile(_ci_forms, re.IGNORECASE)
_CS_RE = _compile(_cs_forms)

_STRIP_CHARS = " \t\r\n\u2022\u00b7*-\u2013\u2014,;:()[]{}\"'"


def normalize_skill(raw: object) -> str | None:
    """Map any spelling to its canonical name. Unknown skills are kept (cleaned), not dropped."""
    if not isinstance(raw, str):
        return None
    s = re.sub(r"\s+", " ", raw).strip()
    # drop a trailing note BEFORE stripping bracket characters: "Python (5 yrs)" -> "Python"
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s) or s
    s = s.strip(_STRIP_CHARS).rstrip(".")
    if not s or len(s) > 40:
        return None
    return _ALIAS_TO_CANONICAL.get(s.lower(), s)


def normalize_skills(items: list[str], limit: int = 80) -> list[str]:
    """Normalize, de-duplicate (case-insensitive), keep order."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        skill = normalize_skill(raw)
        if skill and skill.lower() not in seen:
            seen.add(skill.lower())
            out.append(skill)
        if len(out) >= limit:
            break
    return out


def extract_skills_from_text(text: str) -> list[str]:
    """Find known skills in free text (job description or resume). Returns canonical names,
    in order of first appearance. Only finds skills in the vocabulary above."""
    if not text:
        return []
    first_seen: dict[str, int] = {}
    for m in _CI_RE.finditer(text):
        key = re.sub(r"\s+", " ", m.group(0).lower())
        first_seen.setdefault(_ci_forms[key], m.start())
    for m in _CS_RE.finditer(text):
        first_seen.setdefault(_cs_forms[m.group(0)], m.start())
    return [skill for skill, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]
