# Bookwise — General Document Intelligence CLI

Bookwise is a modular, Gemini-powered command-line application for turning unstructured documents into searchable, multilingual knowledge artifacts.

Although the repository includes a small book collection as a demo corpus, the workflows are domain-neutral. The same commands can be used with books, reports, research papers, manuals, meeting notes, technical documentation, and internal knowledge bases.

## Why this project exists

Reading and searching long documents often requires several repetitive steps: extracting text, summarizing it for different audiences, translating it, preparing review questions, and finding answers across a collection. Bookwise packages those steps behind a consistent CLI and a testable Python architecture.

The project is designed to demonstrate practical AI engineering rather than a single prompt experiment:

- reusable application modules instead of one-off scripts;
- provider isolation so model calls can be mocked in tests;
- explicit configuration with no interactive secret prompts;
- retrieval over a mixed document collection;
- predictable output paths and artifact naming;
- automated linting, testing, and CI.

## Capabilities

### Summarization

Generate one of three formats from a PDF, Markdown, or text file:

- `flash`: a concise, high-signal overview;
- `chapters`: a section-by-section analysis with key ideas, evidence, and implications;
- `executive`: a longer professional summary for decision-makers.

The `--type` option tailors the prompt to the material without restricting the input domain:

```bash
bookwise summarize report.pdf --type report --style executive
bookwise summarize research_paper.pdf --type "research paper" --style chapters
bookwise summarize notes.md --type "meeting notes" --style flash
```

### Translation

Translate a generated artifact while preserving its meaning, tone, paragraphs, and Markdown formatting:

```bash
bookwise translate outputs/report/flash.md --language fr
bookwise translate outputs/report/flash.md --language ar
```

### Structured Q&A

Generate 20 useful question-and-answer pairs focused on definitions, arguments, evidence, examples, and practical lessons:

```bash
bookwise qa report.pdf --type report
```

### Retrieval-augmented question answering

Search a directory recursively across PDF, Markdown, and text files. Documents are split into overlapping chunks, embedded with Gemini, indexed in FAISS, and retrieved before the final answer is generated.

```bash
bookwise ask data/books \
  --question "What do these documents say about consistency?" \
  --top-k 5
```

The assistant is instructed to answer only from retrieved context and to say when the available documents do not contain the answer.

### Text-to-speech

Convert a Markdown or text artifact to an MP3 file. Local playback is optional and uses `mpg123` when available:

```bash
bookwise speak outputs/atomic_habits/flash.md \
  --output outputs/atomic_habits/flash.mp3
```

## Architecture

```text
CLI
 ├── summarize / translate / qa
 │    └── reusable workflows
 ├── ask
 │    └── document discovery → chunking → Gemini embeddings → FAISS → Gemini answer
 └── speak
      └── Markdown/text cleanup → gTTS → optional mpg123 playback

Provider boundary
 ├── GeminiGenerator for production calls
 └── FakeGenerator in tests for offline execution
```

Important modules:

| Module | Responsibility |
| --- | --- |
| `src/bookwise/cli.py` | Command-line interface and argument validation |
| `src/bookwise/workflows.py` | Summary, translation, and Q&A use cases |
| `src/bookwise/rag.py` | Mixed-file discovery, chunking, vector search, and grounded answers |
| `src/bookwise/provider.py` | Gemini integration behind a small `TextGenerator` protocol |
| `src/bookwise/storage.py` | Text extraction, safe filenames, and artifact writing |
| `src/bookwise/config.py` | Environment-based model and project-path configuration |
| `tests/` | Offline workflow, storage, chunking, and CLI tests |

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `GOOGLE_API_KEY` in `.env` or export it in the shell. The application reads the key from the environment and never prompts for or writes credentials.

Optional model settings are also available:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
BOOKWISE_MODEL=gemini-2.5-flash
BOOKWISE_EMBEDDING_MODEL=gemini-embedding-001
BOOKWISE_TEMPERATURE=0.3
```

## Repository layout

```text
src/bookwise/       Installable application package
data/books/         Sample source PDFs used as one example corpus
data/examples/      Curated summaries, translations, Q&A, and audio examples
outputs/             Local generated artifacts; ignored by Git
tests/               Offline automated tests
.github/workflows/   Continuous integration configuration
```

## Development and verification

Run the test and quality checks locally:

```bash
pytest
ruff check .
python -m compileall -q src tests
```

The test suite does not need a Gemini key or network access. Model calls are replaced with a small fake generator, while file parsing, output paths, prompts, chunking, and CLI behavior remain covered.

GitHub Actions runs Ruff and pytest on pushes and pull requests.

## Limitations and responsible use

- Generation quality depends on the selected Gemini model and the quality of extracted text.
- Scanned PDFs may require OCR before they can be processed reliably.
- RAG answers are grounded in retrieved text, but generated answers should still be reviewed for high-stakes use.
- The included books are sample inputs and should be used according to their copyright licenses.
- Generated artifacts are ignored by default; curated examples are stored separately under `data/examples/`.

## CV-ready project description

Built a modular, Gemini-powered document intelligence CLI that extracts and summarizes PDFs/text, generates structured Q&A, translates knowledge artifacts, and answers questions across mixed document collections with FAISS retrieval. Designed reusable workflows behind a provider boundary, added typed configuration, offline tests, CLI ergonomics, and automated CI.

Technologies: Python, Gemini, LangChain, FAISS, PDF/text processing, prompt engineering, CLI design, automated testing, and GitHub Actions.
