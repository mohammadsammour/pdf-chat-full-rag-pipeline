# Research Paper Chat

A local Retrieval-Augmented Generation (RAG) application for asking questions about PDFs. It combines a Streamlit chat interface, a FastAPI backend, persistent Chroma storage, Hugging Face embeddings, cross-encoder reranking, and a Qwen model served by Ollama.

I built this project to learn how to develop and evaluate a RAG system, from PDF processing to a working chat application. I compared four chunking strategies across 14 configurations, selected a character chunking configuration based on retrieval metrics, and tested cross-encoder reranking before integrating it into the application.

On a small ten-question development benchmark, reranking improved Hit@5 from **0.70 to 1.00**, MRR@5 from **0.570 to 0.883**, and judged context support from **0.80 to 0.90**. Mean retrieval/reranking time increased from **0.012 to 1.169 seconds per query**. The experiment details below explain the scoring and its limits.

## Skills developed

| Skill | How I applied it in this project |
| --- | --- |
| Building a RAG pipeline | Connected PDF extraction, chunking, embeddings, vector search, reranking, and grounded answer generation. |
| PDF processing and metadata | Preserved filenames, page numbers, and chunk identifiers so retrieved text could be traced back to its source. |
| Comparing chunking strategies | Tested word, recursive character, character, and semantic chunking with different parameters. |
| Embeddings and vector databases | Used MPNet embeddings and persistent Chroma collections to index and retrieve document chunks. |
| Retrieval evaluation | Used Hit@k, Recall@k, MRR, and nDCG with labeled evidence pages to assess retrieval and ranking. |
| LLM-as-a-judge evaluation | Compared retrieved context against reference answers using a support rubric, alongside deterministic metrics. |
| Cross-encoder reranking | Reranked 50 dense candidates and selected the top 5 chunks for generation. |
| Experiment design | Kept the embedding model and benchmark fixed while comparing chunking, then held chunking fixed while testing reranking. |
| Backend development | Built FastAPI endpoints for document ingestion, chat, and clearing conversation history. |
| Interface development | Built a Streamlit chat interface with PDF uploads, conversation display, and session state. |
| Modular Python code | Separated extraction, chunking, embedding, retrieval, reranking, generation, and interface responsibilities into focused modules. |
| Performance analysis | Measured chunking time in the notebook and considered the retrieval quality versus latency tradeoff when adding reranking. |

## What the application does

- Extracts text from uploaded PDFs and keeps the source filename and page number.
- Splits each page into overlapping character chunks.
- Embeds and stores those chunks in a persistent vector database.
- Retrieves 50 candidate chunks for a question, reranks them, and sends the best 5 to the language model.
- Displays a chat conversation with a fresh input box after each message.
- Prompts the model to cite the filename, page, and chunk beside its answer.
- Allows a new chat while keeping uploaded documents available.

The interface accepts one PDF per upload. Uploading several PDFs one after another adds them to the same collection, so questions search across all documents stored there.

## RAG pipeline

**Document ingestion**

```text
PDF upload
    -> pypdf text extraction, one record per page
    -> character chunking: size 1000, overlap 200
    -> MPNet embeddings
    -> persistent Chroma collection
```

**Question answering**

```text
Question
    -> MPNet query embedding
    -> dense retrieval: 50 candidate chunks
    -> MiniLM cross-encoder reranking
    -> top 5 chunks with source/page/chunk metadata
    -> Qwen3 through local Ollama
    -> answer displayed in Streamlit
```

Dense retrieval compares the question embedding with stored chunk embeddings. The cross-encoder then reads each question–chunk pair together and assigns a relevance score. Reranking changes the order of the retrieved candidates; it cannot recover a chunk missing from the candidate pool.

## Models and tools

| Purpose | Model or tool | Where it is used |
| --- | --- | --- |
| Document and query embeddings | [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | `embedding.py` |
| Cross-encoder reranking | [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | `reranking.py`, explicitly on CPU |
| Answer generation | [qwen3:8b](https://ollama.com/library/qwen3:8b) | `generation.py`, through Ollama |
| LLM evaluation judge | `qwen3:8b` | Recorded retrieval experiments |
| PDF extraction | pypdf | `file_handling.py` |
| Chunking | LangChain text splitters; an existing word chunker | `chunking.py` and the notebook |
| Vector database | Chroma | `vectordb/chroma.py` |
| Backend and interface | FastAPI and Streamlit | `apis.py` and `app.py` |

The embedding model stays fixed across the recorded chunking experiments. Those results compare chunking configurations, rather than different embedding models.

The embedding and reranking models download on first use and are cached locally. Generation uses Ollama at `http://127.0.0.1:11434/v1`. The OpenAI Python SDK acts as a client for Ollama's [OpenAI-compatible API](https://docs.ollama.com/api/openai-compatibility); the configured `"ollama"` API key is a placeholder, and a paid OpenAI API key is not required.

## Project structure

The source code is organized by pipeline responsibility:

```text
chatsystem/
├── README.md
├── app.py
├── apis.py
├── file_handling.py
├── chunking.py
├── embedding.py
├── reranking.py
├── generation.py
├── vectordb/
│   └── chroma.py
├── experiment_notebook.ipynb
├── best_combination.json
├── pyproject.toml
├── uv.lock
├── .python-version
└── .gitignore
```

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit interface: PDF uploads, chat messages, API requests, and starting a new chat. |
| `apis.py` | FastAPI endpoints that connect extraction, chunking, embedding, retrieval, reranking, and generation. |
| `file_handling.py` | Extracts PDF pages into dictionaries containing `text` and a one-based `page` number. |
| `chunking.py` | Word, recursive character, and character chunking functions. Returns LangChain documents with metadata. |
| `embedding.py` | Loads the shared MPNet embedding model and provides `embed_documents()`. |
| `reranking.py` | Loads the CPU cross-encoder and provides `rerank_results()`. |
| `generation.py` | Builds the context, calls Ollama, and maintains chat history in backend memory. |
| `vectordb/chroma.py` | Opens persistent Chroma storage and provides add, upsert, and query functions. |
| `experiment_notebook.ipynb` | Experimental workspace. The saved version currently contains PDF extraction, chunking comparisons, caching, and timing visualization. |
| `best_combination.json` | Records the selected experimental configuration. |
| `pyproject.toml` | Project metadata and declared application dependencies. |
| `uv.lock` | Locked dependency versions for the application environment. |
| `.python-version` | Selects Python 3.13. |
| `.gitignore` | Existing exclusions for Python caches, build artifacts, and the virtual environment. |

Application code lives in the Python modules. Experimental code lives in the notebook.

## Installation and running

### 1. Prepare the environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [Ollama](https://ollama.com/download), then clone or download this repository.

Open a terminal in the project directory and install the locked application dependencies:

```powershell
uv sync --locked
```

The project requires Python 3.13 or newer; `.python-version` selects 3.13. See the [uv project guide](https://docs.astral.sh/uv/guides/projects/) for environment management.

### 2. Download the generation model

```powershell
ollama pull qwen3:8b
```

Keep Ollama running. If its desktop application is already serving requests, no additional server process is needed. Otherwise, run this in a separate terminal:

```powershell
ollama serve
```

Ollama normally listens on port 11434. Its [CLI documentation](https://docs.ollama.com/cli) explains model downloads and server commands.

### 3. Start the backend

From the project directory:

```powershell
uv run --locked uvicorn apis:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Start the interface

In another terminal, from the same directory:

```powershell
uv run --locked streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), upload a PDF using the sidebar, and ask a question after ingestion completes.

The first backend startup can take longer while Hugging Face models download. PDF ingestion and question answering are synchronous, so wait for the displayed operation to finish.

### Hardware considerations

The reranker explicitly runs on CPU. Embedding and generation speed depend on the available hardware and model runtime.

The experiments were designed with a CPU and 8 GB RAM budget in mind. Running Qwen3 8B alongside the embedding model, reranker, database, and notebook can still put pressure on that budget. Start with a few short PDFs and avoid running large notebook experiments while using the chat application. The benchmark timings below do not establish total memory usage or end-to-end chat latency.

## Using the API

| Method | Endpoint | Input | Purpose |
| --- | --- | --- | --- |
| POST | `/documents` | Multipart upload with a field named `file` | Extract, chunk, embed, and store one PDF. |
| POST | `/chat` | JSON containing `question` and `chat_id` | Retrieve, rerank, and generate an answer. |
| DELETE | `/chat/{chat_id}` | Chat ID in the URL | Clear that conversation's backend history. |

Example chat request:

```json
{
  "question": "How many attention heads does the Transformer use?",
  "chat_id": "example-chat"
}
```

Use the same chat ID for subsequent questions in a conversation. Starting a new chat clears conversation history; it does not remove documents from the vector database.

## Chunking and retrieval experiments

The goal was to choose a practical chunking configuration, then test whether reranking improved retrieval while keeping that configuration fixed.

### Stage 1: compare chunking configurations

The notebook defines 14 combinations:

| Strategy | Chunk sizes | Overlaps or thresholds | Combinations |
| --- | --- | --- | --- |
| Word | 250 and 500 words | 0 and 50 words | 4 |
| Recursive character | 1000 and 2000 characters | 0 and 200 characters | 4 |
| Character | 1000 and 2000 characters | 0 and 200 characters | 4 |
| Semantic | Variable, based on semantic boundaries | Percentile thresholds 90 and 95 | 2 |

Word and recursive chunking reuse the functions in `chunking.py`. Character chunking uses LangChain's `CharacterTextSplitter` with an empty separator. Semantic chunking uses `SemanticChunker` from `langchain_experimental`, with MPNet embeddings; it is an experimental notebook option.

The recorded benchmark used:

- Six PDFs searched together.
- At most the first 20 pages of each PDF for chunking.
- Ten answerable development questions with expected source filenames, PDF pages, reference answers, and supporting excerpts.
- The same embedding model, corpus, question set, and evaluation settings across configurations.
- Dense retrieval without reranking for the initial comparison.

This produced 140 configuration–question evaluations. The judge evaluated the retrieved context for those runs.

Chunk sizes are measured in different units: 500 words and 1000 characters are not equivalent text budgets. Semantic chunks also vary in length. These differences are part of the comparison and should be considered when interpreting results.

### Why character chunking was selected

Selected rows from the recorded dense retrieval comparison:

| Strategy and parameters | Hit@1 | Hit@5 | MRR@5 | nDCG@5 | LLM support |
| --- | ---: | ---: | ---: | ---: | ---: |
| Word: 250 words, overlap 0 | 0.30 | 0.70 | 0.403 | 0.475 | 0.85 |
| Word: 500 words, overlap 0 | 0.20 | 0.80 | 0.363 | 0.468 | 0.70 |
| Recursive: 1000 characters, overlap 200 | 0.40 | 0.70 | 0.517 | 0.563 | 0.80 |
| **Character: 1000 characters, overlap 200** | **0.50** | **0.70** | **0.570** | **0.602** | **0.80** |
| Semantic: percentile 90 | 0.40 | 0.70 | 0.483 | 0.536 | 0.85 |

Character chunking with size 1000 and overlap 200 was selected because it gave the strongest early ranking metrics in the comparison, with good judged support. Some other configurations had higher Hit@5 or LLM support. The selection reflects that tradeoff on this small development benchmark.

### Stage 2: add cross-encoder reranking

The selected character configuration and embedding model were held fixed. Each question retrieved the same 50 dense candidates. The cross-encoder scored those candidates, and the best 5 chunks were used as context.

The recorded comparison over the same ten questions was:

| Method | Hit@1 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | LLM support | Mean seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.50 | 0.70 | 0.70 | 0.570 | 0.602 | 0.80 | 0.012 |
| Dense + reranking | 0.80 | 1.00 | 1.00 | 0.883 | 0.913 | 0.90 | 1.169 |

Reranking improved the position of the expected evidence pages and increased the judge's support score. It also added roughly 1.16 seconds per question in this recorded run.

The timing measures retrieval and reranking. It excludes model loading, question embedding, LLM judging, and answer generation. These are recorded development results, not a guarantee for another machine or corpus.

## What I learned from the experiments

- **Configuration choices should follow evidence.** Character chunking was selected for its early ranking performance on this benchmark. Other configurations performed better on some metrics, so there was no single winner across every measure.
- **Retrieval and reranking solve different problems.** Dense retrieval supplies candidates; the cross-encoder improves their order. Missing evidence in the candidate pool remains a retrieval problem.
- **Finding the correct page does not guarantee complete context.** Page-based metrics and judging the actual retrieved chunks provide different information about answer support.
- **Higher retrieval quality has a latency cost.** Reranking improved the recorded scores while adding about 1.16 seconds per query. That tradeoff matters when building an interactive application.
- **Consistent settings make comparisons easier to interpret.** Keeping the model, corpus, and questions fixed helped attribute changes to chunking or reranking.
- **A small benchmark is a starting point.** Results from ten development questions need validation on new questions, and retrieval quality alone does not establish generated answer quality.

## What the evaluation metrics mean

### Deterministic retrieval metrics

The recorded IR metrics compare retrieved source/page pairs against the labeled evidence page. Repeated chunks from the same page are counted once, in order of their first appearance.

| Metric | Meaning |
| --- | --- |
| Hit@1 | Whether the first distinct retrieved page is a labeled evidence page. |
| Hit@5 | Whether a labeled evidence page appears among the first five distinct retrieved pages. |
| Recall@5 | Fraction of labeled relevant pages recovered within those first five pages. |
| MRR@5 | Reciprocal rank of the first relevant page, or zero if none appears by rank 5. |
| nDCG@5 | Gives more credit when relevant evidence appears earlier in the ranking. |

For a question with one labeled relevant page, Recall@5 equals Hit@5. If that page appears at rank `r <= 5`, MRR@5 is `1 / r` and binary nDCG@5 is `1 / log2(r + 1)`. Dataset scores are averages over questions.

These calculations use ground truth and do not require an LLM. A page hit establishes that the expected page was retrieved; it does not establish that the returned chunk contains the complete answer.

In the reranking experiment, page metrics were computed from the ranked candidate list after deduplicating pages. The LLM judge received the first five actual chunks. Therefore, “top five pages” and “top five chunks” can describe different context coverage.

### LLM-as-a-judge support

The judge assessed whether the retrieved context supported the reference answer:

- **0:** no support.
- **1:** partial support.
- **2:** full support.

Scores were divided by two before averaging, giving the reported 0–1 LLM support value. A value of 0.90 is a mean rubric score, not a 90% answer accuracy estimate.

The recorded judge used `qwen3:8b`, temperature 0, seed 42, thinking disabled, a 4096-token context setting, and up to 256 output tokens. The supplied retrieval context was limited to five chunks and 6000 characters.

The judge reads the actual retrieved text, which helps identify page hits with incomplete evidence. Its decisions can still be mistaken or sensitive to prompts and truncation. Human review of questions, evidence, and a sample of judgments remains useful.

### Scope of the results

These experiments evaluated retrieval and whether retrieved context supported reference answers. They did not comprehensively evaluate generated answer correctness, faithfulness, citation accuracy, or the complete chat experience.

The ten questions were used during development rather than held out for a final test. Results are useful for choosing a starting configuration, but should be checked on new questions before drawing broader conclusions.

## Running the experimental notebook

### Extra notebook dependencies

The current application manifest does not declare all notebook dependencies. After `uv sync --locked`, install the experimental and visualization tools into the project environment:

```powershell
uv pip install "langchain-experimental==0.4.2" matplotlib pandas altair ipykernel jupyterlab
```

These additional packages are not covered by the current application lockfile. The saved timing chart uses Altair; Matplotlib is included for the alternative plotting cells used during experimentation.

Open the notebook in your editor and select the project's `.venv` Python kernel. Alternatively, launch JupyterLab using that environment:

Windows:

```powershell
./.venv/Scripts/python.exe -m jupyterlab
```

Linux/macOS:

```bash
./.venv/bin/python -m jupyterlab
```

### Experiment reproducibility

**The saved notebook currently contains extraction, chunking combinations, chunk caching, and timing visualization. The complete retrieval, metric calculation, LLM judging, and reranking evaluation cells are not yet present in the saved notebook.**

The tables in this README report results recorded during development. A fresh clone can run the saved chunking experiment, but cannot yet regenerate the complete evaluation tables solely by running that notebook.

Reproducing the full comparison requires the question dataset and the retrieval, judging, and reranking cells. The ground truth consists of questions, reference answers, expected source/page labels, and verified supporting excerpts. Cached chunks and generated scores are local experiment outputs; they cannot replace those inputs.

### Running the saved chunking experiment

1. Download the PDFs and place them in the project directory.
2. Match the filenames in the notebook's `file_names` list to the actual files.
3. Run extraction and chunking cells in order. The current page limit is `PAGES_PER_FILE = 20`.
4. Run the save cell to create `chunking_results.json`.
5. On a later session, run the imports/function definitions and the load cell to restore chunking results without repeating chunking.
6. Run the timing visualization cell.

Extraction currently reads the entire PDF before the page limit is applied to chunking. The cell that prints extracted pages can also produce very large output for long manuals.

If you change the files, page limit, or chunking parameters, regenerate the corresponding chunks and downstream embeddings/results rather than reusing the old cache.

## Evaluation corpus

The development corpus used the following filenames. Public PDFs are not bundled with the source repository; download them from their official sources and name them to match the notebook.

| Filename | Document and source | What it contributes |
| --- | --- | --- |
| `Attention Is All You Need.pdf` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Research explanations, equations, and result tables. |
| `Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.pdf` | [Original RAG paper](https://arxiv.org/abs/2005.11401) | Related terminology and comparison questions across research papers. |
| `VoiceAgents.pdf` | [Voice-agent tutorial, arXiv 2603.05413](https://arxiv.org/abs/2603.05413) | Technical architecture and latency measurements. The local benchmark used version 2. |
| `NIST AI Risk Management Framework 1.0.pdf` | [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) | Definitions, long sections, headings, and framework descriptions. |
| `NASA Systems Engineering Handbook, Rev. 2.pdf` | [NASA Systems Engineering Handbook](https://www.nasa.gov/reference/systems-engineering-handbook/) | A long engineering manual with lifecycle phases and structured guidance. |
| `PostgreSQL 17 manual PDF.pdf` | [PostgreSQL 17 documentation](https://www.postgresql.org/docs/17/index.html) | Large technical documentation with examples, headings, and many similar terms. |

Use one-based physical PDF page numbers for labels, as returned by `file_handling.py`. Printed page numbers inside a document can differ. Different PDF versions can also shift page numbers and alter extracted text.

The small benchmark's questions covered five of the six PDFs; PostgreSQL contributed corpus content without labeled questions. Its first 20 pages contain substantial front matter. The first-20-pages limit also excludes later sections of the other manuals.

For a more representative follow-up, include evidence from later pages, tables, similar passages, and questions needing multiple pieces of evidence. Keep a few new questions separate from configuration selection. Review labels directly against the PDFs; missing legitimate evidence pages can make correct retrieval appear to be a failure.

## Configuration and storage

Current application settings are defined in code:

| Setting | Current value | Location |
| --- | --- | --- |
| Chunking | Character, size 1000, overlap 200 | `apis.py` |
| Embedding model | `sentence-transformers/all-mpnet-base-v2` | `embedding.py` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L6-v2`, CPU | `reranking.py` |
| Candidate count / final chunk count | 50 / 5 | `apis.py` |
| Generation model and server | `qwen3:8b`, local port 11434 | `generation.py` |
| Backend URL used by the interface | `http://127.0.0.1:8000` | `app.py` |
| Application collection | `Research_Papers_character` | `vectordb/chroma.py` |
| Database directory | `vectordb/chroma_data/` | `vectordb/chroma.py` |

`best_combination.json` records the experimental selection, including `run_10` and the evaluation collection named `test`. **The application does not read this JSON to configure itself.** Changing it alone will not change application behavior.

The application collection and experimental `test` collection are separate. Chroma persists locally, so stored documents remain after restarting the backend. Its directory contains chunk text, metadata, and embeddings, not just configuration.

If you change the embedding model or chunking strategy, use a fresh collection and re-ingest documents so old and new representations are not mixed. Chroma's distance metric is left at the collection default; the code does not explicitly configure cosine distance.

## Limitations and practical notes

- **PDF extraction:** pypdf does not provide OCR here. Scanned pages can yield no useful text, and tables or multi-column layouts can extract in a confusing order.
- **Chunk boundaries:** each page is chunked separately. Evidence spanning pages can require retrieving multiple chunks.
- **Model input length:** character or word counts are not token counts. MPNet's [model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) documents default truncation beyond 384 word pieces, so oversized chunks may lose information during embedding.
- **Small evaluation set:** ten development questions and a restricted page range cannot establish broad retrieval quality. Page labels and judge context truncation also influence the scores.
- **Repeated uploads:** document IDs depend on filename, page, and chunk position. Reuploading the same filename can collide with existing IDs; the upload endpoint uses `add` and does not implement automatic replacement.
- **Shared corpus:** all application uploads go into one collection. There is no per-user document isolation, authentication, or document-management interface.
- **Chat history:** history is stored in backend memory and disappears when that process restarts. Long conversations are not currently trimmed or summarized.
- **Citations:** the generation prompt requests citations, but the application does not independently validate them.
- **Local execution:** the interface and API are configured for a local development setup. First downloads and CPU inference can take time.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Upload or chat reports a connection error | Keep the FastAPI backend running on port 8000. |
| Generation cannot connect | Confirm Ollama is serving on port 11434 and `ollama pull qwen3:8b` has completed. |
| First startup appears slow | Embedding and reranking models may still be downloading or loading. |
| Notebook reports a missing PDF | Check its working directory and the exact filenames in `file_names`. |
| Notebook cannot load cached chunks | Run the chunking/save cells once to create `chunking_results.json`. |
| Semantic or plotting imports fail | Install the extra notebook dependencies into the selected kernel environment. |
| Answers appear to reference old documents | The Chroma collection persists across restarts and contains all earlier uploads. |
| Memory usage or CPU time becomes excessive | Start with fewer/shorter documents and avoid concurrent notebook/model workloads. |

Future evaluation can assess generated answers and citation correctness using new, reviewed questions while holding the selected retrieval configuration fixed.