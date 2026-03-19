# judicial-rag

Sistema de extracción, clasificación y respuesta de variables sobre documentos judiciales en PDF. El usuario sube los PDFs, define las preguntas, y el agente recupera las respuestas junto con el fragmento exacto del documento del que las extrajo.

## Stack

| Capa | Tecnología |
|---|---|
| UI | Streamlit |
| Extracción de PDF | Docling |
| Embeddings | sentence-transformers |
| Vector store | ChromaDB (local, por sesión) |
| LLM | Anthropic Claude / OpenAI |
| Orquestación | LangGraph |
| Validación | Pydantic v2 |
| Despliegue | Hugging Face Spaces + Docker |

## Estructura del proyecto

```
judicial-rag/
│
├── app/
│   ├── main.py
│   ├── components/
│   │   ├── uploader.py
│   │   ├── question_form.py
│   │   └── results_viewer.py
│   └── session_state.py
│
├── pipeline/
│   ├── orchestrator.py
│   ├── extractor.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── rag_agent.py
│
├── models/
│   ├── document.py
│   └── query.py
│
├── services/
│   ├── llm_client.py
│   ├── ocr_service.py
│   └── parallel_runner.py
│
├── storage/
│   ├── session_manager.py
│   └── cleanup.py
│
├── config/
│   ├── settings.py
│   └── prompts.py
│
├── tests/
│   ├── test_extractor.py
│   ├── test_rag_agent.py
│   └── fixtures/
│
├── .huggingface/
│   └── README.md
│
├── Dockerfile
├── CHANGELOG.md
├── pyproject.toml
├── .env.example
└── README.md
```

## Requisitos previos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) para gestión de dependencias
- API key de Anthropic o OpenAI

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/IgnacioFreire/Judicial-RAG.git
cd judicial-rag

# Crear entorno virtual e instalar dependencias

# Copiar y configurar variables de entorno
cp .env.example .env
```

Edita `.env` con tus credenciales:

```env
ANTHROPIC_API_KEY=sk-...
# o bien
OPENAI_API_KEY=sk-...
```

## Uso

```bash
# Arrancar la aplicación
uv run streamlit run app/main.py
```

La interfaz estará disponible en `http://localhost:8501`.

1. Sube uno o varios PDFs desde el panel izquierdo
2. Define las preguntas o variables que quieres extraer
3. Pulsa **Ejecutar** y el agente procesará cada PDF en paralelo
4. Los resultados muestran la respuesta y el fragmento de texto fuente por cada PDF y pregunta

## Desarrollo

```bash
# Instalar dependencias de desarrollo
uv sync --group dev

# Instalar hooks de pre-commit
uv run pre-commit install

# Ejecutar tests
uv run pytest

# Ejecutar tests con cobertura
uv run pytest --cov=pipeline --cov-report=term-missing

# Lint y formato
uv run ruff check .
uv run ruff format .
```

### Commits

Este proyecto usa [Conventional Commits](https://www.conventionalcommits.org/). Usa `commitizen` para generar mensajes de commit guiados:

```bash
uv run cz commit
```

### Versionado

El proyecto sigue [Semantic Versioning](https://semver.org/). Para subir versión:

```bash
uv run bump-my-version bump patch   # 0.1.0 → 0.1.1
uv run bump-my-version bump minor   # 0.1.0 → 0.2.0
uv run bump-my-version bump major   # 0.1.0 → 1.0.0
```

Tras subir versión, actualiza el changelog:

```bash
uv run cz changelog
```

## Despliegue en Hugging Face Spaces

El proyecto se despliega automáticamente en Hugging Face Spaces vía Docker al hacer push a la rama `main`.

Configura los siguientes secretos en tu Space:

- `ANTHROPIC_API_KEY` o `OPENAI_API_KEY`

## Privacidad y datos sensibles

- Los PDFs se almacenan únicamente en un directorio temporal por sesión y se eliminan al cerrar.
- El contenido de los documentos no se loguea en ningún momento.
- ChromaDB opera en modo local sin ninguna conexión externa.
- Cada sesión de usuario está aislada del resto.

## Versión

`0.1.0`
