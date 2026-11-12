# agent-configurator

Langflow blocks that build agents on LLMs from the coworkAI LLM Configurator. Flows store only a config name; provider keys stay in the configurator.

Status: early build. See `progress.md` in the parent workspace and `docs/agent-configurator/analysis.md` for the plan.

## Setup (Windows, Python 3.12)

```
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
.venv\Scripts\python -m pytest tests
```

## Run Langflow with the blocks

Set `LANGFLOW_COMPONENTS_PATH` to the `components` folder, `LLM_CONFIGURATOR_URL` to the configurator (default `http://127.0.0.1:8001`), `LANGFLOW_AUTO_LOGIN=true`, then `langflow run --port 7860`.

Without provider keys, a stand-in configurator is available: `python -m uvicorn tests.mock_configurator:app --port 8001`.

## Layout

- `components/coworkai/` blocks: Configured LLM, Configured Embeddings, Agent, Orchestrator Agent
- `src/agent_configurator/` shared client, chat model, embeddings
- `tests/` tests and the stand-in configurator
