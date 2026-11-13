# agent-configurator

Build agents on a canvas (Langflow) using LLMs that are configured in the coworkAI **LLM Configurator**, then send them to the **Agent Evaluator** with one click.

A flow stores only the *name* of an LLM config. The LLM Configurator holds the provider key and makes the provider call, so no key is typed into, stored in, or returned by anything in this repo.

```
LLM Configurator :8001  <--  Langflow :7860 + CoworkAI blocks  <--  Agent Evaluator :8002
   (coworkAI, read-only)          ^                                        ^
                                  |                                        |
                         Hand-off service and page :8003  -----------------+
```

Status and open items: `progress.md` in the parent workspace. Design and decisions: `docs/agent-configurator/analysis.md`.

## Start

Needs Windows, Python 3.12, Node (for the page) and, ideally, `uv`.

```
.\start.ps1 -Mock
```

`-Mock` runs a stand-in LLM Configurator on port 8001 that gives scripted replies, so everything can be tried with no provider keys. Without `-Mock` the script uses the real LLM Configurator at `LLM_CONFIGURATOR_URL` (default `http://127.0.0.1:8001`) and never starts or changes it.

The first run installs Langflow (several minutes) and the first Langflow start runs database migrations (a few minutes more). The script then:

1. creates `.env` from `.env.example`,
2. starts Langflow with the blocks loaded,
3. creates a Langflow API key and saves it to `.env`,
4. hides Langflow's own model blocks (see "No keys in Langflow"),
5. starts the Agent Evaluator if the `Agent-Evaluation` repo is next to this one and already set up,
6. serves the hand-off page.

| What | Where |
|---|---|
| Build agents | http://127.0.0.1:7860 (blocks are under **Coworkai** in the sidebar) |
| Evaluate an agent | http://127.0.0.1:8003 |
| Agent Evaluator UI | http://127.0.0.1:5174 |
| Logs | `.langflow/logs/` |

Other switches: `-NoEvaluator`, `-NoBlockPolicy`.

### Starting the real LLM Configurator

It lives in the coworkAI repo, which this project treats as read-only. From `coworkAI/llm-configurator/backend`: `uvicorn main:app --port 8001`. It needs `ENCRYPTION_KEY` in coworkAI's root `.env`, which the coworkAI core platform creates on its first start. Its UI is part of the coworkAI frontend, which must run on port 5173 (the configurator only accepts that origin). That is why the evaluator UI is started on 5174 here.

## The blocks

| Block | What it does |
|---|---|
| **Configured LLM** | Dropdown of active configs from the LLM Configurator (use the refresh button). Outputs a language model. Optional temperature and max tokens; max tokens is lowered to the config's cap. |
| **Configured Embeddings** | Same, for configs that include an embedding model. |
| **Agent** | Langflow's Agent (tools, MCP, memory, max iterations, output format) running on a Configured LLM. Has a name and a role. Its **Agent** output connects to an Orchestrator. |
| **Orchestrator Agent** | Runs several agents together. Patterns: Supervisor, Sequential, Router, Parallel + aggregate. Each agent call shows as a step with the agent's name, input and output. |

Starter flows are in `templates/`. Import one from Langflow's flow list, pick an LLM Config on the Configured LLM block, and open the Playground.

Things to know:

- **Replies arrive in one piece.** The LLM Configurator does not stream. Tool calls and agent steps still appear as they happen.
- **Requests per minute.** The limit is per config and counted per call. One supervisor turn with three agents makes eight or more calls. A block retries a `429` a few times, then reports it. Give configs used for agents a higher `rpm`.
- **PII masking.** With `pii_masking` on, the configurator replaces e-mail addresses and phone numbers in user messages before the model sees them. Use a config without it for agents whose tools need those values.
- **Guardrail blocks** show as an error on the block ("Blocked by guardrail: ..."), not as an empty answer.
- **After changing a block's code**, restart Langflow and run `scripts/build_templates.py`. Flows saved earlier keep the code they were saved with.
- **Field names in blocks must not contain `config` or `kwargs`.** Langflow drops such text values when it loads a flow. A test guards this.

## Evaluating an agent

Open http://127.0.0.1:8003 and click **Evaluate this agent** on a flow. The hand-off service registers the flow in the Agent Evaluator (connection type `langflow`), runs the evaluator's connection test, and links to the evaluator. There, pick the agent by name in step 1 of the wizard.

For a flow with an Orchestrator, each connected agent is also registered on its own (`<flow> / <agent>`). Those registrations run the real flow with the Orchestrator's hidden **Solo Agent** field set, so the agent uses its real LLM, tools and memory. If the Orchestrator block is later replaced or the agent renamed, the evaluator rejects the run and asks you to register again, so a whole flow is never scored under one agent's name.

What the evaluator receives from a run: the answer, every tool call with its input and output (including tool calls made inside sub-agents, tagged with the agent's name), and token usage where Langflow reports it. Retrieval context is taken from tools named in the registration's `retrieval_tools` or from an output named in `retrieval_output_component`.

The Langflow API key is in this repo's `.env` and in the evaluator's encrypted secret store. It is not written into flows and the hand-off API never returns it.

## No keys in Langflow

- None of the four blocks has a key or secret field (a test asserts this).
- The client in `src/agent_configurator/client.py` can only call `GET /configs` and `POST /llm/{name}/chat|embeddings`. Any other route, including `/catalog/global_models` (which returns provider keys in plain text), is refused before a request is sent.
- `scripts/set_block_policy.py` hides Langflow's own **Agent**, **Language Model** and **Embedding Model** blocks through Langflow's catalog policy, and Langflow then refuses flows that contain them. `--clear` undoes it, `--show` prints it.
- Not hidden: blocks that only *have* a model picker (Structured Output, Smart Router, Batch Run and others). They accept a connected Configured LLM, but they also still offer Langflow's provider list.

## Layout

```
components/coworkai/          the four blocks (LANGFLOW_COMPONENTS_PATH points at components/)
src/agent_configurator/       client, chat model, embeddings, trace helpers
src/agent_configurator/handoff/   hand-off service (FastAPI)
frontend/                     hand-off page (React + Vite), built into frontend/dist
templates/                    starter flows
scripts/                      create_langflow_key.py, set_block_policy.py, build_templates.py
tests/                        tests and the stand-in LLM Configurator (mock_configurator.py)
start.ps1 / start.bat
```

## Tests

```
.venv\Scripts\python -m pytest tests
```

They run the blocks inside real Langflow graphs against the stand-in configurator. No network or keys needed.

## Upgrading Langflow

Langflow is pinned to `1.13.2` in `requirements.txt`. The Agent block subclasses Langflow's Agent and uses one private helper, so after changing the version: run the tests, restart Langflow, rebuild the templates, and run one flow through the hand-off page.
