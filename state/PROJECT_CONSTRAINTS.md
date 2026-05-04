# Project Constraints

## Global constraints
- Build a minimal Claude Code orchestration control layer for advanced AI and research coding projects.
- Optimize for disciplined phases, small diffs, targeted checks, and compact handoffs.
- Prefer plain Markdown, YAML recipes, and standard-library Python.

## Forbidden actions
- Do not add LangGraph.
- Do not add CrewAI.
- Do not add AutoGen.
- Do not add vector databases.
- Do not add a web UI.
- Do not add background workers.
- Do not create an agent swarm.
- Do not turn this into a generic chatbot framework.
- Do not vendor reference repositories or large external code.

## Style rules
- Keep files concise and reusable.
- Make scope boundaries explicit.
- Use existing repository patterns when copied into another project.
- Favor concrete checklists and outputs over abstract process language.

## Testing rules
- Run targeted checks first.
- Use full checks only when risk justifies the time.
- Record commands and results in state when they matter for handoff.
- Treat missing tests as a risk, not as success.

## Data/checkpoint handling rules
- Exclude data, datasets, checkpoints, artifacts, runs, wandb, logs, build output, and virtual environments from context by default.
- Do not paste large generated files into Claude context.
- Summarize large assets by path, size, and purpose only.

