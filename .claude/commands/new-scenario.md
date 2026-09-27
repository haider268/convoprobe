Create a new convoprobe scenario YAML from this description: $ARGUMENTS

1. Read `src/convoprobe/datamodels.py` for the schema: `Scenario`, the three
   persona shapes (LLM persona with `description` and `goal`, scripted persona
   with `script`, or `generate: N`), and the rubric item types.
2. Write the file under `examples/scenarios/` with a short snake_case name.
   Ids match `^[a-z0-9_]+$`. Use only fake people and `.example` domains.
   Prefer deterministic checks (tool_called, tool_order, regex, max_turns)
   over `llm_judge` wherever the rule can be decided in code.
3. Validate it from the convoprobe folder:
   `uv run python -c "from convoprobe.scenario import load_scenario; load_scenario('<path>')"`
4. If it reports errors, each line names the field (for example
   `rubric.1.pattern`). Fix them and validate again until it loads cleanly.
5. Report the file path and a one-line summary of its personas and rubric.
