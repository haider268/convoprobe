"""The offline demo: toy support agent, scenario, and recorded cassette (package data)."""

# Fixed on purpose, not the HOSTED_* defaults: the cassette was recorded with these
# models, and its cache keys include the model spec. Changing the hosted defaults
# must not break the recorded demo.
DEMO_SIMULATOR = "groq:openai/gpt-oss-20b"
DEMO_JUDGE = "groq:openai/gpt-oss-120b"
CASSETTE_NAME = "cassette.json"
