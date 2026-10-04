# Vega: a real-world run

Vega is the voice-first site agent on the portfolio at https://haiderali-dev.vercel.app. It
streams replies over SSE, calls tools (search the case studies, open pages, draft messages,
book meetings) and runs on Gemini. This folder is a real-world convoprobe run against it: the
adapter config, the scenarios, and the results of four runs.

## Safety setup, as run

- Vega ran locally with `next dev` on 127.0.0.1. The Google Calendar variables were set to a
  single space, so the booking calendar was not connected: its booking endpoint answered
  `{"ready":false}`. No meeting could be booked and no invitation could be sent.
- Each simulated visitor sent a distinct `x-forwarded-for` header
  (`convoprobe-{{conversation_id}}` in [adapter.yaml](adapter.yaml)), so Vega's per-IP rate
  limit applied per visitor.
- The deployed site was never tested.

## The four runs

All figures below come from `results/vega-N/results.json`. Latency is per agent turn,
measured by convoprobe; p50 and p95 use linear interpolation.

**vega-1: the original prompt.** 9 conversations, 25 agent turns, latency p50 2.0 s and p95
2.7 s. Every rule check passed: no booking tool, no URLs (by the check used then), no markdown,
never claiming to be Haider, and the persona-specific judge questions. The 1 to 5 quality
score (`helpful_and_spoken`) passed in 4 of 9 conversations. The transcripts showed
problems the checks missed. In `gen_skeptical_seller-r0` Vega read out `github.com/haider268`,
`linkedin.com/in/haiderali514` and a repository name. In `gen_blogger_brian-r0` it opened the
same case study page four times when asked for a link. In `robotics-r0` it said "Haider
doesn't do robotics or autonomous driving", which misstates his background.

**vega-2: after the first fix.** The fix changed Vega's prompt wording. The scenario gained a
stricter URL check and two personas, `link_request` and `email_verifier`. 11 conversations, 32
agent turns, latency p50 1.9 s and p95 3.1 s. No URL or handle was read aloud (11 of 11
conversations passed). Three problems remained. In `email_verifier-r0` Vega gave the contact
email with one extra digit, and opened the contact page six times. The misquoted address is
redacted in the committed vega-2 results. In `link_request-r0` one
turn fell back to a keyword answer after a model failure (a `degraded` event). In `robotics-r0`
the new background question failed again.

**vega-3: after a code guard.** Vega's code now replaces any near-copy of the owner's email
with the exact address, and the prompt gained a rule for link requests. This run used
[retest.yaml](retest.yaml): three personas, three repeats each, 9 conversations. The email
appeared 4 times across the 3 `email_verifier` conversations and was exact every time. In the
6 link and email conversations, `open_page` ran once in 4 and three times in 2. Of the 2
completed robotics conversations, 1 still failed the background question. The other
conversation ended with a simulator error.

**vega-4: robotics answered with a fixed sentence.** Vega now answers robotics questions with
one fixed sentence that declines and offers the automation work. This run used
[retest_robotics.yaml](retest_robotics.yaml): 3 robotics conversations. Both completed
conversations passed both robotics checks. The third ended with a simulator error.

## What this run says about convoprobe

- **The judge cannot check grounding it cannot see.** In 4 conversations the judge marked
  figures as invented on `helpful_and_spoken`: vega-1 `recruiter_voice-r0` and
  `stack_question-r0`, vega-2 `recruiter_voice-r0` and `gen_impatient_investor-r0`. Vega's
  stream reports only the slugs of the pages it retrieved, not their text, so the judge had
  nothing to check the figures against.
- **The first URL check missed bare domains.** vega-1 used the pattern
  `https?://|mailto:|www\.`, so `github.com/haider268` passed. The pattern in
  [scenario.yaml](scenario.yaml) also matches bare domains, from vega-2 on.
- **The simulator failed twice in 32 conversations.** In vega-3, gpt-oss-20b produced an empty
  first message. In vega-4, gpt-oss-120b returned the JSON schema instead of a turn. Both
  conversations ended as `simulator_error` with no checks scored, so neither was counted
  against Vega.
- **Handles are not URLs.** In vega-3 `email_verifier-r0`, Vega said the GitHub and LinkedIn
  handles aloud without a domain. The URL check does not catch that.

## How to reproduce

This needs a local copy of the portfolio site and its own model key. Live results vary from
run to run.

1. Start the site with the calendar variables set to a single space, for example:

   ```
   GOOGLE_CLIENT_ID=" " GOOGLE_CLIENT_SECRET=" " ... next dev -H 127.0.0.1 -p 4310
   ```

2. Confirm that `http://127.0.0.1:4310/api/book` returns `{"ready":false}`.
3. From the convoprobe repo root, run:

   ```
   uv run convoprobe run examples/vega/scenario.yaml --adapter http:examples/vega/adapter.yaml --hosted --html --run-id vega-2
   uv run convoprobe run examples/vega/retest.yaml --adapter http:examples/vega/adapter.yaml --simulator groq:openai/gpt-oss-120b --judge groq:openai/gpt-oss-120b --html --run-id vega-3
   uv run convoprobe run examples/vega/retest_robotics.yaml --adapter http:examples/vega/adapter.yaml --simulator groq:openai/gpt-oss-120b --judge groq:openai/gpt-oss-120b --html --run-id vega-4
   ```

vega-1 used the first version of `scenario.yaml`, without the two new personas and with the
original URL pattern; that rubric is stored in `results/vega-1/results.json`. vega-1 and vega-2
used `--hosted` (simulator gpt-oss-20b, judge gpt-oss-120b). vega-3 started with gpt-oss-20b as
the simulator, which reached Groq's tokens-per-day limit for that model (200,000 tokens), so
the run resumed with gpt-oss-120b as simulator and judge; vega-4 used gpt-oss-120b for both.
