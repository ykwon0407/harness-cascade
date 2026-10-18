# AGENTS.md

Guidance for coding agents working in this repository. `README.md` is the
short user-facing pitch; this file holds the details.

## What this project is

`cascade` runs one agent harness (Claude Code, Codex, or Pi) headlessly, or
runs two in sequence with a handoff between them. Every harness talks to an
**embedded HTTP gateway** on `127.0.0.1`, which translates the harness's wire
protocol into the provider's and back. The project is used for benchmark
campaigns (Terminal-Bench style), so **anything that changes what the agent
sees or how it is billed changes the measurement**. Treat such changes as
behavior changes, and say so in commit messages.

## Layout

| Path | Role |
|---|---|
| `src/harness_cascade/__init__.py` | `__version__` and the Python entry point `run(args)`; see "Python API". |
| `src/harness_cascade/cli.py` | Argument parsing, entry point `cascade` / `harness-cascade`. |
| `src/harness_cascade/config.py` | `RunPlan` / `StagePlan`, harness aliases, effort names. |
| `src/harness_cascade/models.py` | Model profile registry (`model_profiles.yaml` plus user overrides). |
| `src/harness_cascade/model_profiles.yaml` | Per provider/model: `efforts`, optional `context_window`, `max_output_tokens`. |
| `src/harness_cascade/engine.py` | `HandoffEngine`: runs stages, the handoff summary, and writes run artifacts. |
| `src/harness_cascade/harnesses.py` | One adapter per harness: builds the command line, env, and config files; parses output. |
| `src/harness_cascade/gateway.py` | `EmbeddedGateway`: local HTTP server, Codex catalog, heartbeat streaming, legacy chat client. |
| `src/harness_cascade/providers.py` | `ProviderClient`: upstream HTTP, retries, upstream streaming, request dumps. |
| `src/harness_cascade/limits.py` | Discovers context window and max output from the provider at gateway start. |
| `src/harness_cascade/protocols/canonical.py` | Protocol-neutral IR: `CanonicalRequest`, `Message`, `Block`, `Tool`, `Usage`. |
| `src/harness_cascade/protocols/{anthropic_messages,openai_chat,openai_responses}.py` | Decode/encode between each wire format and the IR, plus SSE rendering. |
| `src/harness_cascade/protocols/state.py` | `OpaqueStateStore` for signed/encrypted reasoning state across turns. |
| `src/harness_cascade/credentials.py` | Key resolution, `--env-file` loading, endpoint URLs, keeping keys from agents. |
| `src/harness_cascade/renderers.py` | `pretty` / `text` / `jsonl` output. |
| `src/harness_cascade/prompts/` | Packaged `handoff-summary.md` and `handoff-continue.md`. |
| `tests/` | Pytest suite; see "Testing". |
| `examples/` | Runnable demos with expected output and measured cost; `cost.py` prices a run directory. They call a paid API, so they are not part of the test suite. |

`docs/`, `scripts/`, and `tmp/` are gitignored. Put committed images in `assets/`.

## Request flow

```
harness CLI ──(its own wire)──▶ EmbeddedGateway ──decode──▶ CanonicalRequest
                                                           │
                                         ProviderClient.complete (encode for provider)
                                                           │
provider ◀──────────── HTTP (streamed for chat) ───────────┘
   │
   └─▶ decode_response ─▶ CanonicalResponse ─▶ response_payload / sse ─▶ harness
```

| Harness | Wire it speaks to the gateway |
|---|---|
| Claude Code | Anthropic Messages (`/v1/messages`) |
| Codex | OpenAI Responses (`/v1/responses`) |
| Pi | OpenAI Chat Completions (`/v1/chat/completions`) |

| Provider | Upstream protocol |
|---|---|
| `anthropic` | Anthropic Messages |
| `openai` | OpenAI Responses |
| `together` | Chat Completions |

The gateway waits for the complete upstream reply, then renders the stream the
harness expects. Replies slower than `stream_heartbeat_seconds` (10s) on the
chat route commit to a 200 early and send SSE comment heartbeats. After that
point, errors can only be reported in-stream (`openai_chat.sse_error`).

## Handoff lifecycle

1. The first stage runs until `switch_after` **tool replies**: model replies
   that asked for at least one tool, however many. The gateway counts them
   (`SwitchGate`), so the unit is the same for every harness. After the Nth,
   the next request is held and never sent upstream. `gate.reached` then makes
   the adapter kill the harness after `SWITCH_GRACE_SECONDS`. Every tool the
   Nth reply asked for has run by then. Claude Code also gets `--max-turns N`
   and stops there by itself. `turns` = tool replies; `model_turns` = all
   replies.
   - If the first stage ends on its own before N, the handoff still runs by
     default. `--skip-second-if-finished` ends the run there instead, with
     `second_skipped: true` and a `handoff.skipped` event.
2. The same session is resumed with tools disabled and asked for a summary
   using `prompts/handoff-summary.md`. The request offers no tools, and the
   gateway drops any tool call an upstream returns anyway (with a warning), so
   no harness acts after the switch.
3. If the reply text is empty but reasoning is not, the engine uses the first
   `REASONING_SUMMARY_CHARS` (12,000) characters of reasoning and emits a
   warning. A text summary is cut to `SUMMARY_CHARS` (48,000) for the second
   prompt, keeping its start and end; `handoff-summary.md` keeps all of it.
4. The engine records `git status`, `git diff --stat`, and the last 20
   commands, bounded so a huge change cannot fill the second harness's
   context: `SNAPSHOT_LINES` (200) lines each, with "... and K more lines"
   (`--stat` keeps its totals line), and `COMMAND_CHARS` (1,000) per command.
5. The second stage starts fresh with `prompts/handoff-continue.md`. Its
   template variables are `$original_prompt`, `$handoff_summary`, `$workspace`,
   `$git_status`, `$git_diff_stat`, and `$recent_commands`.

Artifacts go to `.cascade/runs/<run-id>/`: `configuration.json`,
`trajectory.jsonl`, `result.json`, `first.log`, and for handoffs `handoff.log`,
`second.log`, `handoff-summary.md`, and `second-prompt.md`.

A failed or timed-out stage carries its cause in `error`. The cause is also
appended to the `EngineError` message and to `result.json`'s `error`. Sources,
in order of preference:
- **The harness's own report:** Pi `message_end` `errorMessage` (Pi exits 0
  after one); Claude Code `result` with `is_error`; Codex `error` or
  `turn.failed`.
- **A signal cascade did not send** (exit -9, or 137/143 when the harness
  handles it). In TB 2.1 these were the OOM killer, or the agent's own
  `pkill -f` matching the harness's argv.
- **The gateway's last error to the agent** (`EmbeddedGateway.last_error`).
- **The last non-JSON output line.**

A stage that ends with no answer keeps its reason too, and the "exited without
an assistant result" message carries it. For example, a Pi reply that reached
the output limit before any text.

Usage of a call that finishes after its stage has ended is folded back in at
the end of the run. Calls still unanswered then are listed under
`unanswered_model_calls`.

## Model limits (`limits.py`)

Resolution order, per provider/model/endpoint, cached once per process:

1. **Provider:** Anthropic `GET /v1/models/{id}` (both limits), and Together
   `GET /v1/models` (`context_length` only). OpenAI publishes no limits and is
   deliberately absent from `_PROVIDERS`.
2. **Profile:** `context_window` / `max_output_tokens` in `model_profiles.yaml`.
3. **Fallback:** `FALLBACK_CONTEXT_WINDOW = 200_000`, `FALLBACK_MAX_OUTPUT_TOKENS = 65_536`.

When the provider and the profile disagree, the provider wins and a warning is
emitted. `CASCADE_MAX_OUTPUT_TOKENS` caps output below the model's limit. The
resolved values feed Pi's `models.json`, Claude Code's
`CLAUDE_CODE_MAX_OUTPUT_TOKENS`, the Anthropic `/v1/models` answer, and the Codex
catalog.

Every upstream request is sent with the model's output limit, or the harness's
own if it asked for less (Codex never asks; `CanonicalRequest.max_tokens == 0`
means "not set"). The gateway then lowers it to what the window has left:
`window - estimated input - 4096`, the estimate being the last reply's
`input + output` plus `chars / 4` of the messages added since
(`EmbeddedGateway._output_limit`, as Pi's `clampMaxTokensToContext`). Together
rejects a request whose input plus `max_tokens` exceeds the window.

Profile rule: **only add a limit that has a source** (the vendor's published
figure or the API's answer). Omit it rather than guess.

## Context compaction (`compact_at_percent`)

Default 85 (`--compact-at`; `0` = each harness's
own). Every harness plans against `W` = the resolved window and compacts at
`T = W * pct // 100`; Codex alone at `min(T, 272_000)`
(`CODEX_COMPACT_AT_MOST`, the input above which OpenAI bills a GPT request at
its long-context rate, and the window Codex ships for its own models). Native
defaults this overrides: behind the gateway,
Claude Code assumes 200k for every model and compacts at ~170–180k; Codex uses
272k for every model, its own GPT models included, and compacts at 90% of it;
Pi compacts at `W - 16384`, only at `agent_end`.

- **Claude Code:** `CLAUDE_CODE_MAX_CONTEXT_TOKENS=W` and
  `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=pct`. Its threshold is
  `min(pct% of E, E - 13k)` with `E = window - min(max output, 20k)`. It
  trusts a Claude model's catalog window (1M for Opus 4.7+) only when talking
  to api.anthropic.com, so behind the gateway it uses 200k for every Claude
  model and ignores `CLAUDE_CODE_MAX_CONTEXT_TOKENS` for them. For a Claude
  model with `W > 200k`, cascade passes `<model>[1m]` plus
  `CLAUDE_CODE_AUTO_COMPACT_WINDOW=min(W, 1M)`. `[1m]` alone leaves it waiting
  for the API to reject the prompt.
- **Codex:** `-c model_context_window=W -c model_auto_compact_token_limit=T`.
  Codex clamps the window to the catalog entry's `max_context_window` and the
  limit to 90% of the window, so every model needs an entry carrying `W`. A
  GPT model gets the catalog Codex ships with its own entry's windows raised
  to `W` (`codex_builtin_catalog`); left at 272k it would compact at 244.8k.
  An open model has no entry, so cascade writes a catalog entry that copies Codex's fallback model
  metadata field for field, using the fallback prompt extracted from the
  binary (`codex_fallback_prompt`), plus the shipped entries, so the
  `spawn_agent` override list doesn't change. If the prompt isn't found, it
  warns and keeps 272k. An A/B request dump showed identical requests apart
  from per-run IDs.
- **Pi:** it never compacts mid-prompt, so the gateway does. It tracks the
  last tool request's `input + output`, and once that reaches `T`, refuses the
  next tool request with `CompactionPoint` (400, `context_length_exceeded`,
  `COMPACTION_POINT_MARKER`). Pi takes that as an overflow: it compacts, then
  continues. The refusal is not a gateway failure (no `last_error`, no
  warning). The Pi adapter records a provisional error that a successful
  `compaction_end` clears. Pi skips compaction when nothing is older than its
  `keepRecentTokens` (20k), and then ends the run, which surfaces as "Pi did
  not compact when asked".

Larger windows mean larger requests: more cost, latency and KV memory per
call. Only Codex is capped; a GPT model run under Claude Code or Pi compacts at
85% of its 1.05M and pays OpenAI's long-context rate past 272k.

## Prompt caching and usage accounting

- **Anthropic** caching is opt-in. `_mark_cache_breakpoints` puts up to four
  `cache_control` markers: the last tool, the system block, and the last two
  messages. It replaces blocks rather than mutating them, because the blocks are
  shared with canonical `Block`s. Mutating them would leak stale markers into
  later turns, and more than four markers is a 400.
- **Prefix stability:** `split_preamble` hoists only the *leading* system
  messages. Mid-conversation system messages, such as Claude Code's per-turn
  token budget, stay in place. Hoisting them would rewrite the prompt prefix
  every turn and defeat OpenAI/Together automatic caching.
- **`Usage.input_tokens` is gross**, meaning it includes cached tokens.
  `cache_read_tokens` / `cache_write_tokens` are a breakdown of it, not
  additions. Anthropic reports input net of cache, so its decoder adds the
  cached parts back. Keep this invariant in every decoder.
- `total_tokens` = input + output. Reasoning is a detail of output and is not
  added again.

## Protocol translation rules worth knowing

- Responses `developer` role becomes `system`. Some chat templates silently drop
  unknown roles.
- Codex **custom (freeform) tools** are offered upstream as a function with one
  `input` string, with the grammar appended to the description. They must be
  answered as `custom_tool_call`, not `function_call`. Codex treats the wrong
  shape as fatal.
- **Namespaced tools** are flattened to `namespace__name` upstream and split
  back into `namespace` + `name` in `response_payload`. `Tool.data["openai"]`
  records the original shape.
- Chat encoding merges consecutive assistant messages, because a Codex parallel
  group arrives as one item per call. It also drops tool calls that never got a
  result, which happens when a handoff cuts off mid-group.
- An Anthropic user turn that has both `tool_result` and text is split: first a
  tool message, then the user text.
- The `multi_tool_use.parallel` sentence is stripped from Codex's prompt
  (`_HOSTED_PARALLEL_SENTENCE`). If Codex's wording changes and the phrase
  survives, the gateway raises instead of sending it. Update the regex; don't
  remove the check.

## Harness adapter specifics (`harnesses.py`)

**Claude Code**
- `CLAUDE_CODE_SIMPLE=1` (bare mode: bash, edit, and read only). This matches
  published Terminal-Bench runs and shrinks the preamble from about 12.5k to
  about 0.8k tokens.
- `--max-turns` is passed only for a summary (1) or a handoff point. Solo
  stages are unlimited: `--switch-after` needs `--second`, and a solo or
  second stage that still ends at a switch point fails the run. Claude Code counts one turn per model call and stops
  after the last call's tools have run. That is the same point the gateway
  would hold it at. A `result` with `subtype: error_max_turns` is read as
  `switch_limit`.
- `API_TIMEOUT_MS` and `CLAUDE_STREAM_FIRST_BYTE_TIMEOUT_MS` equal the
  gateway's request timeout.

**Pi**
- The adapter writes `models.json` with the resolved limits and `settings.json`
  with `httpIdleTimeoutMs` equal to the request timeout.
- An error `message_end` marks the stage failed provisionally, since Pi still
  exits 0. A successful `auto_retry_end` or `compaction_end` (Pi's recovery
  from a context overflow) clears it.

**Codex**
- By default, **no custom catalog** is written for models Codex doesn't know
  (all open models), so Codex uses its fallback metadata.
- A catalog is written only when Codex ships an entry the gateway can't serve
  (the gpt-5.6 tier: `tool_mode` or `use_responses_lite`).
- `CASCADE_CODEX_CATALOG=1|0` forces it on or off.
- When a catalog is written, it is **copied from a real entry in the Codex
  binary** (`codex_reference_entry`), never hand-written.
- The `multi_agent`, `goals` and `view_image` features are off
  (`CODEX_DISABLED_FEATURES`), which leaves `exec_command`, `write_stdin`,
  `update_plan` and `request_user_input` upstream. GLM-5.3-Flash called the
  subagent and goal tools by names that do not exist, and vLLM dropped those
  calls, ending the session early. An image from `view_image` is not
  forwarded upstream (tool outputs are sent as text), but Codex keeps
  re-sending it until the request outgrows `MAX_REQUEST_BYTES` (413).

All adapters scrub upstream keys from the agent's environment. Agents only get
the gateway's local token.

The prompt goes on the command line. Linux caps one argument at 128 KiB, so a
longer prompt (`PROMPT_ARG_BYTES`) is written to the harness's stdin instead
(Claude Code `-p` with no prompt, Codex `-`, Pi with no message); all three
send it upstream exactly as they would the argument. The cascade CLI itself
takes long prompts through `--prompt-file` or stdin.

Codex's model catalog and fallback prompt are read from its 258 MB native
binary through `mmap`, as bytes; decoding it took about 1 GB per process.
Hosted tools Codex declares (`web_search`, and `tool_search` on the GPT
entries it ships) are dropped in translation, since no upstream here runs them.

Stopping a harness that is still running (switch point, stage timeout, Ctrl-C,
SIGTERM/SIGHUP) signals its process group **and every descendant**, found in
`/proc` before the kill: Codex runs each tool command in a session of its own.
A harness that exits by itself leaves the processes it started alone, so a
server the agent brought up survives for the verifier. The CLI turns SIGTERM
and SIGHUP into an interrupt; the run then writes `result.json` with
`status: "interrupted"` and the usage the gateway counted, and exits 128+signal
(SIGTERM: 143), within about 2 s (`KILL_GRACE_SECONDS`).

## Python API (`__init__.py`)

`harness_cascade.run(args)` launches `[sys.executable, "-m", "harness_cascade",
*args]` in a new session and waits for it. It does not run the CLI in-process,
so the CLI's credential scrubbing and `PR_SET_DUMPABLE` stay confined to the
child and never touch the caller.

- The child inherits the caller's stdin, stdout, stderr and environment. Under
  `uv run`, the agents' `python` is therefore the caller's virtualenv.
- The return value is the CLI's exit code. A child killed by a signal (negative
  `wait` status) is reported as `128 + signal`, matching what the CLI returns
  when it handles the signal itself.
- It must be called on the main thread (otherwise `RuntimeError`), because it
  installs SIGINT/SIGTERM/SIGHUP handlers for the duration of the call. The
  first stop signal is forwarded to the child once; the CLI then interrupts the
  run as described above. A signal that arrives before `Popen` returns is held
  and sent once the child exists. Ignored signals stay ignored, and the
  previous handlers are restored afterwards.
- A `str` or `bytes` argument raises `TypeError`, since it would otherwise be
  split into characters.

`uv tool install .` does not make the package importable from the user's
Python. Library use needs `uv pip install` / `pip install` into the caller's
environment.

## Reliability knobs (`providers.py`, `gateway.py`)

| Env var | Default | Effect |
|---|---|---|
| `CASCADE_MAX_RETRIES` | 3 | Re-sends for retryable failures (429, 5xx, transport errors, broken streams). |
| `CASCADE_ATTEMPT_TIMEOUT` | 900 s | Ceiling for a single non-streamed attempt. |
| `CASCADE_MAX_RETRY_AFTER` | 60 s | Longest a server's `Retry-After` is honoured before the request is re-sent anyway. |
| `CASCADE_UPSTREAM_STREAM` | on | Stream chat calls upstream and reassemble them. Set to `0` to disable. |
| `CASCADE_STREAM_IDLE_TIMEOUT` | 120 s | Silence after which a streamed attempt counts as lost and is re-sent. |
| `CASCADE_MAX_OUTPUT_TOKENS` | unset | Campaign-wide output cap. |
| `CASCADE_CODEX_CATALOG` | auto | Force the Codex catalog on or off. |
| `CASCADE_DUMP_REQUESTS` | unset | Directory for request/response JSON dumps from both sides of the gateway. They contain full prompts. |
| `CASCADE_API_KEY_FILE` | unset | Read the upstream key from a file, which is wiped once the preflight passes; see Credentials. |

Backoff is `min(2**attempt, BACKOFF_CEILING_SECONDS=30)`, or `Retry-After` when
it is a number of seconds, capped at `CASCADE_MAX_RETRY_AFTER`. Failed attempts are emitted as `WARNING` events with the reason
and duration. 401/403 fail immediately.

A reply that was billed for output it does not carry is sampled again, up to
`UNDELIVERED_REPLY_RETRIES` (2) more times (`undelivered_tokens`). This applies
only when the request offered tools, the reply has no tool call and was not
truncated, and at least 40 output tokens beyond reasoning are missing, making
up more than half of them. This is how vLLM's tool parser returns a call to a
tool it was not offered. Each lost reply is a `WARNING` ("upstream billed a
reply it did not deliver"). Every sample is a billed model call with its own
`model_call.started`/`finished` pair. After the third loss in a row, the reply
is passed on as it came. A dumped request keeps each replaced reply as
`.response.json` beside `.response.retry<n>.json`.

## Credentials

`TOGETHER_API_KEY`, `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`, each falling
back to `API_KEY`; or `CASCADE_API_KEY_FILE`. The process environment wins over
`--env-file`. The workspace `.env` is **not** read: the agent can read
anything in the workspace, and cascade warns when `--env-file` points inside it.

- **Endpoint.** For `together`, `openai` and `anthropic` it comes only from
  `--llm-endpoint`, else the vendor's own. `LLM_END_POINT` (alias
  `LLM_ENDPOINT`) is ignored for them, with a warning, so a stray variable in a
  shell or a task's `.env` can't redirect a vendor key. A vendor key sent to
  another host, or any key over plain HTTP off loopback, is warned about on
  stderr (`cli._warn_about_endpoint`).
- **Key file.** Wiped (zeroed, truncated, mode 000) only when the key was read
  from it, and only after the preflight passes; `cascade_agent` relies on this
  to keep the key from the agent. A symlink is refused, not wiped through.
- **The cascade process.** After resolution, credential variables are deleted
  and the process is made non-dumpable (`prctl(PR_SET_DUMPABLE, 0)`), so an
  agent of the same user can't read the key from `/proc/<pid>/environ`.
  Children regain dumpability at exec.
- **Redaction.** Upstream and preflight error bodies, which some providers echo
  the key into, are redacted before they reach the agent, and `EventBus`
  redacts every event as a backstop.
- **Inference.** A model the user lists in `--model-profiles` is inferred as
  that provider before any packaged entry with the same id.

A key preflight runs before any agent starts. `--no-preflight` skips it, but
only use that for mocked endpoints.

## Model profiles and efforts

Efforts are validated against the exact `(provider, model)` before credentials
or harnesses are touched. When the provider is omitted, it is inferred from the
registry, and `together` wins ties. Together effort ladders were **measured**:
values an endpoint rejects or silently aliases to another level are left out.
Don't add levels without measuring them. Users can add models with
`--model-profiles file.yaml`.

## Testing

```bash
uv sync --extra dev
uv run pytest                                   # full suite, no network
CASCADE_RUN_CLI_E2E=1 uv run pytest tests/test_cli_e2e.py -v -s   # real CLIs, mock server
```

- `tests/conftest.py` has autouse fixtures that disable limit discovery
  (`limits._PROVIDERS = {}`), reset the limits cache, and set
  `CASCADE_UPSTREAM_STREAM=0`.
  - Tests of discovery must restore `_PROVIDERS` themselves.
  - Tests of streaming must re-enable streaming and fake `httpx.stream`. The
    other tests fake `httpx.post`.
- `tests/test_api.py` covers `harness_cascade.run`: argument checks and exit
  codes through the real CLI with `--dry-run`, and signal relay through a stub
  `harness_cascade` package as the child. No agent or provider is involved.
- Two tests in `test_harnesses.py` need Linux (`/proc` and the 128 KiB argument
  limit) and fail on macOS:
  `test_stopping_a_harness_stops_commands_it_started_in_their_own_session` and
  `test_an_argument_over_the_kernel_limit_is_a_clear_error`.
- Never make real provider calls in the default suite.
- Add a regression test with every protocol fix. The existing tests in
  `test_native_providers.py` and `test_gateway.py` show the pattern: build a
  wire body, run it through decode/encode, and assert on the output.
- Validate CLI changes with `cascade --dry-run ...`, which prints the resolved
  plan without running anything.

## Distribution

The package isn't on PyPI. Install it with `pip install -e ../harness-cascade`,
from a Git tag (`harness-cascade @ git+https://github.com/ykwon0407/harness-cascade.git@<tag>`),
or from a wheel built with `uv build`. The wheel includes the prompts and
`model_profiles.yaml`. For the CLI alone, `uv tool install .` is enough; Python
callers of `harness_cascade.run` need the package in their own environment. `cascade_agent` bundles the package into
`/opt/cascade-tools` in each experiment container.

Version 0.2 removed `provider: auto` and implicit `effort: low`, and
`openai` now means OpenAI itself. Every stage must name its effort.

## Conventions

- Python 3.11+, `from __future__ import annotations`, dataclasses
  (`frozen=True, slots=True` for config), and `httpx` for HTTP.
- Comments explain **why** a constraint exists, often with the measurement
  behind it ("26 of 96 samples…"). Keep that style. Don't write comments that
  restate the code.
- Prefer failing loudly over silently inventing behavior. For example, the
  gateway refuses to hand-write a Codex system prompt, and it refuses to forward
  an unrecognized `multi_tool_use` reference.
- Changes that affect **accuracy** (what the agent sees or can do) or **cost**
  (tokens sent, caching, and usage accounting) must be called out in the commit
  message, because campaign results are compared across commits.
