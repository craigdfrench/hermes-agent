# Local Patches

This file documents local patches applied on top of `NousResearch/hermes-agent`
that are pending upstream PR acceptance. If `hermes update` overwrites files,
check out the `working` branch to restore them.

---

## Branch Strategy

Three branches are maintained in `~/.hermes/hermes-agent/`:

| Branch | Purpose |
|--------|---------|
| `working` | Daily driver — upstream `main` + all patches merged in |
| `feat/excluded-providers-and-proxy-grouping` | Clean PR branch → PR #28218 |
| `fix/strip-reasoning-content-for-non-echo-back-providers` | Clean PR branch → PR #28238 |

**Always run Hermes from the `working` branch.**

To update everything after upstream moves:

```bash
hermes-rebase-patch   # defined in ~/.zshrc
```

This fetches upstream, rebases each PR branch onto `origin/main`, force-pushes
to the fork, then resets and rebuilds `working` by merging both branches in.

When a PR is merged upstream: run `hermes update`, remove the branch from
`pr_branches` in `~/.zshrc`, and delete the local branch.

---

## Patch 1: `excluded_providers` + custom provider grouping fix

**Branch:** `feat/excluded-providers-and-proxy-grouping`
**PR:** https://github.com/NousResearch/hermes-agent/pull/28218
**Files changed:** `hermes_cli/model_switch.py`, `hermes_cli/inventory.py`

### Problem

The `/model` picker was not showing the 4 Aperture proxy custom providers
(cerebras, groq, groq-responses, perplexity) because:

1. All providers share the same `base_url` (`http://ai`), which matched the
   built-in `cerebras` endpoint and caused all custom entries to be suppressed
   by the dedup logic.
2. Even when dedup was fixed, the `(api_url, api_key)` grouping key collapsed
   all entries into one row since they share the same proxy URL.

### Fix

**`hermes_cli/model_switch.py`** — Section 4 (custom providers):
- Group key changed from `(api_url, api_key)` → `(api_url, api_key, name_prefix)`
  so providers sharing a base URL but with different names each get their own row.
- Slug assignment: only reuse `current_provider` slug when the provider name
  also matches (prevents all shared-URL entries inheriting one slug).
- Builtin endpoint dedup: skip suppression when the custom entry has an explicit
  `models:` dict — user intentionally defined per-provider model lists on a proxy.
- Live `/v1/models` discovery: skip when `models:` is explicitly defined — the
  proxy endpoint returns all models across all backends, not just this provider's.
- Added `excluded_providers` parameter to `list_authenticated_providers()` with
  checks in sections 1, 2, and 2b.

**`hermes_cli/inventory.py`**:
- Added `excluded_providers: list = None` field to `ConfigContext` dataclass.
- `load_picker_context()` reads `cfg['model_catalog']['excluded_providers']`.
- `build_models_payload()` passes it to `list_authenticated_providers()`.

### Config used

`~/.hermes/config.yaml`:
```yaml
model_catalog:
  excluded_providers:
    - copilot
    - openrouter
    - openai

custom_providers:
  - name: cerebras
    base_url: http://ai
    api_key: <unique-dummy-key>
    models:
      zai-glm-4.7: {}
      llama3.1-8b: {}
      qwen-3-235b-a22b-instruct-2507: {}
      gpt-oss-120b: {}
  # ... groq, groq-responses, perplexity similarly
```

---

## Patch 2: Strip `reasoning_content` for providers that reject it

**Branch:** `fix/strip-reasoning-content-for-non-echo-back-providers`
**PR:** https://github.com/NousResearch/hermes-agent/pull/28238
**Files changed:** `run_agent.py`, `agent/agent_runtime_helpers.py`,
`tests/run_agent/test_deepseek_reasoning_content_echo.py`

### Problem

After switching from `zai-glm-4.7` to `gpt-oss-120b` (Cerebras) in the same
session, the second request failed with:

```
HTTP 400: messages.2.assistant.reasoning_content:
  property 'messages.2.assistant.reasoning_content' is unsupported
```

**Root cause:** Commit `d63abbc3` (upstream, refs #16844) promoted delta
`reasoning_content` from streaming-only reasoning providers (GLM, MiniMax,
gpt-oss-120b) into stored session history so DeepSeek/Kimi history replay
works. When the *next* turn targets a provider that rejects the field
(Cerebras, Groq, Fireworks, Together, etc.), the promoted field poisons the
request.

### Fix

**`run_agent.py`** — added `_rejects_reasoning_content()`:
```python
def _rejects_reasoning_content(self) -> bool:
    return not self._needs_thinking_reasoning_pad()
```
Safe default: strip unless the current provider is known to require echo-back
(DeepSeek V4 thinking, Kimi/Moonshot thinking, MiMo thinking).

**`agent/agent_runtime_helpers.py`** — added early-exit guard at the top of
`copy_reasoning_content_for_api()`:
```python
if agent._rejects_reasoning_content():
    api_msg.pop("reasoning_content", None)
    return
```

---

## Shell Profile Changes (`~/.zshrc`)

- Commented out: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`,
  `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, `JUNIE_OPENROUTER_API_KEY`,
  `COPILOT_PROVIDER_API_KEY`, `COPILOT_PROVIDER_BEARER_TOKEN` — these caused
  unwanted providers to appear in the `/model` picker.
- Added `hermes-rebase-patch()` function — rebases all PR branches and rebuilds
  the `working` branch. See branch strategy above.

---

## Auth Store

Stale credentials for `openrouter` and `copilot` were removed from
`~/.hermes/auth.json` (`credential_pool`). Only `custom:cerebras` remains.
