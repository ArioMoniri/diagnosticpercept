# Git-bus bridge

Drive a live Colab session through `git push`.

## How it works

| direction | path |
|---|---|
| tasks pushed *to* Colab | `bridge/queue/<task_id>.json` |
| results pushed *back* | `bridge/log/<task_id>.json` |

Colab runs `src.bridge.run_bridge_loop(...)` in a persistent cell. It polls
`origin/main` every 30 s, executes any new queue file, and commits the result
back. The same `lm` and other globals from the notebook session are injected
into each task so no model reload is needed.

## One-time setup in Colab

1. Add a **Colab secret** named `GH_TOKEN` containing a GitHub PAT with `repo`
   scope on this repository.
2. The notebook's bridge cell calls `setup_git_auth()` automatically; that
   rewrites `origin` to embed the token for push.

## Task spec

```json
{
  "task_id": "20260520-100000-print-summary",
  "kind": "eval",
  "code": "import json; print(json.loads(open('/content/results/h6/summary.json').read()))"
}
```

or

```json
{
  "task_id": "20260520-101500-sycophancy-probe-500",
  "kind": "call",
  "module": "src.sycophancy",
  "function": "run_sycophancy_probe",
  "args": {"n_questions": 500}
}
```

The `lm` global is auto-injected when the called function accepts an `lm`
parameter.

## Result format

```json
{
  "task_id": "...",
  "ok": true,
  "kind": "eval",
  "stdout": "...",
  "stderr": "",
  "return_value_repr": "...",
  "traceback": null,
  "wall_seconds": 12.3
}
```

If a task fails, `ok=false` and `traceback` carries the full stack.

## Pushing a task from local

```bash
python -c "from src.bridge import push_task; push_task('.', \
  task_id='20260520-test', kind='eval', code='print(2+2)')"
git add bridge/queue/ && git commit -m 'bridge: test' && git push
```
