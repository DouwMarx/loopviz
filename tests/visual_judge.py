"""Headless Claude as a picture judge for the visual tests.

judge() runs `claude -p` (nesting guard removed so it works from inside a
Claude Code session), lets it Read the PNGs, and parses the JSON verdict on
the last line. Prompt and raw answer are saved under runs/visual/<test>/.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs" / "visual"

PROMPT = """You are a strict visual inspector for a 3D-print pipeline test.
Use the Read tool on EVERY image path below, in order. Do not guess from
the file names; look at the pixels.
{images}

Question: {question}

Step 1: for each image write 2-3 sentences describing exactly what you see
(shapes, orientation, which side things are on, text if any).
Step 2: decide. Be literal: if the picture does not clearly show what the
question asks for, the answer is false. Do not give the benefit of the doubt.
On the LAST line output only a JSON object, nothing after it:
{{"pass": true or false, "observed": "<one sentence, what you saw>", "reason": "<one sentence, why pass or fail>"}}
"""


def judge(images: list[Path], question: str, model: str = "sonnet",
          test_name: str = "adhoc") -> dict:
    """Ask Claude whether `images` answer `question`. Returns the parsed
    verdict plus 'raw', 'cost_usd' and 'log' (path of the saved record)."""
    if shutil.which("claude") is None:
        raise RuntimeError("the `claude` CLI is not on PATH; visual tests need it")
    model = os.environ.get("LOOPVIZ_VISUAL_MODEL", model)
    paths = [str(Path(p).resolve()) for p in images]
    for p in paths:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
    prompt = PROMPT.format(images="\n".join(f"- {p}" for p in paths), question=question)
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    out_dir = RUNS / test_name
    out_dir.mkdir(parents=True, exist_ok=True)
    k = len(list(out_dir.glob("judge_*.json")))
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json",
           "--allowedTools", "Read"]
    proc = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, check=False,
                          text=True, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
    reply = json.loads(proc.stdout)
    raw = reply.get("result", "")
    blobs = re.findall(r"\{[^{}]*\}", raw)
    try:
        verdict = json.loads(blobs[-1])
    except (IndexError, json.JSONDecodeError):
        verdict = {"pass": False, "observed": raw[-500:], "reason": "no JSON verdict in reply"}
    verdict = {"pass": bool(verdict.get("pass", False)),
               "observed": str(verdict.get("observed", "")),
               "reason": str(verdict.get("reason", "")),
               "raw": raw, "model": model, "images": paths,
               "cost_usd": reply.get("total_cost_usd")}
    log = out_dir / f"judge_{k}.json"
    log.write_text(json.dumps({"prompt": prompt, "question": question, **verdict}, indent=1))
    verdict["log"] = str(log)
    return verdict


def judge_passes(images: list[Path], question: str, test_name: str, **kw) -> dict:
    """judge() that asserts pass, printing the observation on failure."""
    v = judge(images, question, test_name=test_name, **kw)
    print(f"[{test_name}] pass={v['pass']} cost=${v['cost_usd']}\n"
          f"  observed: {v['observed']}\n  reason: {v['reason']}\n  log: {v['log']}")
    assert v["pass"], f"judge failed: {v['observed']} | {v['reason']} (see {v['log']})"
    return v
