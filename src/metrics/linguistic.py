from typing import Dict, List, Tuple
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
_creds_path = Path(__file__).parents[2] / "gen_ai_credential.json"
_creds = json.loads(_creds_path.read_text())
os.environ["AICORE_AUTH_URL"] = _creds["url"]
os.environ["AICORE_CLIENT_ID"] = _creds["clientid"]
os.environ["AICORE_CLIENT_SECRET"] = _creds["clientsecret"]
os.environ["AICORE_BASE_URL"] = _creds["serviceurls"]["AI_API_URL"]

from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.proxy.langchain.google_genai import init_chat_model as _google_genai
from .persuasiveness import _trim_trailing_unanimous

DIMENSIONS = ["conviction", "anchoring", "dialogic"]

_COMBINED_PROMPT = (
    "A debating agent sent the message below. Rate it on three dimensions.\n\n"
    "Conviction – how certain the agent sounds:\n"
    "1=very uncertain ('I think','maybe')  2=leans one way but with clear doubt  "
    "3=states plainly no hedging  4=confident minor qualifications  "
    "5=completely categorical ('clearly','definitely','without doubt')\n\n"
    "Anchoring – main reason for the belief:\n"
    "1=only own reasoning  2=mostly own briefly mentions a peer  "
    "3=mixes own and peer input equally  4=mostly relies on peers  "
    "5=only what other agents said ('Agent X convinced me','the consensus is')\n\n"
    "Dialogic – engages with another agent:\n"
    "1=no reference to any other agent  2=vague reference to 'others'  "
    "3=names a specific agent in passing  4=responds to a specific agent's general view  "
    "5=directly quotes or paraphrases a specific named agent's claim\n\n"
    "Message: {message}\n\n"
    "Respond with only a JSON object with integer keys conviction, anchoring, dialogic, values 1 to 5."
)


def _build_prompt(message: str) -> List:
    return [HumanMessage(content=_COMBINED_PROMPT.format(message=message[:400]))]


class LocalScorer:
    def __init__(self, model_name: str = "gemini-3.1-flash-lite", max_workers: int = 10):
        if model_name.startswith("gemini"):
            llm = init_llm(
                model_name,
                init_func=_google_genai,
                max_tokens=40,
                temperature=0,
                model_kwargs={"reasoning_effort": "none"},
            )
        else:
            llm = init_llm(
                model_name,
                max_tokens=40,
                temperature=0,
                model_kwargs={"reasoning_effort": "none"},
            )
        self._chain = llm | JsonOutputParser()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def _invoke_with_retry(self, prompt, retries: int = 10) -> List[int]:
        for attempt in range(retries):
            try:
                result = self._chain.invoke(prompt)
                return [int(result[d]) for d in DIMENSIONS]
            except Exception:
                time.sleep(2 ** attempt * 0.5 + random.uniform(0, 1))
        raise RuntimeError(f"Failed after {retries} attempts.")

    def score_many(self, messages: List[str], pbar=None) -> List[List[int]]:
        indexed = [
            (i, _build_prompt(msg))
            for i, msg in enumerate(messages) if msg.strip()
        ]
        results: List[List[int]] = [[1, 1, 1] for _ in range(len(messages))]
        futures = {self._pool.submit(self._invoke_with_retry, prompt): i for i, prompt in indexed}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            if pbar is not None:
                pbar.update(1)
        return results


def collect_prompts(rep: Dict, channel: str = "message") -> Tuple[List[Dict], List[str]]:
    traj = _trim_trailing_unanimous(rep["trajectory"])
    N = len(traj[0]["phase_b"])
    rows, messages = [], []
    for j in range(N):
        for t in range(1, len(traj)):  # skip round 0
            pb = traj[t]["phase_b"][j]
            text = pb.get(channel, "") or ""
            rows.append({"round": t, "agent_idx": j, "agent_id": pb.get("id", j)})
            messages.append(text)
    return rows, messages


def assign_scores(rows: List[Dict], vals: List[List[int]]) -> List[Dict]:
    for row, dims in zip(rows, vals):
        for dim, v in zip(DIMENSIONS, dims):
            row[dim] = v
    return rows


def score_repetition(rep: Dict, scorer: LocalScorer, channel: str = "message") -> List[Dict]:
    rows, messages = collect_prompts(rep, channel)
    vals = scorer.score_many(messages)
    return assign_scores(rows, vals)
