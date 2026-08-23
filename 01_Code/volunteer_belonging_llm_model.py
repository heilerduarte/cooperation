from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import urllib.request
from pathlib import Path
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class ModelParams:
    # ODD cinfuración
    n_agents: int = 30
    n_steps: int = 50
    n_runs: int = 30

    # Estado inicial
    b0: float = 0.50
    q0: float = 0.50
    g0: float = 0.50

    # 
    lambda_b: float = 0.12
    w_reward: float = 1.30
    w_pain: float = 0.80

    eta_q: float = 0.35
    rho_g: float = 0.25
    omega: float = 0.60 
    theta: float = 0.50
    alpha_a: float = 0.40  # acceptance
    alpha_r: float = 0.30  # reciprocity
    alpha_k: float = 0.15  # consistency
    alpha_u: float = 0.15  # task/social utility

    
    tau: float = 7.0
    beta_b: float = 0.35
    beta_q: float = 0.20
    beta_g: float = 0.15
    beta_pi: float = 0.20
    beta_c: float = 0.10
    value_bias: float = -0.05
    pi_help: float = 0.90
    pi_repair: float = 0.75
    pi_delay: float = 0.35
    pi_refuse: float = 0.20
    pi_withdraw: float = 0.10
    cost_help: float = 0.25
    low_collective_reputation_threshold: float = 0.35
    retention_belonging_threshold: float = 0.25
    breakdown_window: int = 10
    breakdown_help_threshold: float = 0.15
    breakdown_retention_threshold: float = 0.40


@dataclass
class ConditionConfig:
    name: str
    use_belonging: bool
    use_private_reputation: bool
    use_collective_reputation: bool
    use_llm: bool
    beta_b: float
    beta_q: float
    beta_g: float
    beta_pi: float
    beta_c: float


@dataclass
class SocialSignals:
    acceptance: float
    reciprocity: float
    consistency: float
    utility: float
    extracted_action: str
    signal_extraction_consistency: Optional[float]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def concat_nonempty_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    cleaned_frames: List[pd.DataFrame] = []

    for frame in frames:
        if frame is None or frame.empty:
            continue

        cleaned = frame.dropna(axis=1, how="all")

        if not cleaned.empty:
            cleaned_frames.append(cleaned)
    if not cleaned_frames:
        return pd.DataFrame()
    return pd.concat(cleaned_frames, ignore_index=True, sort=False)


def weighted_choice_action(p_help: float, q_to_requester: float, helper_b: float,
                           helper_g: float, rng: np.random.Generator) -> str:
    if rng.random() < p_help:
        if helper_g < 0.40 and rng.random() < 0.35:
            return "repair"
        return "help"

    if helper_b < 0.22 and rng.random() < 0.55:
        return "withdraw"
    if q_to_requester < 0.35 and rng.random() < 0.65:
        return "refuse"
    if rng.random() < 0.50:
        return "delay"

    return "refuse"

class VolunteerLLMLayer:
    def __init__(self,
                 backend: str = "template",
                 ollama_model: str = "llama3:8b",
                 ollama_host: str = "http://localhost:11434",
                 timeout_seconds: int = 45):
        self.backend = backend
        self.ollama_model = ollama_model
        self.ollama_host = ollama_host.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def generate_message(self, action: str, requester: int, helper: int) -> str:
        if self.backend == "ollama":
            msg = self._generate_message_ollama(action, requester, helper)
            if msg:
                return msg
        return self._generate_message_template(action)

    @staticmethod
    def _generate_message_template(action: str) -> str:
        templates = {
            "help": "I can help you organize the activity and welcome the new members this week.",
            "refuse": "I cannot help with this task right now. Please ask someone else.",
            "delay": "I may be able to help later, but I cannot commit at the moment.",
            "repair": "I know I did not respond last time. I can help now to make up for it.",
            "withdraw": "I will stay inactive this round and will not take any requests.",
        }
        return templates.get(action, "I cannot commit to this request right now.")

    def _generate_message_ollama(self, action: str, requester: int, helper: int) -> Optional[str]:
        prompt = (
            "You are generating one short message for a volunteer-community simulation. "
            "The formal model has already selected the action; do not change it.\n"
            f"Requester agent: {requester}\n"
            f"Helper agent: {helper}\n"
            f"Selected action: {action}\n"
            "Write one natural English sentence expressing this action. "
            "Do not include numeric values."
        )
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "").strip() or None
        except Exception:
            return None

    def extract_signals(self, message: str, intended_action: str,
                        q_before: float, utility_by_action: Dict[str, float]) -> SocialSignals:
        if self.backend == "ollama":
            extracted = self._extract_signals_ollama(message)
            if extracted is not None:
                return self._signals_from_extracted_json(
                    extracted=extracted,
                    intended_action=intended_action,
                    q_before=q_before,
                    utility_by_action=utility_by_action,
                )

        return self._extract_signals_rule_based(message, intended_action, q_before, utility_by_action)

    def _extract_signals_ollama(self, message: str) -> Optional[dict]:
        prompt = (
            "Classify this volunteer interaction message. Return only valid JSON with keys: "
            "action, acceptance, reciprocity, consistency, utility. "
            "Scores must be numbers from 0 to 1. action must be one of: "
            "help, refuse, delay, repair, withdraw.\n\n"
            f"Message: {message}"
        )
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            # JSON silicita resp. El parser inferior sigue validando todos los campos porque algunos modelos pueden devolver null.
            "format": "json",
            "options": {"temperature": 0.0},
        }
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = data.get("message", {}).get("content", "")
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if not match:
                return None
            return json.loads(match.group(0))
        except Exception:
            return None

    @staticmethod
    def _safe_score(value: object, default: float) -> float:
        if value is None:
            return clip01(default)

        try:
            if isinstance(value, str):
                cleaned = value.strip().replace(",", ".")
                if not cleaned:
                    return clip01(default)
                if cleaned.endswith("%"):
                    return clip01(float(cleaned[:-1]) / 100.0)
                value = cleaned

            numeric = float(value)
            if not np.isfinite(numeric):
                return clip01(default)
            return clip01(numeric)
        except (TypeError, ValueError):
            return clip01(default)

    @staticmethod
    def _signals_from_extracted_json(
        extracted: dict,
        intended_action: str,
        q_before: float,
        utility_by_action: Dict[str, float],
    ) -> SocialSignals:
        valid_actions = {"help", "refuse", "delay", "repair", "withdraw"}

        raw_action = extracted.get("action")

        action = (
            str(raw_action).strip().lower()
            if raw_action is not None
            else intended_action
        )
        if action not in valid_actions:
            action = intended_action if intended_action in valid_actions else "delay"

        fallback = VolunteerLLMLayer._extract_signals_rule_based(
            message="",
            intended_action=action,
            q_before=q_before,
            utility_by_action=utility_by_action,
        )

        consistency_flag = 1.0 if action == intended_action else 0.0

        return SocialSignals(
            acceptance=VolunteerLLMLayer._safe_score(
                extracted.get("acceptance"), fallback.acceptance
            ),
            reciprocity=VolunteerLLMLayer._safe_score(
                extracted.get("reciprocity"), fallback.reciprocity
            ),
            consistency=VolunteerLLMLayer._safe_score(
                extracted.get("consistency"), fallback.consistency
            ),
            utility=VolunteerLLMLayer._safe_score(
                extracted.get("utility"), fallback.utility
            ),
            extracted_action=action,
            signal_extraction_consistency=consistency_flag,
        )

    @staticmethod
    def _extract_signals_rule_based(message: str, intended_action: str,
                                    q_before: float, utility_by_action: Dict[str, float]) -> SocialSignals:
        text = message.lower()
        if any(w in text for w in ["make up", "did not respond", "apolog", "help now"]):
            action = "repair"
        elif any(w in text for w in ["can help", "i can help", "support", "organize"]):
            action = "help"
        elif any(w in text for w in ["not take", "inactive", "withdraw"]):
            action = "withdraw"
        elif any(w in text for w in ["later", "cannot commit", "may be able"]):
            action = "delay"
        elif any(w in text for w in ["cannot help", "please ask someone else", "refuse"]):
            action = "refuse"
        else:
            action = intended_action
        consistency_flag = 1.0 if action == intended_action else 0.0
        base = {
            "help": (1.00, 1.00, 1.0 - abs(1.0 - q_before), utility_by_action["help"]),
            "repair": (0.85, 0.90, 0.70, utility_by_action["repair"]),
            "delay": (0.35, 0.25, 1.0 - abs(0.30 - q_before), utility_by_action["delay"]),
            "refuse": (0.05, 0.00, 1.0 - abs(0.00 - q_before), utility_by_action["refuse"]),
            "withdraw": (0.00, 0.00, 1.0 - abs(0.00 - q_before), utility_by_action["withdraw"]),
        }
        a, r, k, u = base[action]
        return SocialSignals(
            acceptance=clip01(a),
            reciprocity=clip01(r),
            consistency=clip01(k),
            utility=clip01(u),
            extracted_action=action,
            signal_extraction_consistency=consistency_flag,
        )
def build_candidate_map(n_agents: int, local_degree: int = 4) -> Dict[int, List[int]]:
    """
    Small-world-like local candidate map: each volunteer mainly interacts with
    a small set of community members, keeping the setting decentralized.
    """
    candidate_map: Dict[int, List[int]] = {}
    half = max(1, local_degree // 2)
    for i in range(n_agents):
        candidates = []
        for d in range(1, half + 1):
            candidates.append((i - d) % n_agents)
            candidates.append((i + d) % n_agents)
        candidate_map[i] = sorted(set(c for c in candidates if c != i))
    return candidate_map


def choose_helper(requester: int, candidates: List[int], Q: np.ndarray, G: np.ndarray,
                  condition: ConditionConfig, params: ModelParams,
                  rng: np.random.Generator) -> int:
    q_vals = np.array([Q[requester, j] if condition.use_private_reputation else params.q0 for j in candidates])
    g_vals = np.array([G[j] if condition.use_collective_reputation else params.g0 for j in candidates])
    noise = rng.normal(0.0, 0.025, len(candidates))
    scores = 0.65 * q_vals + 0.35 * g_vals + noise
    max_score = np.max(scores)
    best = [candidates[k] for k, v in enumerate(scores) if np.isclose(v, max_score)]
    return int(rng.choice(best))


def compute_helping_value(helper: int, requester: int, B: np.ndarray, Q: np.ndarray, G: np.ndarray,
                          condition: ConditionConfig, params: ModelParams) -> float:
    b_term = B[helper] if condition.use_belonging else 0.0
    q_term = Q[helper, requester] if condition.use_private_reputation else params.q0
    g_term = G[requester] if condition.use_collective_reputation else params.g0
    value = (
        condition.beta_b * b_term
        + condition.beta_q * q_term
        + condition.beta_g * g_term
        + condition.beta_pi * params.pi_help
        - condition.beta_c * params.cost_help
        + params.value_bias
    )
    return float(value)


def structured_signals(action: str, q_before: float, params: ModelParams) -> SocialSignals:
    utility_by_action = {
        "help": params.pi_help,
        "repair": params.pi_repair,
        "delay": params.pi_delay,
        "refuse": params.pi_refuse,
        "withdraw": params.pi_withdraw,
    }
    base = {
        "help": (1.00, 1.00, 1.0 - abs(1.0 - q_before), utility_by_action["help"]),
        "repair": (0.85, 0.90, 0.70, utility_by_action["repair"]),
        "delay": (0.35, 0.25, 1.0 - abs(0.30 - q_before), utility_by_action["delay"]),
        "refuse": (0.05, 0.00, 1.0 - abs(0.00 - q_before), utility_by_action["refuse"]),
        "withdraw": (0.00, 0.00, 1.0 - abs(0.00 - q_before), utility_by_action["withdraw"]),
    }
    a, r, k, u = base[action]
    return SocialSignals(
        acceptance=clip01(a),
        reciprocity=clip01(r),
        consistency=clip01(k),
        utility=clip01(u),
        extracted_action=action,
        signal_extraction_consistency=None,
    )


def social_evaluation(signals: SocialSignals, params: ModelParams) -> Tuple[float, float, float]:
    E = (
        params.alpha_a * signals.acceptance
        + params.alpha_r * signals.reciprocity
        + params.alpha_k * signals.consistency
        + params.alpha_u * signals.utility
    )
    reward = max(0.0, E - params.theta)
    pain = max(0.0, params.theta - E)
    return float(E), float(reward), float(pain)


def observed_reciprocity_from_action(action: str) -> float:
    if action == "help":
        return 1.0
    if action == "repair":
        return 0.85
    if action == "delay":
        return 0.30
    return 0.0
def simulate_run(
    condition: ConditionConfig,
    params: ModelParams,
    run_id: int,
    seed: int,
    llm_layer: VolunteerLLMLayer,
    progress_path: Optional[Path] = None,
    resume_progress: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    no recuerdo 
    """
    rng = np.random.default_rng(seed)
    n = params.n_agents
    candidate_map = build_candidate_map(n)

    B = np.full(n, params.b0, dtype=float)
    Q = np.full((n, n), params.q0, dtype=float)
    np.fill_diagonal(Q, np.nan)
    G = np.full(n, params.g0, dtype=float)
    agent_records: List[Dict[str, object]] = []
    weekly_records: List[Dict[str, object]] = []
    start_step = 0

    if resume_progress and progress_path is not None and progress_path.exists():
        with progress_path.open("rb") as handle:
            saved = pickle.load(handle)

        expected_identity = {
            "condition": condition.name,
            "run_id": run_id,
            "seed": seed,
            "n_agents": params.n_agents,
            "n_steps": params.n_steps,
        }
        saved_identity = saved.get("identity", {})
        if saved_identity != expected_identity:
            raise RuntimeError(
                f"Incompatible per-cycle checkpoint: {progress_path}. "
                "Use the same experiment parameters or a new output directory."
            )

        B = np.asarray(saved["B"], dtype=float)
        Q = np.asarray(saved["Q"], dtype=float)
        G = np.asarray(saved["G"], dtype=float)
        agent_records = saved["agent_records"]
        weekly_records = saved["weekly_records"]
        start_step = int(saved["next_step"])
        rng.bit_generator.state = saved["rng_state"]

        print(
            f"Resuming incomplete run at activity cycle "
            f"{start_step + 1}/{params.n_steps}: "
            f"{condition.name}, run {run_id + 1}/{params.n_runs}"
        )

    utility_by_action = {
        "help": params.pi_help,
        "repair": params.pi_repair,
        "delay": params.pi_delay,
        "refuse": params.pi_refuse,
        "withdraw": params.pi_withdraw,
    }

    for step in range(start_step, params.n_steps):
        requests = 0
        action_counts = {"help": 0, "repair": 0, "delay": 0, "refuse": 0, "withdraw": 0}
        reward_values: List[float] = []
        pain_values: List[float] = []
        delta_b_values: List[float] = []
        help_prob_values: List[float] = []
        help_value_values: List[float] = []
        signal_consistency_values: List[float] = []
        selected_helpers: List[int] = []

        requester_order = rng.permutation(n)
        for requester in requester_order:
            request_probability = 0.35 + 0.60 * B[requester]
            if rng.random() > request_probability:
                continue

            requests += 1
            candidates = candidate_map[requester]
            helper = choose_helper(requester, candidates, Q, G, condition, params, rng)
            selected_helpers.append(helper)

            b_before = float(B[requester])
            q_before = float(Q[requester, helper]) if condition.use_private_reputation else params.q0
            g_before = float(G[helper]) if condition.use_collective_reputation else params.g0

            help_value = compute_helping_value(helper, requester, B, Q, G, condition, params)
            p_help = float(sigmoid(params.tau * (help_value - 0.5)))
            helper_q_to_requester = float(Q[helper, requester]) if condition.use_private_reputation else params.q0
            helper_g = float(G[helper]) if condition.use_collective_reputation else params.g0
            action = weighted_choice_action(p_help, helper_q_to_requester, float(B[helper]), helper_g, rng)
            action_counts[action] += 1

            if condition.use_llm:
                message = llm_layer.generate_message(action, requester, helper)
                signals = llm_layer.extract_signals(message, action, q_before, utility_by_action)
            else:
                message = ""
                signals = structured_signals(action, q_before, params)

            E, reward_signal, pain_signal = social_evaluation(signals, params)

            if condition.use_belonging:
                delta_b = params.lambda_b * (
                    params.w_reward * reward_signal - params.w_pain * pain_signal
                )
                B[requester] = clip01(B[requester] + delta_b)
            else:
                delta_b = 0.0

            # Private or public revisar.
            observed = observed_reciprocity_from_action(action)
            q_after = q_before
            if condition.use_private_reputation:
                Q[requester, helper] = clip01(Q[requester, helper] + params.eta_q * (observed - Q[requester, helper]))
                q_after = float(Q[requester, helper])


            g_after = g_before
            if condition.use_collective_reputation:
                G[helper] = clip01(G[helper] + params.rho_g * params.omega * (observed - G[helper]))
                g_after = float(G[helper])

            reward_values.append(reward_signal)
            pain_values.append(pain_signal)
            delta_b_values.append(float(B[requester] - b_before))
            help_prob_values.append(p_help)
            help_value_values.append(help_value)
            if signals.signal_extraction_consistency is not None:
                signal_consistency_values.append(float(signals.signal_extraction_consistency))

            agent_records.append({
                "run": run_id,
                "step": step,
                "condition": condition.name,
                "requester": int(requester),
                "helper": int(helper),
                "action": action,
                "use_llm": int(condition.use_llm),
                "message": message,
                "extracted_action": signals.extracted_action,
                "acceptance": signals.acceptance,
                "reciprocity": signals.reciprocity,
                "consistency": signals.consistency,
                "utility": signals.utility,
                "E": E,
                "reward_signal": reward_signal,
                "pain_signal": pain_signal,
                "B_before": b_before,
                "B_after": float(B[requester]),
                "delta_B": float(B[requester] - b_before),
                "q_before": q_before,
                "q_after": q_after,
                "G_before": g_before,
                "G_after": g_after,
                "helping_value": help_value,
                "help_probability": p_help,
                "signal_extraction_consistency": signals.signal_extraction_consistency,
            })

        total_requests = max(1, requests)
        help_like = action_counts["help"] + action_counts["repair"]
        help_rate = help_like / total_requests
        reciprocal_support_rate = (
            action_counts["help"] + 0.85 * action_counts["repair"] + 0.30 * action_counts["delay"]
        ) / total_requests

        low_g_agents = np.where(G < params.low_collective_reputation_threshold)[0].tolist()
        if low_g_agents:
            selected_set = set(selected_helpers)
            avoided = sum(1 for a in low_g_agents if a not in selected_set)
            collective_avoidance_rate = avoided / len(low_g_agents)
        else:
            collective_avoidance_rate = 0.0

        retention_rate = float(np.mean(B >= params.retention_belonging_threshold))
        withdrawal_rate = action_counts["withdraw"] / total_requests

        weekly_records.append({
            "run": run_id,
            "step": step,
            "condition": condition.name,
            "mean_belonging": float(np.mean(B)),
            "std_belonging": float(np.std(B)),
            "mean_collective_reputation": float(np.mean(G)),
            "support_requests": requests,
            "helping_rate": help_rate,
            "reciprocal_support_rate": reciprocal_support_rate,
            "repair_rate": action_counts["repair"] / total_requests,
            "delay_rate": action_counts["delay"] / total_requests,
            "refusal_rate": action_counts["refuse"] / total_requests,
            "withdrawal_rate": withdrawal_rate,
            "volunteer_retention": retention_rate,
            "collective_avoidance_rate": collective_avoidance_rate,
            "mean_reward_signal": safe_mean(reward_values),
            "mean_pain_signal": safe_mean(pain_values),
            "mean_delta_B": safe_mean(delta_b_values),
            "mean_help_probability": safe_mean(help_prob_values),
            "mean_helping_value": safe_mean(help_value_values),
            "signal_extraction_consistency": safe_mean(signal_consistency_values) if signal_consistency_values else np.nan,
        })

        if progress_path is not None:
            progress_payload = {
                "identity": {
                    "condition": condition.name,
                    "run_id": run_id,
                    "seed": seed,
                    "n_agents": params.n_agents,
                    "n_steps": params.n_steps,
                },
                "next_step": step + 1,
                "B": B,
                "Q": Q,
                "G": G,
                "rng_state": rng.bit_generator.state,
                "agent_records": agent_records,
                "weekly_records": weekly_records,
            }
            _atomic_write_pickle(progress_payload, progress_path)
            print(
                f"  Cycle checkpoint saved: {step + 1}/{params.n_steps} "
                f"({condition.name}, run {run_id + 1}/{params.n_runs})"
            )

    agent_df = pd.DataFrame(agent_records)
    weekly_df = pd.DataFrame(weekly_records)

    last_window = weekly_df.tail(params.breakdown_window)
    community_breakdown = int(
        last_window["helping_rate"].mean() < params.breakdown_help_threshold
        or last_window["volunteer_retention"].mean() < params.breakdown_retention_threshold
    )

    run_summary = {
        "run": run_id,
        "condition": condition.name,
        "final_mean_belonging": float(weekly_df["mean_belonging"].iloc[-1]),
        "final_std_belonging": float(weekly_df["std_belonging"].iloc[-1]),
        "final_mean_collective_reputation": float(weekly_df["mean_collective_reputation"].iloc[-1]),
        "mean_helping_rate": float(weekly_df["helping_rate"].mean()),
        "mean_reciprocal_support_rate": float(weekly_df["reciprocal_support_rate"].mean()),
        "mean_volunteer_retention": float(weekly_df["volunteer_retention"].mean()),
        "mean_withdrawal_rate": float(weekly_df["withdrawal_rate"].mean()),
        "mean_collective_avoidance_rate": float(weekly_df["collective_avoidance_rate"].mean()),
        "mean_signal_extraction_consistency": float(weekly_df["signal_extraction_consistency"].dropna().mean())
        if weekly_df["signal_extraction_consistency"].notna().any() else np.nan,
        "community_breakdown": community_breakdown,
    }

    return weekly_df, agent_df, run_summary


def build_conditions(params: ModelParams) -> List[ConditionConfig]:
    return [
        ConditionConfig(
            name="payoff_only",
            use_belonging=False,
            use_private_reputation=False,
            use_collective_reputation=False,
            use_llm=False,
            beta_b=0.0,
            beta_q=0.0,
            beta_g=0.0,
            beta_pi=0.55,
            beta_c=0.45,
        ),
        ConditionConfig(
            name="reputation_only",
            use_belonging=False,
            use_private_reputation=True,
            use_collective_reputation=True,
            use_llm=False,
            beta_b=0.0,
            beta_q=0.35,
            beta_g=0.25,
            beta_pi=0.25,
            beta_c=0.15,
        ),
        ConditionConfig(
            name="belonging_only",
            use_belonging=True,
            use_private_reputation=False,
            use_collective_reputation=False,
            use_llm=False,
            beta_b=0.35,
            beta_q=0.0,
            beta_g=0.0,
            beta_pi=0.30,
            beta_c=0.35,
        ),
        ConditionConfig(
            name="full_structured_model",
            use_belonging=True,
            use_private_reputation=True,
            use_collective_reputation=True,
            use_llm=False,
            beta_b=params.beta_b,
            beta_q=params.beta_q,
            beta_g=params.beta_g,
            beta_pi=params.beta_pi,
            beta_c=params.beta_c,
        ),
        ConditionConfig(
            name="full_llm_mediated_model",
            use_belonging=True,
            use_private_reputation=True,
            use_collective_reputation=True,
            use_llm=True,
            beta_b=params.beta_b,
            beta_q=params.beta_q,
            beta_g=params.beta_g,
            beta_pi=params.beta_pi,
            beta_c=params.beta_c,
        ),
    ]
def _atomic_write_csv(df: pd.DataFrame, destination: Path) -> None:
    """Write a CSV atomically so an interruption does not leave a valid-looking partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    df.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def _atomic_write_json(payload: dict, destination: Path) -> None:
    """Write JSON atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _atomic_write_pickle(payload: object, destination: Path) -> None:
    """Write a Python checkpoint atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, destination)


def _checkpoint_paths(checkpoint_dir: Path, condition_name: str, run_id: int) -> Dict[str, Path]:
    stem = f"{condition_name}__run_{run_id:03d}"
    return {
        "weekly": checkpoint_dir / f"{stem}__weekly.csv",
        "agents": checkpoint_dir / f"{stem}__agents.csv",
        "summary": checkpoint_dir / f"{stem}__summary.json",
        "progress": checkpoint_dir / f"{stem}__progress.pkl",
    }


def _checkpoint_is_complete(paths: Dict[str, Path]) -> bool:
    required = ("weekly", "agents", "summary")
    return all(
        paths[key].exists() and paths[key].stat().st_size > 0
        for key in required
    )


def _save_run_checkpoint(
    weekly_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    run_summary: Dict[str, float],
    paths: Dict[str, Path],
) -> None:
    """
    Save one completed condition/run.

    The summary file is written last and acts as the completion marker.
    """
    _atomic_write_csv(weekly_df, paths["weekly"])
    _atomic_write_csv(agent_df, paths["agents"])
    _atomic_write_json(run_summary, paths["summary"])

    if paths["progress"].exists():
        paths["progress"].unlink()


def _read_run_checkpoint(
    paths: Dict[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    weekly_df = pd.read_csv(paths["weekly"])
    agent_df = pd.read_csv(paths["agents"])
    run_summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    return weekly_df, agent_df, run_summary


def _experiment_metadata(
    params: ModelParams,
    llm_layer: VolunteerLLMLayer,
    conditions: List[ConditionConfig],
    seed_base: int,
) -> dict:
    return {
        "params": asdict(params),
        "llm_backend": llm_layer.backend,
        "ollama_model": llm_layer.ollama_model,
        "ollama_host": llm_layer.ollama_host,
        "seed_base": seed_base,
        "conditions": [condition.name for condition in conditions],
    }


def _validate_or_create_metadata(
    metadata_path: Path,
    current_metadata: dict,
    resume: bool,
) -> None:
    if metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous_metadata != current_metadata:
            raise RuntimeError(
                "The existing checkpoint metadata does not match the current parameters, backend, model, host, or conditions. Cambiar 1111"
                "--output-dir, or restore the original command before using --resume."
            )
        if not resume:
            raise RuntimeError(
                "This output directory already contains checkpoints. Use --resume to continue, Cambiar dir --output-dir for a new experiment."
            )
        return

    if resume:
        print("No previous checkpoint was found. A new resumable experiment will be started.")

    _atomic_write_json(current_metadata, metadata_path)


def run_main_experiment(
    params: ModelParams,
    output_dir: str,
    llm_layer: VolunteerLLMLayer,
    resume: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)
    output_path = Path(output_dir)
    checkpoint_dir = output_path / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seed_base = 20260707
    conditions = build_conditions(params)
    total_runs = len(conditions) * params.n_runs

    metadata = _experiment_metadata(params, llm_layer, conditions, seed_base)
    metadata_path = checkpoint_dir / "experiment_metadata.json"
    _validate_or_create_metadata(metadata_path, metadata, resume)

    completed_before_start = 0
    for condition in conditions:
        for run in range(params.n_runs):
            paths = _checkpoint_paths(checkpoint_dir, condition.name, run)
            if _checkpoint_is_complete(paths):
                completed_before_start += 1

    if completed_before_start:
        print(
            f"Checkpoint status: {completed_before_start}/{total_runs} "
            "condition-runs already completed."
        )

    completed = completed_before_start

    try:
        for cond_idx, condition in enumerate(conditions):
            for run in range(params.n_runs):
                paths = _checkpoint_paths(checkpoint_dir, condition.name, run)

                if _checkpoint_is_complete(paths):
                    print(
                        f"[{completed}/{total_runs}] Skipping completed: "
                        f"{condition.name}, run {run + 1}/{params.n_runs}"
                    )
                    continue

                seed = seed_base + cond_idx * 10000 + run
                print(
                    f"[{completed + 1}/{total_runs}] Running: "
                    f"{condition.name}, run {run + 1}/{params.n_runs}, seed={seed}"
                )

                weekly_df, agent_df, run_summary = simulate_run(
                    condition=condition,
                    params=params,
                    run_id=run,
                    seed=seed,
                    llm_layer=llm_layer,
                    progress_path=paths["progress"],
                    resume_progress=resume,
                )

                _save_run_checkpoint(
                    weekly_df=weekly_df,
                    agent_df=agent_df,
                    run_summary=run_summary,
                    paths=paths,
                )
                completed += 1
                print(
                    f"[{completed}/{total_runs}] Checkpoint saved: "
                    f"{condition.name}, run {run + 1}/{params.n_runs}"
                )

    except KeyboardInterrupt:
        print(
            "\nExecution interrupted by the user. All completed runs were saved. "
            "Restart with the same command and add --resume. "
            "Completed runs will be skipped and the incomplete run will continue "
            "from its last saved community activity cycle."
        )
        raise

    all_weekly: List[pd.DataFrame] = []
    all_agents: List[pd.DataFrame] = []
    summaries: List[Dict[str, float]] = []

    missing: List[str] = []
    for condition in conditions:
        for run in range(params.n_runs):
            paths = _checkpoint_paths(checkpoint_dir, condition.name, run)
            if not _checkpoint_is_complete(paths):
                missing.append(f"{condition.name}/run_{run:03d}")
                continue

            weekly_df, agent_df, run_summary = _read_run_checkpoint(paths)
            all_weekly.append(weekly_df)
            all_agents.append(agent_df)
            summaries.append(run_summary)

    if missing:
        raise RuntimeError(
            "The experiment did not finish. Missing checkpoints: "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )

    weekly_all = concat_nonempty_frames(all_weekly)
    agents_all = concat_nonempty_frames(all_agents)
    summary_runs = pd.DataFrame(summaries)

    summary_by_condition = (
        summary_runs.groupby("condition", as_index=False)
        .agg(
            final_mean_belonging_mean=("final_mean_belonging", "mean"),
            final_mean_belonging_std=("final_mean_belonging", "std"),
            mean_helping_rate_mean=("mean_helping_rate", "mean"),
            mean_helping_rate_std=("mean_helping_rate", "std"),
            mean_reciprocal_support_rate_mean=("mean_reciprocal_support_rate", "mean"),
            mean_reciprocal_support_rate_std=("mean_reciprocal_support_rate", "std"),
            mean_volunteer_retention_mean=("mean_volunteer_retention", "mean"),
            mean_volunteer_retention_std=("mean_volunteer_retention", "std"),
            mean_withdrawal_rate_mean=("mean_withdrawal_rate", "mean"),
            mean_collective_avoidance_rate_mean=("mean_collective_avoidance_rate", "mean"),
            community_breakdown_probability=("community_breakdown", "mean"),
            mean_signal_extraction_consistency=("mean_signal_extraction_consistency", "mean"),
        )
    )

    _atomic_write_csv(weekly_all, output_path / "volunteer_weekly_metrics.csv")
    _atomic_write_csv(agents_all, output_path / "volunteer_agent_log.csv")
    _atomic_write_csv(summary_runs, output_path / "volunteer_run_summary.csv")
    _atomic_write_csv(summary_by_condition, output_path / "volunteer_condition_summary.csv")

    make_main_plots(weekly_all, summary_runs, summary_by_condition, output_dir)

    completion_payload = {
        "completed_condition_runs": total_runs,
        "status": "complete",
    }
    _atomic_write_json(completion_payload, checkpoint_dir / "experiment_complete.json")

    return weekly_all, agents_all, summary_runs, summary_by_condition


def run_sensitivity_experiment(base_params: ModelParams, output_dir: str, llm_layer: VolunteerLLMLayer) -> pd.DataFrame:
    ensure_dir(output_dir)
    grid = {
        "theta": [0.40, 0.50, 0.60],
        "beta_b": [0.20, 0.35, 0.50, 0.65],
        "lambda_b": [0.05, 0.12, 0.20],
        "omega": [0.00, 0.25, 0.60, 1.00],
    }

    rows: List[Dict[str, float]] = []
    weekly_chunks: List[pd.DataFrame] = []
    full_condition = [c for c in build_conditions(base_params) if c.name == "full_structured_model"][0]
    seed_base = 20260721
    scenario_idx = 0

    for parameter, values in grid.items():
        for value in values:
            params = replace(base_params)
            condition = full_condition
            if parameter == "theta":
                params = replace(params, theta=value)
            elif parameter == "beta_b":
                params = replace(params, beta_b=value)
                # Keep weights interpretable by redistributing the remaining mass.
                remaining = max(0.0, 1.0 - value)
                qgc_sum = base_params.beta_q + base_params.beta_g + base_params.beta_pi + base_params.beta_c
                params = replace(
                    params,
                    beta_q=remaining * (base_params.beta_q / qgc_sum),
                    beta_g=remaining * (base_params.beta_g / qgc_sum),
                    beta_pi=remaining * (base_params.beta_pi / qgc_sum),
                    beta_c=remaining * (base_params.beta_c / qgc_sum),
                )
                condition = replace(full_condition, beta_b=params.beta_b, beta_q=params.beta_q,
                                    beta_g=params.beta_g, beta_pi=params.beta_pi, beta_c=params.beta_c)
            elif parameter == "lambda_b":
                params = replace(params, lambda_b=value)
            elif parameter == "omega":
                params = replace(params, omega=value)
            else:
                raise ValueError(parameter)

            summaries = []
            for run in range(params.n_runs):
                seed = seed_base + scenario_idx * 10000 + run
                weekly_df, _, summary = simulate_run(condition, params, run, seed, llm_layer)
                weekly_df["parameter"] = parameter
                weekly_df["value"] = value
                weekly_chunks.append(weekly_df)
                summaries.append(summary)

            s_df = pd.DataFrame(summaries)
            rows.append({
                "parameter": parameter,
                "value": value,
                "final_mean_belonging": s_df["final_mean_belonging"].mean(),
                "mean_helping_rate": s_df["mean_helping_rate"].mean(),
                "mean_reciprocal_support_rate": s_df["mean_reciprocal_support_rate"].mean(),
                "mean_volunteer_retention": s_df["mean_volunteer_retention"].mean(),
                "mean_withdrawal_rate": s_df["mean_withdrawal_rate"].mean(),
                "mean_collective_avoidance_rate": s_df["mean_collective_avoidance_rate"].mean(),
                "community_breakdown_probability": s_df["community_breakdown"].mean(),
            })
            scenario_idx += 1

    sensitivity_df = pd.DataFrame(rows)
    sensitivity_df.to_csv(os.path.join(output_dir, "volunteer_sensitivity_summary.csv"), index=False)
    if weekly_chunks:
        sensitivity_weekly = concat_nonempty_frames(weekly_chunks)
        sensitivity_weekly.to_csv(
            os.path.join(output_dir, "volunteer_sensitivity_weekly_metrics.csv"),
            index=False,
        )
    make_sensitivity_plots(sensitivity_df, output_dir)
    return sensitivity_df

def condition_order() -> List[str]:
    return [
        "payoff_only",
        "reputation_only",
        "belonging_only",
        "full_structured_model",
        "full_llm_mediated_model",
    ]


def make_main_plots(weekly_all: pd.DataFrame, summary_runs: pd.DataFrame,
                    summary_by_condition: pd.DataFrame, output_dir: str) -> None:
    # 1. evolucion teeemporal
    temporal = weekly_all.groupby(["condition", "step"], as_index=False).agg(mean_belonging=("mean_belonging", "mean"))
    plt.figure(figsize=(9, 5))
    for cond in condition_order():
        subset = temporal[temporal["condition"] == cond]
        if not subset.empty:
            plt.plot(subset["step"], subset["mean_belonging"], label=cond)
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel("Community activity cycle")
    plt.ylabel("Mean belonging")
    plt.title("Temporal evolution of belonging in the volunteer community")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "belonging_over_time_volunteer.png"), dpi=300)
    plt.close()

    # ayudar 2 
    ordered_summary = summary_by_condition.set_index("condition").reindex(condition_order()).reset_index()
    x = np.arange(len(ordered_summary))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ordered_summary["mean_helping_rate_mean"], width, label="Helping rate")
    plt.bar(x + width / 2, ordered_summary["mean_reciprocal_support_rate_mean"], width, label="Reciprocal support rate")
    plt.xticks(x, ordered_summary["condition"], rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Helping and reciprocal support rates")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "helping_and_reciprocal_support_rates.png"), dpi=300)
    plt.close()

    # 3 . Retirada
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ordered_summary["mean_volunteer_retention_mean"], width, label="Volunteer retention")
    plt.bar(x + width / 2, ordered_summary["mean_withdrawal_rate_mean"], width, label="Withdrawal rate")
    plt.xticks(x, ordered_summary["condition"], rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Volunteer retention and withdrawal")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "volunteer_retention_withdrawal.png"), dpi=300)
    plt.close()

    # 44colectiva 
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, ordered_summary["mean_collective_avoidance_rate_mean"], width, label="Collective avoidance")
    plt.bar(x + width / 2, ordered_summary["community_breakdown_probability"], width, label="Community breakdown")
    plt.xticks(x, ordered_summary["condition"], rotation=20, ha="right")
    plt.ylabel("Rate")
    plt.title("Collective avoidance and community breakdown")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "collective_reputation_avoidance.png"), dpi=300)
    plt.close()

    # 5. distrubsion normal
    plt.figure(figsize=(10, 5))
    data = [summary_runs[summary_runs["condition"] == cond]["final_mean_belonging"].values for cond in condition_order()]
    plt.boxplot(data, tick_labels=condition_order(), showmeans=True)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Final mean belonging")
    plt.title("Final belonging distribution across runs")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_belonging_distribution_volunteer.png"), dpi=300)
    plt.close()

    # 6. llm
    struct_llm = ordered_summary[ordered_summary["condition"].isin(["full_structured_model", "full_llm_mediated_model"])]
    plt.figure(figsize=(7, 5))
    plt.bar(struct_llm["condition"], struct_llm["mean_reciprocal_support_rate_mean"])
    plt.ylabel("Mean reciprocal support rate")
    plt.title("Structured vs LLM-mediated reciprocal support")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "structured_vs_llm_comparison.png"), dpi=300)
    plt.close()

    # 7. seguro
    llm_weekly = weekly_all[weekly_all["condition"] == "full_llm_mediated_model"]
    if llm_weekly["signal_extraction_consistency"].notna().any():
        cons = llm_weekly.groupby("step", as_index=False).agg(
            signal_extraction_consistency=("signal_extraction_consistency", "mean")
        )
        plt.figure(figsize=(8, 5))
        plt.plot(cons["step"], cons["signal_extraction_consistency"])
        plt.xlabel("Community activity cycle")
        plt.ylabel("Signal extraction consistency")
        plt.title("LLM-mediated signal extraction consistency")
        plt.ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "llm_signal_extraction_consistency.png"), dpi=300)
        plt.close()


def make_sensitivity_plots(sensitivity_df: pd.DataFrame, output_dir: str) -> None:
    for parameter in sensitivity_df["parameter"].unique():
        subset = sensitivity_df[sensitivity_df["parameter"] == parameter].sort_values("value")
        plt.figure(figsize=(7, 4.5))
        plt.plot(subset["value"], subset["final_mean_belonging"], marker="o")
        plt.xlabel(parameter)
        plt.ylabel("Final mean belonging")
        plt.title(f"Sensitivity of final belonging to {parameter}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sensitivity_{parameter}_final_belonging.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.plot(subset["value"], subset["mean_helping_rate"], marker="o", label="Helping")
        plt.plot(subset["value"], subset["mean_reciprocal_support_rate"], marker="o", label="Reciprocal support")
        plt.xlabel(parameter)
        plt.ylabel("Rate")
        plt.title(f"Sensitivity of helping and reciprocal support to {parameter}")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sensitivity_{parameter}_helping_reciprocal.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(7, 4.5))
        plt.plot(subset["value"], subset["community_breakdown_probability"], marker="o")
        plt.xlabel(parameter)
        plt.ylabel("Community breakdown probability")
        plt.title(f"Sensitivity of community breakdown to {parameter}")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"sensitivity_{parameter}_community_breakdown.png"), dpi=300)
        plt.close()
#cliente
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-mediated volunteer community simulation.")
    parser.add_argument("--output-dir", default="results_volunteer_llm", help="Directory for CSV and figures.")
    parser.add_argument("--n-agents", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--n-runs", type=int, default=30)
    parser.add_argument("--llm-backend", choices=["template", "ollama"], default="template")
    parser.add_argument("--ollama-model", default="llama3:8b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--run-sensitivity", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from per-run checkpoints in the selected output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = ModelParams(n_agents=args.n_agents, n_steps=args.n_steps, n_runs=args.n_runs)
    llm_layer = VolunteerLLMLayer(
        backend=args.llm_backend,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )

    print("Running main volunteer-community experiment...")
    run_main_experiment(params, args.output_dir, llm_layer, resume=args.resume)

    if args.run_sensitivity:
        print("Running sensitivity analysis...")
        run_sensitivity_experiment(params, args.output_dir, llm_layer)

    print(f"Done. Results saved in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
