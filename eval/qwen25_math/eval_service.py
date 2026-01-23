import requests
import json
import math
import time
from typing import List, Dict, Optional, Union, Any


class RewardService:
    def __init__(
        self, url: str, token: str, mock_mode: bool = False, verbose: bool = False
    ):
        self.url = url
        self.token = token
        self.mock_mode = mock_mode
        self.verbose = verbose

    def extract_answer_for_r1_distill_qwen(self, model_response: str) -> str:
        if "</think>" in model_response:
            answer = model_response.split("</think>")[1].strip()
            return answer
        elif "</reasoning>" in model_response:
            answer = model_response.split("</reasoning>")[1].strip()
            return answer
        else:
            return ""

    def pack_message(self, question: Union[str, List[Dict]], answer: str) -> List[Dict]:
        if isinstance(question, str):
            return [
                {"role": "system", "content": "Reasoning: high"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
        elif isinstance(question, list):
            # Deep copy to avoid modifying original
            input_list = [dict(x) for x in question]
            if input_list[0]["role"] == "system":
                messages = input_list
            else:
                messages = [
                    {"role": "system", "content": "Reasoning: high"}
                ] + input_list

            messages.append({"role": "assistant", "content": answer})
            return messages
        else:
            raise ValueError(
                f"question must be a string or a list, but got {type(question)}"
            )

    def get_reward(
        self,
        prompt: Union[str, List[Dict]],
        response: str,
        ref_answer: str,
        exp_id: str = "cli_test",
        router_id: str = "Logic-Traditional-11",
        extra_param: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Calls the reward service API to score the response.
        """
        if extra_param is None:
            extra_param = {}

        # Mock mode handling
        if self.mock_mode:
            import random

            mock_score = random.random()
            return {
                "code": 0,
                "msg": "success",
                "data": {
                    "score": mock_score,
                    "correct": mock_score > 0.5,
                    "reward": {"final_score": mock_score, "error_code": 0},
                },
            }

        # Special handling for ARC-AGI tasks (from user snippet)
        mask_arc_agi_prompt = bool(extra_param.get("mask_arc_agi_prompt", False))
        if (
            mask_arc_agi_prompt
            and "raw_data_source" in extra_param
            and "arc_agi" in str(extra_param["raw_data_source"])
        ):
            prompt_for_pack = "This is an ARC-AGI task. The specific problem details are omitted. Please focus solely on verifying the consistency of the answer."
            if "messages" in extra_param:
                extra_param.pop("messages")
            if not isinstance(ref_answer, str):
                ref_answer = str(ref_answer)
        else:
            prompt_for_pack = prompt
            if (
                "raw_data_source" in extra_param
                and "arc_agi" in str(extra_param["raw_data_source"])
                and not isinstance(ref_answer, str)
            ):
                ref_answer = str(ref_answer)

        try:
            class1, class2, router_version = router_id.split("-")
        except ValueError:
            # Fallback if format doesn't match
            class1, class2, router_version = "Logic", "Traditional", "11"

        # Extract answer
        # Logic-Traditional-7 uses r1_distill_qwen extraction logic
        model_answer = self.extract_answer_for_r1_distill_qwen(response)

        messages = self.pack_message(prompt_for_pack, model_answer)

        # Handle extra_param JSON strings (from user snippet)
        if (
            "command_params" in extra_param
            and isinstance(extra_param["command_params"], str)
            and extra_param["command_params"] != ""
        ):
            try:
                extra_param["command_params"] = json.loads(
                    extra_param["command_params"]
                )
            except:
                pass

        if (
            "instruction_id_list" in extra_param
            and isinstance(extra_param["instruction_id_list"], str)
            and extra_param["instruction_id_list"] != ""
        ):
            try:
                extra_param["instruction_id_list"] = json.loads(
                    extra_param["instruction_id_list"]
                )
            except:
                pass

        if (
            "kwargs" in extra_param
            and isinstance(extra_param["kwargs"], str)
            and extra_param["kwargs"] != ""
        ):
            try:
                extra_param["kwargs"] = json.loads(extra_param["kwargs"])
            except:
                pass

        if (
            "messages" in extra_param
            and isinstance(extra_param["messages"], str)
            and extra_param["messages"] != ""
        ):
            try:
                loaded_messages = json.loads(extra_param["messages"])
                if loaded_messages[-1]["role"] == "user":
                    loaded_messages.append(
                        {"role": "assistant", "content": model_answer}
                    )
                    messages = loaded_messages
            except:
                pass

        data = {
            "exp_id": exp_id,
            "messages": messages,
            "ref_answer": ref_answer,
            "class1": class1,
            "class2": class2,
            "router_version": router_version,
            "extra": extra_param,
        }

        headers = {"token": self.token, "Content-Type": "application/json"}

        # Retry logic (from user snippet)
        retry_count = 0
        max_retries = 3
        reward = -100
        result_json = None

        total_attempts = max_retries + 1
        while retry_count <= max_retries:
            try:
                res = requests.post(
                    self.url, data=json.dumps(data), headers=headers, timeout=1800
                )
                result_json = res.json()

                reward_data = result_json.get("data", {}).get("reward", {})
                error_code = reward_data.get("error_code", 1)
                final_score = reward_data.get("final_score", -100)

                if (
                    final_score > -50
                    and error_code == 0
                    and isinstance(final_score, (float, int))
                ):
                    reward = final_score
                    break  # Success
                else:
                    if self.verbose:
                        print(
                            f"[RewardService] retry {retry_count + 1}/{total_attempts} "
                            f"invalid response: {result_json.get('data', {})}"
                        )
                    reward = -100
            except Exception as e:
                if self.verbose:
                    if isinstance(e, requests.exceptions.Timeout):
                        print(
                            f"[RewardService] timeout retry {retry_count + 1}/{total_attempts}: {e}"
                        )
                    else:
                        print(
                            f"[RewardService] retry {retry_count + 1}/{total_attempts} exception: {e}"
                        )
                reward = -100

            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(1)  # Wait a bit before retry

        if reward == -100:
            if self.verbose:
                print(
                    f"[RewardService] failed after {total_attempts} attempts for router_id: {router_id}"
                )
            return None

        # Return in a format compatible with run_eval.py expectations
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "score": reward,
                "correct": reward > 0.5,  # Assuming > 0.5 is correct, adjust if needed
                "reward": {"final_score": reward, "error_code": 0},
            },
        }

    # Alias for compatibility if needed
    def parse_response(self, response: str) -> str:
        return self.extract_answer_for_r1_distill_qwen(response)


def calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k using the unbiased estimator.
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    prob_all_incorrect = 1.0
    for i in range(k):
        numerator = n - c - i
        denominator = n - i
        if numerator < 0:
            prob_all_incorrect = 0.0
            break
        prob_all_incorrect *= numerator / denominator
    return 1.0 - prob_all_incorrect


def calculate_majority_vote(responses: List[str], scores: List[bool]) -> float:
    """
    Calculate Majority Vote accuracy.

    Args:
        responses: List of extracted answer strings.
        scores: List of booleans indicating if the corresponding response is correct.

    Returns:
        1.0 if the majority answer is correct, 0.0 otherwise.
    """
    if not responses:
        return 0.0

    from collections import Counter

    # Count frequencies of answer strings
    # We assume responses are already extracted and normalized (stripped)
    counts = Counter(responses)

    # Find the most common answer(s)
    # most_common returns a list of (element, count)
    most_common = counts.most_common()

    if not most_common:
        return 0.0

    # Get the top answer string
    top_answer = most_common[0][0]

    # Check if this answer is correct
    # We look for the first instance of this answer in the responses list
    # and check its corresponding score.
    # (Assuming consistency: if an answer string is correct once, it's correct always.
    # If the reward model is noisy, this might vary, but we take the first judgment or average.)

    # Let's check if ANY instance of this answer was marked correct.
    # Or better, check the consistency.

    # For simplicity and robustness against noisy reward models:
    # We consider the majority answer "Correct" if the majority of its instances are correct?
    # Or just if it matches the reference? Here we only have 'scores' from reward model.

    # Let's find the index of the top answer
    try:
        idx = responses.index(top_answer)
        return 1.0 if scores[idx] else 0.0
    except ValueError:
        return 0.0


class ModelConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type

    def get_prompt_suffix(self) -> str:
        if self.model_type == "base":
            return "请逐步推理，并将思考过程包含在 <reasoning> 和 </reasoning> 标签中。最后给出简要的答案和解释。"
        return ""
