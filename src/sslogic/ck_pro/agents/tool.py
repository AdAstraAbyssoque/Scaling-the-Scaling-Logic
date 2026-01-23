#

import requests
from .utils import KwargsInitializable, rprint, GET_ENV_VAR


class Tool(KwargsInitializable):
    def __init__(self, **kwargs):
        self.name = ""
        super().__init__(**kwargs)

    def get_function_definition(self, short: bool):
        raise NotImplementedError("To be implemented")

    def __call__(self, *args, **kwargs):
        # 添加 Phoenix 追踪支持
        try:
            from .phoenix_tracer import is_phoenix_enabled, ToolTracer

            if is_phoenix_enabled():
                with ToolTracer(self.name, args, kwargs) as tracer:
                    result = self._execute(*args, **kwargs)
                    tracer.set_output(result)
                    return result
            else:
                return self._execute(*args, **kwargs)
        except ImportError:
            return self._execute(*args, **kwargs)

    def _execute(self, *args, **kwargs):
        """子类应该重写这个方法而不是 __call__"""
        raise NotImplementedError("To be implemented")


# --
# useful tools


class StopResult(dict):
    pass


class StopTool(Tool):
    def __init__(self, agent=None):
        super().__init__(name="stop")
        self.agent = agent

    def get_function_definition(self, short: bool):
        if short:
            return """- def stop(output: str, log: str) -> Dict:  # Finalize and formalize the answer when the task is complete."""
        else:
            return """- stop
```python
def stop(output: str, log: str) -> dict:
    \""" Finalize and formalize the answer when the task is complete.
    Args:
        output (str): The concise, well-formatted final answer to the task.
        log (str): Brief notes or reasoning about how the answer was determined.
    Returns:
        dict: A dictionary with the following structure:
            {
                'output': <str>  # The well-formatted answer, strictly following any specified output format.
                'log': <str>     # Additional notes, such as steps taken, issues encountered, or relevant context.
            }
    Examples:
        >>> answer = stop(output="Inter Miami", log="Task completed. The answer was found using official team sources.")
        >>> print(answer)
    \"""
```"""

    def _execute(self, output: str, log: str):
        ret = StopResult(output=output, log=log)
        if self.agent is not None:
            self.agent.put_final_result(ret)  # mark end and put final result
        return ret


class AskLLMTool(Tool):
    def __init__(self, llm=None):
        super().__init__(name="ask_llm")
        self.llm = llm

    def set_llm(self, llm):
        self.llm = llm

    def get_function_definition(self, short: bool):
        if short:
            return """- def ask_llm(query: str) -> str:  # Directly query the language model for tasks that do not require external tools."""
        else:
            return """- ask_llm
```python
def ask_llm(query: str) -> str:
    \""" Directly query the language model for tasks that do not require external tools.
    Args:
        query (str): The specific question or instruction for the LLM.
    Returns:
        str: The LLM's generated response.
    Notes:
        - Use this function for fact-based or reasoning tasks that can be answered without web search or external data.
        - Phrase the query clearly and specifically.
    Examples:
        >>> answer = ask_llm(query="What is the capital city of the USA?")
        >>> print(answer)
    \"""
```"""

    def _execute(self, query: str):
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's query with your internal knowledge. Ensure to follow the required output format if specified.",
            },
            {"role": "user", "content": query},
        ]
        response = self.llm(messages)
        return response


class SimpleSearchTool(Tool):
    def __init__(self, target="", llm=None, max_results=7, list_enum=True, **kwargs):
        super().__init__(name="simple_web_search")
        self.llm = llm
        self.max_results = max_results
        self.list_enum = list_enum
        if not target:
            target = GET_ENV_VAR(
                "SEARCH_BACKEND", df="DuckDuckGo"
            )  # use which backend search engine
        rprint(f"Setup SimpleSearchTool with {target}")
        self.target = target
        if target == "DuckDuckGo":
            self.ddgs_params = kwargs.copy()
        elif target == "Google":
            self.google_params = {
                "key": GET_ENV_VAR("SEARCH_API_KEY"),
                "cx": GET_ENV_VAR("SEARCH_CSE_ID"),
            }
        else:
            raise ValueError(f"UNK search target = {target}")
        # --

    def set_llm(self, llm):
        self.llm = llm  # might be useful for formatting?

    def get_function_definition(self, short: bool):
        if short:
            return """- def simple_web_search(query: str) -> str:  # Perform a quick web search using a search engine for straightforward information needs."""
        else:
            return """- simple_web_search
```python
def simple_web_search(query: str) -> str:
    \""" Perform a quick web search using a search engine for straightforward information needs.
    Args:
        query (str): A simple, well-phrased search term or question.
    Returns:
        str: A string containing search results, including titles, URLs, and snippets.
    Notes:
        - Use for quick lookups or when you need up-to-date information.
        - Avoid complex or multi-step queries; keep the query simple and direct.
        - Do not use for tasks requiring deep reasoning or multi-source synthesis.
    Examples:
        >>> answer = simple_web_search(query="latest iPhone")
        >>> print(answer)
    \"""
```
"""

    def _execute(self, query: str):
        target = self.target
        if target == "DuckDuckGo":
            from ddgs import DDGS

            ddgs = DDGS(**self.ddgs_params)
            rprint(f"Query ddgs with: query={query}, max_results={self.max_results}")
            results = ddgs.text(query, max_results=self.max_results)
            search_results = [
                {
                    "title": _item["title"],
                    "link": _item["href"],
                    "content": _item["body"],
                }
                for _item in results
            ]
        elif target == "Google":
            url = "https://www.googleapis.com/customsearch/v1"
            params = self.google_params.copy()
            params.update({"q": query, "num": self.max_results})
            rprint(f"Query google-search with params={params}")
            response = requests.get(url, params=params)
            results = response.json()
            search_results = [
                {
                    "title": _item["title"],
                    "link": _item["link"],
                    "content": _item["snippet"],
                }
                for _item in results.get("items", [])
            ]
        else:
            raise ValueError(f"UNK search target = {target}")
        # --
        if len(search_results) == 0:
            ret = "Search Results: No results found! Try a less restrictive/simpler query."
        elif self.list_enum:
            ret = "Search Results:\n" + "\n".join(
                [
                    f"({ii}) title={repr(vv['title'])}, link={repr(vv['link'])}, content={repr(vv['content'])}"
                    for ii, vv in enumerate(search_results)
                ]
            )
        else:
            ret = "Search Results:\n" + "\n".join(
                [
                    f"- title={repr(vv['title'])}, link={repr(vv['link'])}, content={repr(vv['content'])}"
                    for ii, vv in enumerate(search_results)
                ]
            )
        return ret


class ReadTextTool(Tool):
    def __init__(self, max_chars=10000):
        super().__init__(name="read_text")
        self.max_chars = max_chars

    def get_function_definition(self, short: bool):
        if short:
            return """- def read_text(file_path: str) -> str:  # Read the content of a text file (e.g., .txt, .json, .md, .py) from the local file system."""
        else:
            return """- read_text
```python
def read_text(file_path: str) -> str:
    \""" Read the content of a text file from the local file system.
    Args:
        file_path (str): The path to the text file to read. Can be absolute or relative path.
    Returns:
        str: The content of the file as a string. If the file is too large, only the first part will be returned.
    Notes:
        - Suitable for reading plain text files such as .txt, .json, .md, .py, .yaml, etc.
        - For complex file types (PDF, Excel, images), use file_agent instead.
        - If the file is very large, consider using file_agent for better handling.
        - The file content will be truncated if it exceeds the maximum character limit.
    Examples:
        >>> content = read_text(file_path="./output/result.json")
        >>> print(content)
        >>> config = read_text(file_path="/path/to/config.yaml")
        >>> print(config)
    \"""
```"""

    def _execute(self, file_path: str):
        import os

        try:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return f"Error: File not found at path: {file_path}"

            # 检查是否是文件
            if not os.path.isfile(file_path):
                return f"Error: Path is not a file: {file_path}"

            # 读取文件内容
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # 如果内容太长，截断并提示
            if len(content) > self.max_chars:
                content = content[: self.max_chars]
                content += f"\n\n... (Content truncated. Total file size exceeds {self.max_chars} characters. Use file_agent for full content.)"

            return content

        except UnicodeDecodeError:
            return f"Error: File is not a text file or uses unsupported encoding: {file_path}"
        except PermissionError:
            return f"Error: Permission denied when reading file: {file_path}"
        except Exception as e:
            return f"Error reading file: {str(e)}"


class SubmitAnswerTool(Tool):
    """Tool for blind review agent to submit answers and get feedback."""

    def __init__(self, official_answer=None):
        super().__init__(name="submit_answer")
        self.official_answer = official_answer
        self.submission_count = 0

    def set_official_answer(self, answer):
        """Set the official answer for comparison."""
        self.official_answer = answer

    def get_function_definition(self, short: bool):
        if short:
            return """- def submit_answer(answer: str) -> dict:  # Submit your answer and get immediate feedback."""
        else:
            return """- submit_answer
```python
def submit_answer(answer: str) -> dict:
    \""" Submit your answer and get immediate feedback.
    Args:
        answer (str): Your answer to the problem.
    Returns:
        dict: A dictionary with the following structure:
            {
                'is_correct': <bool>,  # Whether your answer is correct
                'official_answer': <str | None>,  # The official answer (only provided if your answer is incorrect)
                'message': <str>  # Feedback message
            }
    Notes:
        - You can only submit once in blind review mode.
        - If your answer is incorrect, you will receive the official answer.
        - You should then analyze the difference and reflect on your reasoning.
    Examples:
        >>> result = submit_answer(answer="42")
        >>> if result['is_correct']:
        >>>     print("Correct! Now finalize your reasoning.")
        >>> else:
        >>>     print(f"Incorrect. Official answer: {result['official_answer']}")
        >>>     print("Analyzing the difference...")
    \"""
```"""

    def _execute(self, answer: str):
        """Execute the submission and return feedback."""
        self.submission_count += 1

        if self.submission_count > 1:
            return {
                "is_correct": False,
                "official_answer": None,
                "message": "Error: You have already submitted an answer. Only one submission is allowed in blind review mode.",
            }

        if self.official_answer is None:
            return {
                "is_correct": False,
                "official_answer": None,
                "message": "Error: Official answer not set. Cannot evaluate submission.",
            }

        # Normalize answers for comparison
        submitted = str(answer).strip().lower()
        official = str(self.official_answer).strip().lower()

        is_correct = submitted == official

        if is_correct:
            return {
                "is_correct": True,
                "official_answer": None,
                "message": "✓ Correct! Your answer matches the official answer. Please finalize your reasoning and use stop() to submit your final output.",
            }
        else:
            return {
                "is_correct": False,
                "official_answer": str(self.official_answer),
                "message": f"✗ Incorrect. Your answer: '{answer}'\nOfficial answer: '{self.official_answer}'\n\nPlease analyze:\n1. Where did your reasoning go wrong?\n2. What is the correct approach?\n3. What insights can you gain from this?\n\nThen use stop() to submit your final output with reflection.",
            }
