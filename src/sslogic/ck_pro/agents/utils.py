#

import os
import time
import random
import re
import sys
import json
import types
import contextlib
from typing import Union, Callable
from functools import partial
import signal
import numpy as np

from rich.console import Console as rich_console
from rich.text import Text

# rprint
_console = rich_console(
    force_terminal=(False if os.getenv("NO_FORCE_TERMINAL", False) else True)
)


def rprint(inputs, style=None, timed=False):
    if isinstance(inputs, str):
        inputs = [inputs]  # with style as the default

    text = Text()
    for one_item in inputs:
        if isinstance(one_item, str):
            one_item = (one_item, None)
        one_str, one_style = one_item  # pairs
        one_style = style if one_style is None else one_style
        text.append(str(one_str), style=one_style)

    if timed:
        timestamp = Text(f"[{time.ctime()}] ")
        timestamp.append_text(text)
        text = timestamp

    _console.print(text)


# --
# simple adpators
zlog = rprint
zwarn = lambda x: rprint(x, style="white on red")
# --


class MyJsonEncoder(json.JSONEncoder):
    def default(self, one: object):
        if hasattr(one, "to_dict"):
            return one.to_dict()
        else:
            try:
                return json.JSONEncoder.default(self, one)
            except:  # note: simply return a str for whatever the python code may return
                zwarn(
                    f"WARNING: MyJsonEncoder cannot encode the object of {type(one)}, simply put its str: {str(one)}"
                )
                return str(one)


def my_json_dumps(*args, **kwargs):
    return json.dumps(*args, **kwargs, cls=MyJsonEncoder)


def tuple_keys_to_str(d):
    if isinstance(d, dict):
        return {str(k): tuple_keys_to_str(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [tuple_keys_to_str(i) for i in d]
    else:
        return d


# wrapping a function and try it multiple times
def wrapped_trying(
    func, default_return=None, max_times=10, wait_error_names=(), reraise=False
):
    # --
    if max_times < 0:
        return func()  # directly no wrap (useful for debugging)!
    # --
    remaining_tryings = max_times
    ret = default_return
    while True:
        try:
            ret = func()
            break  # remember to jump out!!!
        except Exception as e:
            # 转义 rich 特殊字符，避免解析错误
            error_msg = str(e).replace("[", "\\[").replace("]", "\\]")
            rprint(f"Retry with Error: {error_msg}", style="white on red")
            rand = random.randint(1, 5)
            time.sleep(rand)
            if type(e).__name__ in wait_error_names:
                continue  # simply wait it
            else:
                remaining_tryings -= 1
                if remaining_tryings <= 0:
                    if reraise:
                        raise e
                    else:
                        break
    return ret


# get env variable until hitting a key or returning the default value
def GET_ENV_VAR(*keys: str, df=None):
    for k in keys:
        if k in os.environ:
            return os.getenv(k)
    return df


# get until hit
def get_until_hit(d, keys, df=None):
    for k in keys:
        if k in d:
            return d[k]
    return df


# easier init with kwargs
class KwargsInitializable:
    def __init__(self, _assert_existing=True, _default_init=False, **kwargs):
        updates = {}
        new_updates = {}
        for k, v in kwargs.items():
            if _assert_existing:
                assert hasattr(self, k), f"Attr {k} not existing!"
            v0 = getattr(self, k, None)
            if v0 is not None and isinstance(v0, KwargsInitializable):
                new_val = type(v0)(**v)  # further make a new one!
                updates[k] = f"__new__ {type(new_val)}"
            elif v0 is None:  # simply directly update
                new_val = v
                new_updates[k] = new_val
            else:
                new_val = type(v0)(v)  # conversion
                updates[k] = new_val
            setattr(self, k, new_val)
        if not _default_init:
            rprint(f"Finish init {self}, updates={updates}, new_updates={new_updates}")


# --
# templated string (also allowing conditional prompts)
class TemplatedString:
    def __init__(self, s: Union[str, Callable]):
        self.str = s

    def format(self, **kwargs):
        if isinstance(self.str, str):
            return TemplatedString.eval_fstring(self.str, **kwargs)
        else:  # direct call it!
            return self.str(**kwargs)

    @staticmethod
    def eval_fstring(s: str, _globals=None, _locals=None, **kwargs):
        if _locals is None:
            _inner_locals = {}
        else:
            _inner_locals = _locals.copy()
        _inner_locals.update(kwargs)
        assert '"""' not in s, "Special seq not allowed!"
        ret = eval('f"""' + s + '"""', _globals, _inner_locals)
        return ret


# a simple wrapper class for with expression
class WithWrapper:
    def __init__(self, f_start: Callable = None, f_end: Callable = None, item=None):
        self.f_start = f_start
        self.f_end = f_end
        self.item: object = item

    def __enter__(self):
        if self.f_start is not None:
            self.f_start()
        if self.item is not None and hasattr(self.item, "__enter__"):
            self.item.__enter__()
        # return self if self.item is None else self.item
        return self.item

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.item is not None and hasattr(self.item, "__exit__"):
            self.item.__exit__()
        if self.f_end is not None:
            self.f_end()


def my_open_with(fd_or_path, mode="r", empty_std=False, **kwargs):
    if empty_std and fd_or_path == "":
        fd_or_path = sys.stdout if ("w" in mode) else sys.stdin
    if isinstance(fd_or_path, str) and fd_or_path:
        return open(fd_or_path, mode=mode, **kwargs)
    else:
        # assert isinstance(fd_or_path, IO)
        return WithWrapper(None, None, fd_or_path)


# get unique ID
def get_unique_id(prefix=""):
    import datetime
    import threading

    dt = datetime.datetime.now().isoformat()
    ret = f"{prefix}{dt}_P{os.getpid()}_T{threading.get_native_id()}"  # PID+TID
    return ret


# update dict (in an incremental way)
def incr_update_dict(trg, src_dict):
    for name, value in src_dict.items():
        path = name.split(".")
        curr = trg
        for _piece in path[:-1]:
            if _piece not in curr:  # create one if not existing
                curr[_piece] = {}
            curr = curr[_piece]
        _piece = path[-1]
        if _piece in curr and curr[_piece] is not None:
            assigning_value = type(curr[_piece])(value)  # value to assign
            if isinstance(assigning_value, dict) and isinstance(curr[_piece], dict):
                incr_update_dict(curr[_piece], assigning_value)  # further do incr
            else:
                curr[_piece] = assigning_value  # with type conversion
        else:
            curr[_piece] = value  # directly assign!


# --
# common response format; note: let each agent specify their own ...
# RESPONSE_FORMAT_REQUIREMENT = """## Output
# Please generate your response, your reply should strictly follow the format:
# Thought: {First, explain your reasoning for your outputs in one line.}
# Code: {Then, output your python code blob.}
# """


# parse specific formats
def parse_response(s: str, seps: list, strip=True, return_dict=False):
    assert len(seps) == len(set(seps)), f"Repeated items in seps: {seps}"
    ret = []
    remaining_s = s
    # parse them one by one
    for one_sep_idx, one_sep in enumerate(seps):
        try:
            p1, p2 = remaining_s.split(one_sep, 1)
            if p1.strip():
                rprint(f"Get an unexpected piece: {p1}")
            sep_val = p2
            for one_sep2 in seps[one_sep_idx + 1 :]:
                if one_sep2 in p2:
                    sep_val = p2.split(one_sep2, 1)[0]
                    break  # finding one is enough!
            assert p2.startswith(sep_val), "Internal error for unmatched prefix??"
            remaining_s = p2[len(sep_val) :]
            one_val = sep_val
        except:  # by default None
            one_val = None
        ret.append(one_val)
    # --
    if strip:
        if isinstance(strip, str):
            ret = [(z.strip(strip) if isinstance(z, str) else z) for z in ret]
        else:
            ret = [(z.strip() if isinstance(z, str) else z) for z in ret]
    if return_dict:
        ret = {k: v for k, v in zip(seps, ret)}
    return ret


class CodeExecutor:
    def __init__(self, global_dict=None):
        # self.code = code
        self.results = []
        self.globals = global_dict if global_dict else {}
        # self.additional_imports = None
        self.internal_functions = {
            "print": self.custom_print,
            "input": CodeExecutor.custom_input,
            "exit": CodeExecutor.custom_exit,
            "ask_llm": self._disabled_tool_factory("ask_llm"),
            "web_agent": self._disabled_tool_factory("web_agent"),
            "file_agent": self._disabled_tool_factory("file_agent"),
        }  # customized ones
        self.null_stdin = not bool(
            int(GET_ENV_VAR("NO_NULL_STDIN", df="0"))
        )  # for easier debugging and program interacting

    def add_global_vars(self, **kwargs):
        self.globals.update(kwargs)

    @staticmethod
    def extract_code(s: str):
        # CODE_PATTERN = r"```(?:py[^t]|python)(.*?)```"
        CODE_PATTERN = r"```(?:py[^t]|python)(.*)```"  # get more codes
        orig_s, hit_code = s, False
        # strip _CODE_PREFIX
        _CODE_PREFIX = "<|python_tag|>"
        if _CODE_PREFIX in s:  # strip _CODE_PREFIX
            hit_code = True
            _idx = s.index(_CODE_PREFIX)
            s = s[_idx + len(_CODE_PREFIX) :].lstrip()  # strip tag
        # strip all ```python ... ``` pieces
        # m = re.search(r"```python(.*)```", s, flags=re.DOTALL)
        if "```" in s:
            hit_code = True
            all_pieces = []
            for piece in re.findall(CODE_PATTERN, s, flags=re.DOTALL):
                all_pieces.append(piece.strip())
            s = "\n".join(all_pieces)
        # --
        # cleaning
        while s.endswith("```"):  # a simple fix
            s = s[:-3].strip()
        ret = s if hit_code else ""
        return ret

    def custom_print(self, *args):
        # output = " ".join(str(arg) for arg in args)
        # results.append(output)
        self.results.extend(args)  # note: simply adding!

    @staticmethod
    def custom_input(*args):
        return "No input available."

    @staticmethod
    def custom_exit(*args):
        return "Cannot exit."

    @staticmethod
    def _disabled_tool_factory(name: str):
        def _disabled_tool(*args, **kwargs):
            raise RuntimeError(
                f"工具 `{name}` 在当前流程中已被禁用，如需使用请改写方案或联系管控人员。"
            )

        return _disabled_tool

    def get_print_results(self, return_str=False, clear=True):
        ret = self.results.copy()  # a list of results
        if clear:
            self.results.clear()
        if len(ret) == 1:
            ret = ret[0]  # if there is only one output
        if return_str:
            ret = "\n".join(ret)
        return ret

    def _exec(self, code, null_stdin, timeout):
        import threading

        original_stdin = sys.stdin  # original stdin

        # 检查是否在主线程中，signal 只能在主线程使用
        is_main_thread = threading.current_thread() is threading.main_thread()

        if timeout > 0 and is_main_thread:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        try:
            with open(os.devnull, "r") as fd:
                if null_stdin:  # change stdin
                    sys.stdin = fd
                exec(code, self.globals)  # note: no locals since things can be strange!
        finally:
            if null_stdin:  # change stdin
                sys.stdin = original_stdin
            if timeout > 0 and is_main_thread:
                signal.alarm(0)  # Disable the alarm
            # simply remove global vars to avoid pickle errors for multiprocessing running!
            # self.globals.clear()  # note: simply create a new executor for each run!

    def run(self, code, catch_exception=True, null_stdin=None, timeout=0):
        if null_stdin is None:
            null_stdin = self.null_stdin  # use the default one
        # --
        if code:  # some simple modifications
            # Fix common formatting issues from LLM output - replace Chinese punctuation with English equivalents
            # This prevents syntax errors when LLM accidentally uses Chinese punctuation in code
            chinese_punct_replacements = [
                ("，", ","),  # Chinese comma
                ("：", ":"),  # Chinese colon
                ("；", ";"),  # Chinese semicolon
                ("（", "("),  # Chinese left paren
                ("）", ")"),  # Chinese right paren
                ("【", "["),  # Chinese left bracket
                ("】", "]"),  # Chinese right bracket
                ("｛", "{"),  # Full-width left brace
                ("｝", "}"),  # Full-width right brace
                ('"', '"'),  # Left double quote
                ('"', '"'),  # Right double quote
                (
                    """, "'"),    # Left single quote
                (""",
                    "'",
                ),  # Right single quote
                # Common math symbols that may appear in Chinese context
                ("≥", ">="),  # Greater than or equal to
                ("≤", "<="),  # Less than or equal to
                ("×", "*"),  # Multiplication sign
                ("÷", "/"),  # Division sign
                ("≠", "!="),  # Not equal to
                ("≈", "=="),  # Approximately equal (convert to equal for code)
            ]
            has_chinese_punct = False
            for old_char, new_char in chinese_punct_replacements:
                if old_char in code:
                    has_chinese_punct = True
                    code = code.replace(old_char, new_char)
            if has_chinese_punct:
                zwarn(
                    f"Auto-fixed Chinese punctuation and special characters in code to prevent syntax errors"
                )

            # Additional cleanup: remove or escape problematic Chinese characters in code
            # This handles cases where Chinese text appears outside of strings
            lines = code.split("\n")
            fixed_lines = []
            for line in lines:
                # If the line appears to be pure Chinese text without proper string delimiters,
                # convert it to a comment
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    # Check if line contains Chinese characters but isn't in a string
                    # Simple heuristic: if there are Chinese chars and no quotes wrapping them properly
                    has_chinese = bool(re.search(r"[\u4e00-\u9fff]", line))
                    if has_chinese:
                        # Check if it's likely problematic (Chinese text not in strings)
                        # Look for patterns like "- Result X: 中文内容" which are clearly not valid Python
                        if re.match(
                            r"^[\s\-]*(?:Result|结果|观察|Observation).*[:：]", stripped
                        ):
                            # This looks like output text, comment it out
                            line = "# " + line if not line.startswith("#") else line
                            has_chinese_punct = True
                fixed_lines.append(line)

            if has_chinese_punct and fixed_lines != lines:
                code = "\n".join(fixed_lines)
                zwarn(
                    f"Auto-commented problematic lines with Chinese text to prevent syntax errors"
                )

            code_nopes = []
            code_lines = [f"import {lib}\n" for lib in ["os", "sys"]] + ["", ""]
            for one_line in code.split("\n"):
                if any(
                    re.match(r"from\s*.*\s*import\s*" + function_name, one_line.strip())
                    for function_name in self.globals.keys()
                ):  # no need of such imports
                    code_nopes.append(one_line)
                else:
                    code_lines.append(one_line)
            code = "\n".join(code_lines)
            if code_nopes:
                zwarn(f"Remove unneeded lines of {code_nopes}")
        self.globals.update(self.internal_functions)  # add internal functions
        # --
        if catch_exception:
            try:
                self._exec(code, null_stdin, timeout)
            except Exception as e:
                err = self.format_error(code)
                # self.results.append(err)
                if self.results:
                    err = (
                        f"{err.strip()}\n(* Partial Results={self.get_print_results()})"
                    )
                if isinstance(e, TimeoutError):
                    err = f"{err}\n-> Please revise your code and simplify the next step to control the runtime."
                self.custom_print(err)  # put err
                zwarn(f"Error executing code: {e}")
        else:
            self._exec(code, null_stdin, timeout)
        # --

    @staticmethod
    def format_error(code: str):
        import traceback

        err = traceback.format_exc()
        _err_line = None
        _line_num = None
        for _line in reversed(err.split("\n")):
            ps = re.findall(r"line (\d+),", _line)
            if ps:
                _err_line, _line_num = _line, ps[0]
                break
        # print(_line_num, code.split('\n'))
        try:
            _line_str = code.split("\n")[int(_line_num) - 1]
            err = err.replace(_err_line, f"{_err_line}\n    {_line_str.strip()}")
        except:  # if we cannot get the line
            pass
        return f"Code Execution Error:\n{err}"


def timeout_handler(signum, frame):
    raise TimeoutError("Code execution exceeded timeout")


def get_np_generator(seed):
    return np.random.RandomState(seed)


# there are images in the messages
def have_images_in_messages(messages):
    for message in messages:
        contents = message.get("content", "")
        if not isinstance(contents, list):
            contents = [contents]
        for one_content in contents:
            if isinstance(one_content, dict):
                if one_content.get("type") == "image_url":
                    return True
    return False
