import re

# system message for each mode
SYSTEM = {
    "patch": ("You edit CadQuery (Python CAD) programs. Given a program and an edit instruction, reply with only a "
              "unified diff (--- a/part.py, +++ b/part.py, then @@ hunks) that makes the edit. Change as few lines as possible."),
    "rewrite": ("You edit CadQuery (Python CAD) programs. Given a program and an edit instruction, reply with only the complete edited program."),
}

# user message: the instruction, then the program in a python code block
def user_message(instruction, program):
    return f"Instruction: {instruction}\n\nProgram:\n```python\n{program}```"

# chat messages [system, user] (+ assistant = the target reply, for training)
def messages(mode, instruction, program, target=None):
    msgs = [{"role": "system", "content": SYSTEM[mode]},
            {"role": "user", "content": user_message(instruction, program)}]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs

# the zero-shot model usually wraps its reply in ```...```, take what is inside
def strip_fence(text):
    m = re.search(r"```[a-zA-Z]*\n(.*?)```", text, re.S)
    return m.group(1) if m else text
