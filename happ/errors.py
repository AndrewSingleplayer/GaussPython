"""Error reporting for the HA++ compiler.

Errors carry a source location and print the offending line with a caret, so
both people and AI assistants can see exactly what went wrong.
"""

_SOURCES = {}


def register_source(path, text):
    _SOURCES[path] = text.splitlines()


class Loc:
    __slots__ = ("file", "line", "col")

    def __init__(self, file, line, col):
        self.file = file
        self.line = line
        self.col = col

    def __str__(self):
        return f"{self.file}:{self.line}:{self.col}"


class HappError(Exception):
    def __init__(self, msg, loc=None, hint=None):
        super().__init__(msg)
        self.msg = msg
        self.loc = loc
        self.hint = hint

    def __str__(self):
        out = f"{self.loc}: error: {self.msg}" if self.loc else f"error: {self.msg}"
        if self.loc is not None:
            lines = _SOURCES.get(self.loc.file)
            if lines and 1 <= self.loc.line <= len(lines):
                src = lines[self.loc.line - 1].expandtabs(4)
                out += f"\n    {src}\n    {' ' * (self.loc.col - 1)}^"
        if self.hint:
            out += f"\n  hint: {self.hint}"
        return out
