"""Tokenizer for HA++ source files."""

from .errors import HappError, Loc, register_source

KEYWORDS = {
    "fn", "export", "extern", "kernel", "struct", "let", "var", "const",
    "if", "else", "while", "for", "in", "return", "break", "continue",
    "as", "true", "false", "import", "shared",
}

PUNCT = sorted([
    "<<=", ">>=", "..", "->", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
    "+", "-", "*", "/", "%", "&", "|", "^", "~", "!", "<", ">", "=",
    ".", ",", ":", ";", "(", ")", "[", "]", "{", "}", "@",
], key=len, reverse=True)

DIGITS = "0123456789"


def is_ident_start(c):
    return c.isascii() and (c.isalpha() or c == "_")


def is_ident_char(c):
    return c.isascii() and (c.isalnum() or c == "_")


ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\", '"': '"', "'": "'"}


class Token:
    __slots__ = ("kind", "value", "loc")

    def __init__(self, kind, value, loc):
        self.kind = kind    # ident | int | float | str | kw | op | eof
        self.value = value
        self.loc = loc

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


def tokenize(text, path):
    register_source(path, text)
    toks = []
    i, line, col = 0, 1, 1
    n = len(text)

    def loc():
        return Loc(path, line, col)

    while i < n:
        c = text[i]
        if c == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if c in " \t\r":
            i += 1
            col += 1
            continue
        if text.startswith("//", i):
            while i < n and text[i] != "\n":
                i += 1
            continue
        if text.startswith("/*", i):
            start = loc()
            j = text.find("*/", i + 2)
            if j < 0:
                raise HappError("unterminated block comment", start)
            chunk = text[i:j + 2]
            line += chunk.count("\n")
            col = (len(chunk) - chunk.rfind("\n")) if "\n" in chunk else col + len(chunk)
            i = j + 2
            continue
        start = loc()
        if is_ident_start(c):
            j = i
            while j < n and is_ident_char(text[j]):
                j += 1
            word = text[i:j]
            toks.append(Token("kw" if word in KEYWORDS else "ident", word, start))
            col += j - i
            i = j
            continue
        if c in DIGITS:
            j = i
            is_float = False
            if text.startswith(("0x", "0X"), i):
                j = i + 2
                while j < n and (text[j] in "0123456789abcdefABCDEF_"):
                    j += 1
                raw = text[i + 2:j].replace("_", "")
                if not raw:
                    raise HappError("hex literal needs digits", start)
                val = int(raw, 16)
            elif text.startswith(("0b", "0B"), i):
                j = i + 2
                while j < n and text[j] in "01_":
                    j += 1
                raw = text[i + 2:j].replace("_", "")
                if not raw:
                    raise HappError("binary literal needs digits", start)
                val = int(raw, 2)
            else:
                while j < n and (text[j] in DIGITS or text[j] == "_"):
                    j += 1
                # '1.5' is a float, but '0..n' is a range: require a digit after '.'
                if j + 1 < n and text[j] == "." and text[j + 1] in DIGITS:
                    is_float = True
                    j += 1
                    while j < n and (text[j] in DIGITS or text[j] == "_"):
                        j += 1
                if j < n and text[j] in "eE":
                    k = j + 1
                    if k < n and text[k] in "+-":
                        k += 1
                    if k < n and text[k] in DIGITS:
                        is_float = True
                        j = k
                        while j < n and text[j] in DIGITS:
                            j += 1
                raw = text[i:j].replace("_", "")
                val = float(raw) if is_float else int(raw)
            if j < n and (text[j].isalnum() or text[j] == "_"):
                raise HappError(f"unexpected character '{text[j]}' after number",
                                Loc(path, line, col + (j - i)),
                                "use 'as' to pick a type, e.g. '1.0 as f16'")
            toks.append(Token("float" if is_float else "int", val, start))
            col += j - i
            i = j
            continue
        if c == '"':
            j = i + 1
            buf = []
            while True:
                if j >= n or text[j] == "\n":
                    raise HappError("unterminated string", start)
                ch = text[j]
                if ch == '"':
                    break
                if ch == "\\":
                    if j + 1 >= n or text[j + 1] not in ESCAPES:
                        raise HappError("unknown escape in string", Loc(path, line, col + (j - i)))
                    buf.append(ESCAPES[text[j + 1]])
                    j += 2
                    continue
                buf.append(ch)
                j += 1
            toks.append(Token("str", "".join(buf), start))
            col += j + 1 - i
            i = j + 1
            continue
        for p in PUNCT:
            if text.startswith(p, i):
                toks.append(Token("op", p, start))
                i += len(p)
                col += len(p)
                break
        else:
            raise HappError(f"unexpected character '{c}'", start)
    toks.append(Token("eof", None, loc()))
    return toks
