import re
from typing import Any, Mapping


_TOKEN_RE = re.compile(r"{{\s*([a-zA-Z0-9_.-]+)\s*}}")


def _resolve_token(context: Mapping[str, Any], token: str) -> Any:
    current: Any = context
    for part in token.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return ""
    return current


def render_prompt(template: str, context: Mapping[str, Any]) -> str:
    def replacer(match: re.Match[str]) -> str:
        token = match.group(1)
        value = _resolve_token(context, token)
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value)
        return str(value)

    return _TOKEN_RE.sub(replacer, template)
