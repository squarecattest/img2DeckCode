from pathlib import Path
import json
import warnings

filepath = Path(__file__).parent / "description.json"

try:
    with open(filepath, "r", encoding="utf-8") as file:
        description = json.load(file)
    if not isinstance(description, dict):
        raise TypeError
except (FileNotFoundError, json.JSONDecodeError, TypeError):
    warnings.showwarning(
        f"Cannot find description from {filepath}",
        Warning,
        __file__,
        9,
        line=""
    )
    description = {}

del filepath


def get_localizations(path: str) -> dict:
    '''
    Get the localization key-value pairs in ``description.json`` for the given path.

    Parameters
    ----------
    path: :class:`str`
        The jsonpath of desired localizations.
    '''
    dct = description
    keys = path.split(".")
    if not keys:
        return dct
    for key in keys:
        dct = dct.get(key, None)
        if not isinstance(dct, dict):
            raise ValueError(f"Cannot get localizations {path}")
    return dct

def get_localization_value(path: str, *, locale: str = "en-US") -> str:
    '''
    Get the specific translation in ``description.json`` for the given path.

    Parameters
    ----------
    path: :class:`str`
        The jsonpath of desired context.

    locale: :class:`Optional[str]`
        The language of the context. Default to ``en-US``.
    '''
    lc = get_localizations(path)
    if locale not in lc.keys():
        raise ValueError(f"Language {locale} is not supported in {path}")
    return str(lc[locale])