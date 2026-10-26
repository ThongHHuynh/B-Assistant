"""Voice commands handled without the LLM."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Command:
    kind: str           # stop | cancel | sleep | mute | remember | forget | recall | think | yes | no
    arg: str = ""


def _norm(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"^(hey|okay|ok|um+|uh+|so|please)[,\s]+", "", t)
    t = re.sub(r"^(bi|b|bee|jarvis)[,\s]+", "", t)
    return re.sub(r"[^\w\s']", " ", t).strip()


_STOP = {"stop", "stop it", "stop talking", "be quiet", "quiet", "shush", "shut up", "enough", "that's enough",
         "okay stop", "pause"}
_CANCEL = {"cancel", "cancel that", "never mind", "nevermind", "forget it", "nothing", "no thanks",
           "that's all", "thats all", "thank you that's all", "that is all"}
_SLEEP = {"go to sleep", "sleep", "goodbye", "bye", "good night", "goodnight", "that's all for now",
          "go to sleep now", "you can sleep", "bye bye"}
_MUTE = {"mute", "mute the mic", "mute microphone", "mute the microphone", "stop listening", "turn off the mic",
         "turn off the microphone", "disable the mic"}
_YES = {"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "go ahead", "please do", "correct",
        "confirm", "affirmative", "yes please", "sounds good", "right", "that's right", "absolutely", "of course"}
_NO = {"no", "nope", "nah", "don't", "do not", "no thanks", "negative", "cancel", "stop", "wrong", "no don't",
       "not that", "never mind"}

_REMEMBER = re.compile(r"^(?:please\s+)?(?:remember|don't forget|do not forget|note to self|keep in mind)"
                       r"(?:\s+(?:this|that))?[\s:,]+(.+)$")
_FORGET = re.compile(r"^(?:please\s+)?(?:forget|delete|erase|remove)\s+(?:the fact\s+|that\s+|about\s+|what i said about\s+)?(.+)$")
_RECALL = re.compile(r"^(?:what do you (?:know|remember) about me|what have you learned about me|"
                     r"what do you know|tell me what you know about me|what do you remember)\b")
_THINK = re.compile(r"^(?:think harder|think hard|think carefully|think deeply|use the big model)[\s,:]*(.*)$")


def parse_command(text: str) -> Command | None:
    t = _norm(text)
    if not t:
        return None
    if t in _STOP:
        return Command("stop")
    if t in _CANCEL:
        return Command("cancel")
    if t in _SLEEP:
        return Command("sleep")
    if t in _MUTE:
        return Command("mute")
    if _RECALL.match(t):
        return Command("recall")
    m = _REMEMBER.match(t)
    if m:
        # keep the user's original casing for the stored fact
        orig = re.sub(r"^.*?(?:remember|forget|note to self|keep in mind)(?:\s+(?:this|that))?[\s:,]+", "",
                      text.strip(), count=1, flags=re.I)
        return Command("remember", orig.strip() or m.group(1))
    m = _FORGET.match(t)
    if m and t not in _CANCEL:
        return Command("forget", m.group(1).strip())
    m = _THINK.match(t)
    if m:
        orig = re.sub(r"^.*?think\s+\w+[\s,:]*", "", text.strip(), count=1, flags=re.I) if m.group(1) else ""
        return Command("think", orig.strip())
    return None


def parse_yes_no(text: str) -> str | None:
    t = _norm(text)
    if t in _YES or re.match(r"^(yes|yeah|yep|sure|ok|okay)\b", t):
        return "yes"
    if t in _NO or re.match(r"^(no|nope|nah)\b", t):
        return "no"
    return None
