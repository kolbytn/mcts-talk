from __future__ import annotations
from typing import Dict, List, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from dialogue import Dialogue
import enum

from llm import get_response


class myEnum(enum.Enum):
    @classmethod
    def get_enum(cls, value: Union[int, str, myEnum]) -> myEnum:
        return value if isinstance(value, cls) else [att for att in cls if att.name == value or att.value == value][0]


class Attitude(myEnum):
    CALM = 0
    HAPPY = 1
    SAD = 2
    ANGRY = 3
    SCARED = 4
    EXCITED = 5
    CONFUSED = 6
    DISGUSTED = 7
    SURPRISED = 8
    PLAYFUL = 9
    NERVOUS = 10


class Relation(myEnum):
    NEUTRAL = 0
    FRIENDLY = 1
    HOSTILE = 2
    ROMANTIC = 3
    FAMILIAL = 4
    PROFESSIONAL = 5


class CharacterState:
    def __init__(self, attitude: Union[int, str, Attitude] = Attitude.CALM, relations: Dict[Character, Union[int, str, Relation]] = dict()):
        self.attitude: Attitude = Attitude.get_enum(attitude) if attitude is not None else None
        self.relations: Dict[Character, Relation] = {
            char: Relation.get_enum(aff)
            for char, aff in relations.items() if aff is not None
        }


class Character:
    def __init__(self, name, bio):
        self.name: str = name
        self.bio: str = bio
        self.state = CharacterState()
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Character) and self.name == other.name
    
    @property
    def attitude(self) -> Attitude:
        return self.state.attitude

    def set_attitude(self, value: Union[int, str, Attitude]) -> None:
        if value is not None:
            self.state.attitude = Attitude.get_enum(value)

    @property
    def relations(self) -> Dict[Character, Relation]:
        return self.state.relations
    
    def get_relation(self, character: Character) -> Relation:
        return self.relations.get(character, Relation.NEUTRAL)

    def set_relation(self, character: Character, value: Union[int, str, Relation]) -> None:
        if value is not None:
            self.state.relations[character] = Relation.get_enum(value)
    
    def update_state(self, state: CharacterState) -> None:
        if state.attitude is not None:
            self.state.attitude = state.attitude
        if state.relations is not None:
            for character, relation in state.relations.items():
                self.set_relation(character, relation)

    def get_state_desc(self, relevant: List[Character] = None) -> str:
        return "{} is feeling {} and feels {}".format(
            self.name,
            self.attitude.name.lower(),
            ", ".join(
                "{} towards {}".format(
                    relation.name.lower(),
                    character.name
                )
                for character, relation in self.relations.items()
                if relevant is None or character in relevant
            )
        )
    
    def translate(self, dialogue: Dialogue, text: str) -> str:
        system_prompt = dialogue.get_system_prompt()
        system_prompt += "\n\n" + dialogue.get_character_prompt()
        message = dialogue.get_talking_point_prompt()
        message += "\n\n" + dialogue.get_dialogue_prompt()
    
        message += "\n\nConvert the following text to match the character's style and, if necessary, not contradict the previous dialogue:"
        message += "\n\n{}: {}".format(self.name, text)
        message += "\n\nUse the format \"{}: converted text\"".format(self.name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        return get_response(messages).split(self.name+":")[-1].strip()

    def adjust_state(self, dialogue: Dialogue, character: Character) -> CharacterState:
        system_prompt = dialogue.get_system_prompt()
        system_prompt += "\n\n" + dialogue.get_character_prompt()
        message = dialogue.get_talking_point_prompt()
        message += "\n\n" + dialogue.get_dialogue_prompt()

        # ask for new attitude
        message += "\n\nWhich of the following attitudes best describes how {}'s words have affected {}?".format(character.name, self.name)
        message += "\n\nAttitudes: {}".format(", ".join(att.name.lower() for att in Attitude))
        message += "\n\nUse the format \"New Attitude: attitude from list\"".format(self.name)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
        res = get_response(messages)

        # update state
        success = True
        if res.startswith("New Attitude:"):
            try:
                self.set_attitude(res.split("New Attitude:")[1].lower().strip())
            except:
                success = False
        else:
            success = False
        if not success:
            res = "New Attitude: none"
        messages.append({
            "role": "assistant",
            "content": res
        })
        
        # ask for new relations
        message = "Which of the following relationships best describe how {} now feels about {}?".format(self.name, character.name)
        message += "\n\nRelationships: {}".format(", ".join(aff.name.lower() for aff in Relation))
        message += "\n\nUse the format \"New Relationship: relationship from list\"".format(self.name)
        messages.append({
            "role": "user",
            "content": message
        })
        res = get_response(messages)

        # update state
        try:
            self.set_attitude(res.split("New Relationship:")[1].lower().strip())
        except:
            pass
        return self.state
