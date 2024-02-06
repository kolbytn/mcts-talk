from __future__ import annotations
from typing import List, Dict
import json
from copy import deepcopy

from character import Character, CharacterState
from mcts import MCTS


class DialogueTurn:
    def __init__(self, character: Character, text: str, state: Dict[Character, CharacterState]):
        self.character = character
        self.text = text
        self.state = state


class TalkingPoint:
    def __init__(self, character: Character, order: int, desc: str, texts: Dict[str, CharacterState]):
        self.character: Character = character
        self.order: int = order
        self.description: str = desc
        self.text_effects: Dict[str, CharacterState] = texts

    @property
    def targets(self) -> List[str]:
        return list(self.text_effects.keys())


class Dialogue:
    def __init__(self):
        self.pc: Character = None
        self.npcs: List[Character] = []
        self.turns: List[DialogueTurn] = []
        self.talking_points: List[TalkingPoint] = []
        self.last_options: List[str] = None
        self.max_player_options = 2
    
    @property
    def characters(self) -> List[Character]:
        return [self.pc] + self.npcs
    
    def get_character(self, name: str) -> Character:
        return [npc for npc in self.npcs if npc.name == name][0]
    
    def load(self, load_path: str) -> None:
        with open(load_path, "r") as file:
            data = json.load(file)
        
        # load player character
        self.pc = Character(data["pc"]["name"], data["pc"]["bio"])

        # load non-player characters
        for npc in data["npcs"]:
            character = Character(npc["name"], npc["bio"])
            character.set_attitude(npc["attitude"])
            character.set_relation(self.pc, npc["relation"])
            self.npcs.append(character)
        
        # load talking points
        for talkingpoint in data["talking_points"]:
            character = self.get_character(talkingpoint["character"])
            order = int(talkingpoint["order"])
            desc = talkingpoint["description"]
            texts = {
                point["text"]: CharacterState(
                    point["attitude"],
                    {self.pc: point["relation"]}
                )
                for point in talkingpoint["points"]
            }
            self.talking_points.append(TalkingPoint(character, order, desc, texts))

        # load initial text
        for turn in data["turns"]:
            character = self.get_character(turn["character"])
            text = turn["text"]
            self.turns.append(DialogueTurn(character, text, self.get_state()))

    def get_state(self) -> Dict[Character, CharacterState]:
        if not self.turns or not self.turns[-1].state:
            return deepcopy({npc: npc.state for npc in self.npcs})
        return deepcopy(self.turns[-1].state)
    
    def get_next_talking_points(self) -> List[TalkingPoint]:
        min_order = min([tp.order for tp in self.talking_points])
        return [tp for tp in self.talking_points if tp.order == min_order]
    
    def get_talking_point(self, text: str) -> TalkingPoint:
        for tp in self.talking_points:
            if text in tp.targets:
                return tp
        return None

    def add_pc_turn(self, text: str) -> None:
        if self.last_options and text not in self.last_options:
            text = self.pc.translate(self, text)
        self.turns.append(DialogueTurn(self.pc, text, self.get_state()))
        for npc in self.npcs:
            npc.update_state(npc.adjust_state(self, self.pc))
            self.turns[-1].state[npc] = deepcopy(npc.state)
        self.last_options = None
        return self.turns[-1]

    def get_pc_options(self) -> List[str]:
        if self.last_options:
            return self.last_options
        mcts = MCTS(self, is_pc=True)
        self.last_options = [text for (_ , text) in mcts.search()[:self.max_player_options]]
        return self.last_options

    def take_npc_turn(self) -> DialogueTurn:
        mcts = MCTS(self)
        character, text = mcts.search()[0]
        tp = self.get_talking_point(text)
        if tp:
            self.talking_points = [tp for tp in self.talking_points if text not in tp.targets]
            character.update_state(tp.text_effects[text])
            text = character.translate(self, text)
            state = self.get_state()
            state[character] = deepcopy(character.state)
        else:
            character.update_state(character.adjust_state(self, self.pc))
            state = self.get_state()
            state[character] = deepcopy(character.state)
        self.turns.append(DialogueTurn(character, text, state))
        self.last_options = None
        return self.turns[-1]

    def get_system_prompt(self) -> str:
        return "You are a fictional author writting a dialogue between characters."
    
    def get_character_prompt(self) -> str:
        res = ""
        for character in self.characters:
            res += "\n\nExample speech from and information about {}:\n".format(character.name)
            res += character.bio
        return res.strip()
    
    def get_talking_point_prompt(self) -> str:
        tps = self.get_next_talking_points()
        res = ""
        if len(tps) > 0:
            res += "Current dialogue objective:"
            for tp in tps:
                res += "\n" + tp.description
        return res
    
    def get_dialogue_prompt(self) -> str:
        res = "Dialogue so far:"
        last_states = {character: "" for character in self.npcs}
        for turn in self.turns:
            res += "\n{}: {}".format(turn.character.name, turn.text)
            state_info = turn.character.get_state_desc()
            if turn.character in last_states and state_info != last_states[turn.character]:
                res += " ({})".format(state_info)
                last_states[turn.character] = state_info
        return res
