from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from dialogue import Dialogue
    from character import Character
import random
from tqdm import tqdm

from llm import get_response


class MCTSNode:
    def __init__(self, dialogue: str, next: List[Character], parent: MCTSNode, character: Character = None, text: str = ""):
        self.dialogue: str = dialogue
        self.next: List[Character] = next
        self.parent: MCTSNode = parent
        self.children: List[MCTSNode] = []
        self.character: Character = character
        self.text: str = text
        self.visits: int = 0
        self.reward: float = 0.0
        self.done: bool = False


class MCTS:
    def __init__(self, dialogue: Dialogue, is_pc: bool = False, max_iterations: int = 10, num_expand: int = 2,
                 pc_exand: int = 5, rollout_depth: int = 2, rollout_width: int = 1):
        self.max_iterations: int = max_iterations
        self.num_expand: int = num_expand
        self.pc_exand: int = pc_exand
        self.rollout_depth: int = rollout_depth
        self.rollout_width: int = rollout_width
        self.is_pc: bool = is_pc

        self.system_prompt: str = dialogue.get_system_prompt() + "\n\n" + dialogue.get_character_prompt()
        self.talking_point_prompt: str = dialogue.get_talking_point_prompt()
        self.pc: Character = dialogue.pc
        self.npcs: List[Character] = dialogue.npcs
        self.talking_points: List[str] = ["{}: {}".format(tp.character.name, text) for tp in dialogue.get_next_talking_points() for text in tp.targets]
        self.root = MCTSNode(
            dialogue.get_dialogue_prompt(),
            [dialogue.pc] if is_pc else dialogue.npcs,
            None
        )

    def check_talking_points(self, node: MCTSNode) -> Tuple[Character, str]:
        message = node.dialogue
        message += "\n\nIn the context of the above conversation, is the following output semantically similar to or encapsulate the target text? Output yes or no."
        message += "\n\nOutput: " + node.text
        for tp in self.talking_points:
            message + "\n\nTarget text: " + tp
            res = get_response([
                dict(role="system", content=self.system_prompt),
                dict(role="user", content=message + "\n\nTarget text: " + tp)
            ])
            if "yes" in res.lower():
                character = tp.split(":")[0].strip()
                character = [c for c in self.npcs if c.name == character][0]
                text = ":".join(tp.split(":")[1:]).strip()
                return character, text
        return None

    def search(self) -> List[Tuple[Character, str]]:
        for _ in range(self.max_iterations):
            node = self.select()
            if not node.done and (node.parent is None or node.visits > 0):
                node = self.expand(node, self.pc_exand if self.is_pc and node.parent is None else self.num_expand)
            reward = None
            if node.text and node.character in self.npcs:
                tp = self.check_talking_points(node)
                if tp and node.parent.parent is None:
                    return [tp]
                elif tp:
                    node.done = True
                    reward = 1
            if reward is None:
                reward = self.rollout(node)
            self.backpropagate(node, reward)
        sorted_children = sorted(self.root.children, key=lambda x: x.reward, reverse=True)
        return [(x.character, x.text) for x in sorted_children]

    def select(self) -> MCTSNode:
        def ucb(node: MCTSNode) -> float:
            if node.visits == 0:
                return float("inf")
            return node.reward / node.visits + 2 * (2 * node.parent.visits / node.visits) ** 0.5
        node = self.root
        while node.children and node.visits > 0:
            node = max(node.children, key=lambda x: ucb(x))
        return node

    def expand(self, parent: MCTSNode, num_expand: int) -> MCTSNode:
        for character in parent.next:
            dialogue = parent.dialogue
            if parent.character and parent.text:
                dialogue += "\n{}: {}".format(parent.character.name, parent.text)

            message = dialogue
            if character in self.npcs:
                message = self.talking_point_prompt + "\n\n" + message
            message += "\n\nContinue the conversation with a message from {}.".format(character.name)
            message += "\nUse the format \"{}: message\"".format(character.name)

            for i in range(num_expand):
                res = get_response([
                    dict(role="system", content=self.system_prompt),
                    dict(role="user", content=message)
                ])
                if i == 0:
                    message += "\n\nMake your response very disimilar from the following examples:"
                message += "\n" + res
                child = MCTSNode(
                    dialogue,
                    self.npcs if character == self.pc else [self.pc],
                    parent,
                    character=character,
                    text = res.split("{}:".format(character.name))[-1].strip()
                )
                parent.children.append(child)
        return parent.children[0]
    
    def rollout(self, node: MCTSNode) -> float:
        dialogue = node.dialogue
        if node.character and node.text:
            dialogue += "\n{}: {}".format(node.character.name, node.text)

        def add_pc_turn(dialogue):
            message = dialogue
            message += "\n\nContinue the conversation with a message from {}. Use the format: \"{}: message\"".format(self.pc.name, self.pc.name)
            res = get_response([
                dict(role="system", content=self.system_prompt),
                dict(role="user", content=message)
            ])
            dialogue += "\n" + res
            return dialogue
        
        if node.character in self.npcs:
            dialogue = add_pc_turn(dialogue)

        for i in range(self.rollout_depth):
            message = dialogue
            message += "\n\nContinue the conversation above with a new message. Use the format: \"name: message\""

            outputs = []
            for _ in range(self.rollout_width):
                res = get_response([
                    dict(role="system", content=self.system_prompt),
                    dict(role="user", content=message)
                ])
                if len(outputs) == 0:
                    message += "\n\nMake your response very disimilar from the following examples:"
                message += "\n" + res
                outputs.append(res)

            for tp in self.talking_points:
                message = dialogue
                message += "\n\nDoes the first output below do a better, worse, or equal job of continuing the above conversation than the the best of the alternative output(s)?"
                message += "\n\nOutput: " + tp
                message += "\n\nAlternative Output(s):\n" + "\n".join(outputs)
                res = get_response([
                    dict(role="system", content=self.system_prompt),
                    dict(role="user", content=message)
                ])
                if "equal" in res.lower() or "better" in res.lower():
                    return .9 ** i
            
            dialogue += "\n" + outputs[0]
            dialogue = add_pc_turn(dialogue)

        return 0

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        while node:
            node.visits += 1
            node.reward += reward
            node: MCTSNode = node.parent
