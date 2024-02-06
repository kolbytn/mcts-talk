from dialogue import Dialogue, DialogueTurn


def print_turn(turn: DialogueTurn) -> None:
    print("\n{}: {}".format(turn.character.name, turn.text))

if __name__ == "__main__":

    dialogue = Dialogue()
    dialogue.load("dialogue.json")

    for turn in dialogue.turns:
        print_turn(turn)

    while True:
        options = dialogue.get_pc_options()
        print("\nSelect an option by entering a number, custom text, or press enter to continue:")
        for i, option in enumerate(options):
            print("{}. {}".format(i+1, option))
        inp = input("You say: ")
        if inp.isdigit():
            inp = int(inp)
            if inp < 1 or inp > len(options):
                print("Invalid choice.")
                continue
            inp = options[inp-1]
        turn = dialogue.add_pc_turn(inp)
        print_turn(turn)
        turn = dialogue.take_npc_turn()
        print_turn(turn)
