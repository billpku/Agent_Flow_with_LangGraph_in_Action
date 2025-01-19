import datetime
import json
import os
import random
import uuid
from datetime import datetime
from typing import Annotated, Dict, List, Sequence, TypedDict, Union

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver, sqlite3
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

# Load environment variables from .env file
# Please make sure to set the OPENAI_API_KEY in the .env file
load_dotenv()

# Initialize memory store
sqlite3_conn = sqlite3.connect("checkpoints.sqlite")
MEMORY = SqliteSaver(sqlite3_conn)
# Initialize LLM
LLM_MODEL_CHAT = ChatOpenAI(
    model="gpt-4o", temperature=0, api_key=os.getenv("OPENAI_API_KEY")
)

# Player choices mapping
CHOICES_MAP = {"1": "rock", "2": "paper", "3": "scissors", "4": "END"}


# Prompt template
AGENT_SYSTEM_PROMPT = """You are an AI agent playing Rock Paper Scissors. 
    Make strategic decisions based on game history but without knowing current round choices.
    Analyze patterns in player behavior to make informed choices. You can optimize your strategy by learning from previous rounds.
    To optimize your strategy, you can follow the following steps:
    1. Analyze the game history to identify patterns in player behavior.
    2. Use the identified patterns to make informed choices.
    3. Optimize your strategy by learning from previous rounds.
    Here is the game history: 
    {history}

    Then, based on the game history, what is your next move (rock/paper/scissors)?
    Your next move:
    """


class AgentChoice(BaseModel):
    """Pydantic model for agent's move choice in Rock Paper Scissors game"""

    move: Annotated[
        str,
        "The agent's chosen move, can only be one of the following: rock/paper/scissors",
    ]


# Define state management
# Define state management
class GameState(TypedDict):
    session_id: str  # UUID for each game session
    history: List[Dict]  # [{round: n, choices: {}, winner: str, timestamp: str}]
    current_round: Dict  # {player1: choice, player2: choice, agent: None}
    scores: Dict  # {player1: wins, player2: wins, agent: wins}
    is_active: bool  # True until END received


# Game configuration
VALID_MOVES = ["rock", "paper", "scissors", "END"]
WIN_CONDITIONS = {"rock": "scissors", "paper": "rock", "scissors": "paper"}


# Agent prompt template
agent_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="""You are an AI agent playing Rock Paper Scissors. 
    Make strategic decisions based on game history but without knowing current round choices.
    Analyze patterns in player behavior to make informed choices."""
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessage(
            content="Based on the game history, what is your next move (rock/paper/scissors)?"
        ),
    ]
)


def _player_1_choice_real(state: GameState) -> Union[GameState, str]:
    """Get player 1's choice from terminal input"""
    while True:
        print("\nChoose your move Player 1:")
        for key, move in CHOICES_MAP.items():
            print(f"{key}: {move}")
        choice = input("Enter choice number (or END): ").strip()

        if choice.lower() == "end":
            state["current_round"]["player1"] = "END"
            return state

        if choice in CHOICES_MAP:
            state["current_round"]["player1"] = CHOICES_MAP[choice]
            return state
        print("Invalid choice! Please select a valid number or END")


def _player_2_choice_real(state: GameState) -> Union[GameState, str]:
    """Get player 2's choice from terminal input"""
    while True:
        print("\nChoose your move Player 2:")
        for key, move in CHOICES_MAP.items():
            print(f"{key}: {move}")
        choice = input("Enter choice number (or END): ").strip()

        if choice.lower() == "end":
            state["current_round"]["player2"] = "END"
            return state

        if choice in CHOICES_MAP:
            state["current_round"]["player2"] = CHOICES_MAP[choice]
            return state
        print("Invalid choice! Please select a valid number or END")


def process_human_choices(state: GameState) -> Union[GameState, str]:
    """Process and validate human player choices"""
    # Get real player choices instead of mock choices
    state = _player_1_choice_real(state)
    state = _player_2_choice_real(state)

    # Check for END condition (if any player chooses END)
    if "END" in [state["current_round"]["player1"], state["current_round"]["player2"]]:
        state["is_active"] = False
        return state

    # Validate moves
    for player in ["player1", "player2"]:
        choice = state["current_round"][player]
        if choice not in VALID_MOVES:
            raise ValueError(f"Invalid move {choice} from {player}")

    return state


def agent_decision(state: GameState) -> Dict:
    """AI agent makes decision based on game history"""

    # Format history for agent prompt including session memory
    history_msgs = []
    for round in state["history"]:
        history_msgs.append(
            f"Round {round['round']}: {round['choices']} - Winners: {round['winners']}"
        )

    response = LLM_MODEL_CHAT.with_structured_output(AgentChoice).invoke(
        [
            SystemMessage(
                content=AGENT_SYSTEM_PROMPT.format(history="\n".join(history_msgs))
            )
        ]
    )

    # Update current round with agent's choice
    state["current_round"]["agent"] = response.move.lower().strip()

    return state


def analyze_result(state: GameState) -> GameState:
    """Determine round winner and update game state"""
    choices = state["current_round"]

    # First check if all moves are the same (Draw case 1)
    if len(set(choices.values())) == 1:
        result = {
            "round": len(state["history"]) + 1,
            "choices": choices.copy(),
            "winners": "draw",
            "timestamp": str(datetime.now()),
        }
        data = {"state": state, "result": result}
        return _update_state(data)

    # Compare each pair to count wins/losses for each player
    win_count = {"player1": 0, "player2": 0, "agent": 0}

    # Compare all pairs
    pairs = [("player1", "player2"), ("player1", "agent"), ("player2", "agent")]

    for p1, p2 in pairs:
        if WIN_CONDITIONS[choices[p1]] == choices[p2]:
            win_count[p1] += 1
        elif WIN_CONDITIONS[choices[p2]] == choices[p1]:
            win_count[p2] += 1

    # Find winners based on win counts
    max_wins = max(win_count.values())
    winners = []
    if max_wins > 0:  # If anyone won against anyone else
        winners = [player for player, wins in win_count.items() if wins == max_wins]

    # If everyone has same win count, it's a circular draw (Draw case 2)
    if len(set(win_count.values())) == 1:
        winners = "draw"

    result = {
        "round": len(state["history"]) + 1,
        "choices": choices.copy(),
        "winners": winners,
        "timestamp": str(datetime.now()),
    }

    data = {"state": state, "result": result}
    return _update_state(data)


def _update_state(data: Dict) -> GameState:
    """Update game state with round results"""
    # Add error handling for missing keys
    if not isinstance(data, dict) or "state" not in data or "result" not in data:
        raise ValueError(f"Invalid data format received: {data}")

    state = data["state"]
    result = data["result"]

    # Update history
    state["history"].append(result)

    # Check if there are winners
    if result["winners"]:
        if result["winners"] == "draw":
            print("Draw, no scores updated")
        else:
            for winner in result["winners"]:
                state["scores"][winner] += 1

    # Reset current round
    state["current_round"] = {"player1": None, "player2": None, "agent": None}

    return state


def decision_if_end(state: GameState) -> str:
    """Decision if END is received"""
    if state["is_active"]:
        return "agent_decision"
    else:
        return "END"


def create_workflow(memory: SqliteSaver):
    """Create and return the workflow graph"""
    # Build graph
    workflow = StateGraph(GameState)

    # Add nodes
    workflow.add_node("process_human_choices", process_human_choices)
    workflow.add_node("agent_decision", agent_decision)
    workflow.add_node("analyze_result", analyze_result)

    # Add edges with data transformations
    workflow.add_edge("agent_decision", "analyze_result")
    workflow.add_edge("analyze_result", "process_human_choices")

    # Add conditional edge for END
    workflow.add_conditional_edges(
        "process_human_choices",
        decision_if_end,
        {"END": END, "agent_decision": "agent_decision"},
    )

    # Set entry point
    workflow.set_entry_point("process_human_choices")

    return workflow.compile(store=memory)


def visualization_workflow(graph: StateGraph):
    """Visualize the workflow graph"""
    print(graph.get_graph().draw_mermaid())

    # Save graph to file
    # Save graph as Mermaid syntax
    with open("workflow_graph.mmd", "w") as f:
        f.write(graph.get_graph().draw_mermaid())

    try:
        # Save graph as PNG using Mermaid.Ink API
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("workflow_graph.png", "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        print(f"Failed to save PNG: {e}")


def report_results(results: List[Dict]):
    """Report the results of the game by analyzing history and showing scores for all players"""
    # Get final state which has complete history and scores
    final_state = results[-1]

    # Handle nested state structure
    if isinstance(final_state, dict) and any(
        key in final_state
        for key in ["process_human_choices", "agent_decision", "analyze_result"]
    ):
        # Get the actual state from the nested structure
        for key in ["process_human_choices", "agent_decision", "analyze_result"]:
            if key in final_state:
                final_state = final_state[key]
                break

    scores_dict = {
        "player1": final_state["scores"]["player1"],
        "player2": final_state["scores"]["player2"],
        "agent": final_state["scores"]["agent"],
    }
    print("\nGame Results:")
    print("=============")

    # Print final scores
    print("\nFinal Scores:")
    print(f"Player 1: {scores_dict['player1']}")
    print(f"Player 2: {scores_dict['player2']}")
    print(f"Agent: {scores_dict['agent']}")

    # Find the highest score
    highest_score = max(scores_dict.values())
    winners = [key for key, value in scores_dict.items() if value == highest_score]
    print(f"\nFinal Winner: {', '.join(winners)}")

    # print("\nGame History:")
    # for round_data in final_state["history"]:
    #     choices = round_data["choices"]
    #     print(f"\nRound {round_data['round']}:")
    #     print(f"Player 1: {choices['player1']}")
    #     print(f"Player 2: {choices['player2']}")
    #     print(f"Agent: {choices['agent']}")
    #     print(f"Winner: {round_data['winner']}")


if __name__ == "__main__":
    workflow = create_workflow(MEMORY)
    visualization_workflow(workflow)
    results = []

    # Create UUID for session
    thread_id = str(uuid.uuid4())
    thread = {"configurable": {"thread_id": thread_id, "recursion_limit": 100}}

    # Initialize the state with all required fields
    initial_state = {
        "session_id": thread_id,
        "history": [],
        "current_round": {"player1": None, "player2": None, "agent": None},
        "scores": {"player1": 0, "player2": 0, "agent": 0},
        "is_active": True,
    }

    for state in workflow.stream(initial_state, thread):
        results.append(state)
        # Convert state to dictionary if it's not already
        state_dict = json.loads(json.dumps(state))

        # Check if analyze_result exists in the state
        if "analyze_result" in state_dict:
            last_round = (
                state_dict["analyze_result"]["history"][-1]
                if state_dict["analyze_result"]["history"]
                else None
            )
            if last_round:
                print(f"\nRound {last_round['round']}:")
                print(f"Player 1: {last_round['choices']['player1']}")
                print(f"Player 2: {last_round['choices']['player2']}")
                print(f"Agent: {last_round['choices']['agent']}")
                print(f">>>>>>>>>>Winners: {last_round['winners']} <<<<<<<<<<")

    # Save results to file
    with open("agent_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Report results
    report_results(results)
