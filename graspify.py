import os
import ast
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def call_gpt(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def filter_objects_by_grasp(objects, grasp_type):
    """
    Returns a list of objects that are NOT suitable for the given grasp_type.
    """
    definitions = (
        "Definitions:\n"
        "1) Power grasp: Usually for larger or heavier objects where fingers wrap fully around.\n"
        "2) Precision grasp: Usually for smaller or lighter objects where fingertips are used.\n\n"
    )

    prompt = (
        f"You are a robotic grasping expert. Here is a list of objects:\n{objects}\n\n"
        f"{definitions}"
        f"Identify which of these objects are NOT suitable for a '{grasp_type}' "
        f"based on the definitions above and common robotic manipulation techniques.\n\n"
        f"Return ONLY the unsuitable items as a valid Python list, like:\n['item1', 'item2']\n"
        f"Return only the list. No extra explanation or formatting."
    )
    result = call_gpt(prompt)
    print(f"\n🔧 GPT raw output for excluded objects:\n{result}\n")
    try:
        excluded = ast.literal_eval(result)
        if not isinstance(excluded, list):
            raise ValueError("Parsed result is not a list.")
        return excluded
    except Exception as e:
        print(f"⚠️ Failed to parse excluded objects. Error: {e}")
        return []

def create_categorization_question(objects):
    """
    Asks GPT to split the objects into 3 categories (A, B, C),
    and show a 4th line 'D) Other' (with no bracketed items).

    We want the format like:
      Question: Which group of objects are you thinking about?
      A) CategoryName: [items]
      B) CategoryName: [items]
      C) CategoryName: [items]
      D) Other

    We do NOT want leftover items shown on D).
    """
    prompt = (
        f"Split the following objects into exactly 3 non-overlapping categories, "
        f"each with at least 1 item. Do not create any empty category. If something does not fit, place it in some category that makes sense."
        f"Name each category. However, do NOT list any leftover items."
        f"In other words, place *all* given objects into one of the 3 categories. "
        f"Then provide a 'D) Other' line with no brackets or items, just the word 'Other'.\n\n"
        f"Format exactly:\n"
        f"Question: Which group of objects are you thinking about?\n"
        f"A) CategoryName: [list, of, items]\n"
        f"B) CategoryName: [list, of, items]\n"
        f"C) CategoryName: [list, of, items]\n"
        f"D) Other\n\n"
        f"Return ONLY this question in the specified format (no extra commentary).\n\n"
        f"Objects:\n{objects}"
    )
    return call_gpt(prompt)

def robust_parse_bracketed_list(bracketed_text: str) -> list:
    """
    Given a string that is supposed to look like '[hammer, screwdriver]'
    or '["hammer", "screwdriver"]', return a Python list of strings.

    Steps:
    1) Strip special whitespace from ends.
    2) Confirm it starts with '[' and ends with ']'.
    3) Try ast.literal_eval(...) first:
       - If it succeeds, return the resulting list (assuming it's a list).
    4) Otherwise, do a manual parse by splitting on commas inside the brackets.
    5) Return a list of trimmed items, e.g. ['hammer', 'screwdriver'].

    If anything goes wrong, returns an empty list.
    """
    text = bracketed_text.strip(" \t\n\r\u200b\u200c\u200d\ufeff")
    if not (text.startswith("[") and text.endswith("]")):
        return []

    # First, try literal_eval for correct Python syntax:
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            # Convert each item to str in case they're not strings
            return [str(item).strip() for item in parsed]
        else:
            return []
    except:
        # Fallback: manual parse
        inside = text[1:-1].strip()  # remove leading '[' and trailing ']'
        if not inside:
            return []
        raw_parts = inside.split(',')
        items = [part.strip() for part in raw_parts if part.strip()]
        return items

def parse_three_categories(question_text: str):
    """
    Given the GPT-generated question text (with lines for A, B, C, plus 'D) Other'),
    parse out the lists for A, B, and C. Return a dict:
      {
        'A': [...],
        'B': [...],
        'C': [...]
      }

    'D) Other' is intentionally not parsed, since it has no bracketed items.
    """
    # Regex to find lines like:
    #   A) Something: [list, of, items]
    #   B) Something: [list, of, items]
    #   C) Something: [list, of, items]
    pattern = r'^([ABC])\)\s.*:\s(\[.*\])'
    lines = question_text.splitlines()
    categories = {'A': [], 'B': [], 'C': []}

    for line in lines:
        clean_line = line.strip(" \t\n\r\u200b\u200c\u200d\ufeff")
        match = re.match(pattern, clean_line)
        if match:
            letter = match.group(1)  # 'A', 'B', or 'C'
            bracketed_part = match.group(2)
            items = robust_parse_bracketed_list(bracketed_part)
            categories[letter] = items

    return categories

def narrow_down_interactive(current_objects, other_objects):
    """
    We categorize 'current_objects' into A/B/C plus "D) Other".
    If user chooses A/B/C, we get that subset -> keep exploring.
    If user chooses D) Other, we flip sets: now we categorize 'other_objects'.
    This recurses until:
      - we get down to 1 or 2 items from A/B/C choice
      - user chooses to stop
    We never directly cut off when user picks 'Other'.
    """
    # If we have 0 or 1 or 2 items in current_objects, handle them directly:
    if len(current_objects) == 0:
        print("\nNo objects left in this set. Perhaps go back to 'Other' next time.")
        return
    elif len(current_objects) == 1:
        # Direct final
        print(f"\n✅ Only one object remains: {current_objects[0]}")
        return
    elif len(current_objects) == 2:
        # Let user pick
        print(f"\nWe have two possible objects left: {current_objects}")
        choice = None
        while choice not in ["1", "2"]:
            choice = input("Which one is it? Enter 1 or 2: ").strip()
        final_obj = current_objects[0] if choice == "1" else current_objects[1]
        print(f"✅ Final object: {final_obj}")
        return

    # Otherwise, we ask GPT to categorize the current set
    question_text = create_categorization_question(current_objects)
    print("\n" + question_text)

    user_choice = input("Your choice (A/B/C/D): ").strip().upper()
    if user_choice not in ["A", "B", "C", "D"]:
        print("⚠️ Invalid choice. Exiting.")
        return

    if user_choice in ["A", "B", "C"]:
        # parse the categories, pick the chosen subset
        categories = parse_three_categories(question_text)
        print(categories)  # For debugging
        chosen_subset = categories.get(user_choice, [])

        if not chosen_subset:
            print("⚠️ Could not parse or empty subset, cannot refine further.")
            return

        # Recurse with the chosen subset
        narrow_down_interactive(chosen_subset, other_objects)

    else:
        # user_choice == "D": flip sets
        print("\nYou chose 'Other': switching to categorize the other set.")
        narrow_down_interactive(other_objects, current_objects)

def main():
    # Example usage:
    object_list = [
        "wine glass", "hammer", "apple", "screwdriver",
        "credit card", "tennis ball", "paintbrush", "laptop",
        "book", "fork", "bottle", "remote control", "cell phone",
        "basketball", "soap bar", "toothbrush", "scissors", "notebook",
        "mug", "key", "banana", "flashlight", "watermelon", "tablet"
    ]
    grasp_type = input("Enter desired grasp type (e.g., 'precision grasp' or 'power grasp'): ")

    # 1) Find which items GPT says are excluded for this grasp
    excluded = filter_objects_by_grasp(object_list, grasp_type)
    suitable = [obj for obj in object_list if obj not in excluded]
    
    # 2) Start by categorizing the "suitable" set, with the "excluded" set as the 'other' set
    print(f"Objects suitable for '{grasp_type}': {suitable}")
    print(f"Objects initially excluded for '{grasp_type}': {excluded}")
    narrow_down_interactive(suitable, excluded)

if __name__ == "__main__":
    main()
