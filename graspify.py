import os
import ast
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
    Filters out objects that are NOT suitable for the given grasp_type.

    We provide GPT with short definitions:
    - Power grasp: used for bigger or heavier objects that typically require the
      entire hand (e.g., grabbing a hammer handle, a basketball, or a large bottle).
    - Precision grasp: used for smaller objects requiring fingertip control 
      (e.g., picking up a pen, key, or credit card).
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

def categorize_and_ask(objects):
    """
    Asks GPT to split the given objects into 3 non-overlapping categories, each with at least one item,
    plus a 4th category 'Other' if needed. Returns a multiple-choice question as a string.
    """
    prompt = (
        f"Split the following objects into exactly 3 non-overlapping categories. "
        f"Each category must have at least ONE item. If any items don't fit into those 3 categories, "
        f"place them into a 4th option 'D) Other'. Provide a multiple-choice question with the format:\n\n"
        f"Question: Which group of objects are you thinking about?\n"
        f"A) CategoryName: [list, of, items]\n"
        f"B) CategoryName: [list, of, items]\n"
        f"C) CategoryName: [list, of, items]\n"
        f"D) Other: [any leftover items]\n\n"
        f"Do not include empty lists. Do not overlap items. Use each object exactly once. "
        f"Return only this formatted question (no extra text or explanation).\n\n"
        f"Objects:\n{objects}"
    )
    return call_gpt(prompt)

def extract_objects_from_choice(question, user_choice):
    """
    Extracts ONLY the list of objects from the user-chosen category in the question. 
    Returns a Python list or None if parsing fails.
    """
    prompt = (
        f"The following multiple-choice question was asked:\n{question}\n\n"
        f"The user selected option {user_choice}.\n"
        f"Extract ONLY the list of objects under that option and return it as a valid Python list.\n"
        f"Example output: ['item1', 'item2']\n"
        f"Do NOT return the option label, category name, or any explanation—just the list."
    )
    result = call_gpt(prompt)
    print(f"\n📦 GPT raw output for chosen option {user_choice}:\n{result}\n")
    try:
        parsed = ast.literal_eval(result)
        if not isinstance(parsed, list):
            raise ValueError("Parsed result is not a list.")
        return parsed
    except Exception as e:
        print(f"⚠️ Could not parse selected objects. Error: {e}")
        return None

def narrow_down_loop(objects, grasp_type):
    """
    Narrow down the user's intention by:
    1) Excluding objects that do not fit the desired grasp.
    2) If multiple objects remain, recursively prompt with a multiple-choice question
       to narrow down further categories.
    3) Handle special cases where only 2 items or 1 item remain.
    4) Stop if the list does not change or if only 1 item is left.
    """
    excluded = filter_objects_by_grasp(objects, grasp_type)
    remaining = [obj for obj in objects if obj not in excluded]
    
    print(f"\n🔍 Excluding based on '{grasp_type}': {excluded}")
    
    if not excluded:
        print("❌ No objects were excluded. Cannot refine further.")
        return
    
    # If after exclusion we only have 1 or 2 items, no need for categorization:
    if len(excluded) == 1:
        print(f"✅ Final grasp intention: {excluded[0]}")
        return
    elif len(excluded) == 2:
        # Directly ask user which one:
        print(f"\nWe have two possible objects left for '{grasp_type}': {excluded}")
        choice = None
        while choice not in ["1", "2"]:
            choice = input("Which one is it? Enter 1 or 2: ").strip()
        final_obj = excluded[0] if choice == "1" else excluded[1]
        print(f"✅ Final grasp intention: {final_obj}")
        return
    
    visited_sets = set()  # Keep track of previously seen sets to avoid infinite loops
    
    # Begin narrowing loop
    while excluded:
        # If there's only 1 item left, done
        if len(excluded) == 1:
            print(f"\n✅ Final grasp intention: {excluded[0]}")
            return
        
        # If there's exactly 2 items left, ask directly
        if len(excluded) == 2:
            print(f"\nWe have two possible objects left for '{grasp_type}': {excluded}")
            choice = None
            while choice not in ["1", "2"]:
                choice = input("Which one is it? Enter 1 or 2: ").strip()
            final_obj = excluded[0] if choice == "1" else excluded[1]
            print(f"✅ Final grasp intention: {final_obj}")
            return
        
        # Check for infinite loops (same set again)
        frozen = frozenset(excluded)
        if frozen in visited_sets:
            print("\n❌ We are stuck. The same objects keep repeating. Cannot refine further.")
            return
        visited_sets.add(frozen)
        
        # Ask GPT to categorize
        question = categorize_and_ask(excluded)
        print("\n" + question)
        
        user_choice = input("Your choice (A/B/C/D): ").strip().upper()
        # If user does not choose a valid option, we skip
        if user_choice not in ["A", "B", "C", "D"]:
            print("⚠️ Invalid choice. Exiting.")
            return
        
        # If user picks 'Other' but there's no leftover, it means no refinement
        if user_choice == "D":
            # Attempt to see if GPT gave leftover items
            new_selection = extract_objects_from_choice(question, "D")
            if not new_selection:
                print("\n⚠️ No items under 'Other'. Cannot refine further.")
                return
            else:
                excluded = new_selection
                # If 'Other' is the entire set, it won't narrow anything
                if set(excluded) == set(frozen):
                    print("\n❌ 'Other' didn't reduce the set. Stopping.")
                    return
            continue
        
        # Otherwise, pick from A/B/C
        new_selection = extract_objects_from_choice(question, user_choice)
        if not new_selection:
            print("⚠️ Could not extract objects. Keeping previous list.")
            continue
        if set(new_selection) == set(excluded):
            # No improvement
            print("\n❌ This selection is the entire set, no refinement. Stopping.")
            return
        
        # Switch excluded to new selection
        excluded = new_selection
    
    # If we somehow fall out of the loop
    if excluded and len(excluded) == 1:
        print(f"\n✅ Final grasp intention: {excluded[0]}")
    else:
        print("\n❌ Could not determine the object or no objects remain.")

if __name__ == "__main__":
    object_list = [
        "wine glass", "hammer", "apple", "screwdriver",
        "credit card", "tennis ball", "paintbrush", "laptop",
        "book", "fork", "bottle", "remote control", "cell phone",
        "basketball", "soap bar", "toothbrush", "scissors", "notebook",
        "mug", "key", "banana", "flashlight", "watermelon", "tablet"
    ]
    grasp_type = input("Enter desired grasp type (e.g., 'precision grasp' or 'power grasp'): ")
    narrow_down_loop(object_list, grasp_type)
