import subprocess

def ask_model_mistral(prompt, model="mistral"):
    """Send a prompt to the Mistral model using Ollama subprocess."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

def ask_model_llama2(prompt, model="llama2"):
    """Send a prompt to the llama2 model using Ollama subprocess."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"

with open("/Users/chandurajana/Documents/Chandu_Training/2025_09_GenAI_chandu/chat_history.txt", "a") as f:
    while True:
        user_input_model = input("which model yow want to run using ollama? mistral or llama2 or exit: ").strip()
        if user_input_model.lower() == "mistral":
            print("-" * 50)
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue  # skip empty input

            print("Chat:", end=" ", flush=True)
            response = ask_model_mistral(user_input)
            print(f"{user_input_model.lower()}:", response)

            f.write(f"You: {user_input}\n")
            f.write(f"{user_input_model.lower()}: {response}\n")

            print("-" * 50)
        elif user_input_model.lower() == "llama2":
            print("-" * 50)
            user_input = input("You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                print("👋 Goodbye!")
                break

            if not user_input:
                continue  # skip empty input

            print("Chat:", end=" ", flush=True)
            response = ask_model_llama2(user_input)
            print(f"{user_input_model.lower()}:", response)

            f.write(f"You: {user_input}\n")
            f.write(f"{user_input_model.lower()}: {response}\n")


            print("-" * 50)
        elif user_input_model.lower() in ["quit", "exit"]:
            break