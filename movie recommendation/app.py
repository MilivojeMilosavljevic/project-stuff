from llama_cpp import Llama

# 1. Initialize the model
llm = Llama(
    model_path="./movie-recommendation-system.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

print("\n--- Movie Recommender AI (Chatter) ---")

# 2. Start a conversation loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]: break

    prompt = f"### Instruction: Recommend movies based on the user request.\n### User: {user_input}\n### Assistant:"

# 3. Generate the recommendations
    output = llm(
        prompt,
        max_tokens=300,
        stop=["###", "User:"], 
        echo=False,
        temperature=0.7 # Adds a bit of randomness for more engaging responses
    )

# 4. Print the result
    response = output["choices"][0]["text"].strip()
    
    if not response:
        print("AI: (The model returned an empty response. Let me try a different format...)")
    else:
        print(f"\nAI: {response}")