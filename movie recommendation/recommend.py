from llama_cpp import Llama

# 1. Initialize the model
llm = Llama(
    model_path="./movie-recommendation-system.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

print("--- Model Loaded. Thinking... ---")

# 2. Here you can change the prompt to test different recommendations
prompt = "Recommend 5 movies for someone who likes Conjuring."

# 3. Generate the recommendations
output = llm(
    f"Question: {prompt}\nAnswer:",
    max_tokens=500,
    stop=["Question:"], 
    echo=False
)

# 4. Print the result
result = output["choices"][0]["text"].strip()
if result:
    print("\nRecommendations:\n", result)
else:
    print("\nThe model returned an empty response. Try changing the prompt.")