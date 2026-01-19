from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("🧠 AI is ready. Type 'exit' to quit.\n")

while True:
    user_prompt = input("You: ")

    if user_prompt.lower() in ["exit", "quit"]:
        print("👋 Bye")
        break

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a concise, intelligent assistant. Answer clearly."
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.7  # creativity control
    )

    ai_reply = response.choices[0].message.content
    print("\nAI:", ai_reply, "\n")
