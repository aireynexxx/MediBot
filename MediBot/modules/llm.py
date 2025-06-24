import ollama

def generate_answer(query, context_chunks, chat_history=[]):

    # Limit chunks and history for prompt length safety
    context = "\n\n".join(context_chunks[:4])

    history_text = ""
    for role, msg in chat_history[-6:]:
        if role == "user":
            history_text += f"User: {msg.strip()}\n"
        else:
            history_text += f"Assistant: {msg.strip()}\n"

    # Construct the full prompt
    prompt = f"""
You are MediBot, an intelligent medical assistant.

Your job is to:
- Use only the provided medical context and conversation history
- Suggest 2 to 3 medically reasonable possibilities for what might be causing the user's symptoms
- Explain them in simple language
- If necessary, ask follow-up questions

You must NEVER mention real patients or case studies.
NEVER say "I'm just a model" or "I'm not a doctor".

---
Medical case context:
{context}

Chat history so far:
{history_text}

Current user question:
{query}

Answer in 2–3 medically grounded sentences. Be specific, avoid vague phrases like "consult a doctor", and use simple language:

Answer:
""".strip()

    # Generate with Ollama
    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response['message']['content'].strip()
