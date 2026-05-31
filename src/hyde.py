import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── NO CHANGES NEEDED — reads GROQ_API_KEY from .env automatically ──
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HYDE_PROMPT = """You are a research scientist writing technical content.

A student asked: "{query}"

Write a dense, technical paragraph (5-7 sentences) that reads like an excerpt 
from a peer-reviewed research paper answering this question.

Rules:
- Use academic vocabulary and technical precision
- Include likely variable names, equations, or algorithm steps if relevant
- Do NOT say "In this paper" or "We propose"
- Write as if explaining to a fellow researcher
- Be specific, not vague

Paragraph:"""


def expand_query_with_hyde(query: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": HYDE_PROMPT.format(query=query)
            }],
            temperature=0.3,
            max_tokens=250
        )
        hypothetical_doc = response.choices[0].message.content.strip()
        # combine original query + hypothetical doc for best retrieval
        combined = f"{query}\n\n{hypothetical_doc}"
        return combined

    except Exception as e:
        print(f"HyDE failed, using original query: {e}")
        return query


if __name__ == "__main__":
    result = expand_query_with_hyde("How does the GAN discriminator update work?")
    print(result)