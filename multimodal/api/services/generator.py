import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))


async def generate_answer(question: str, chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant content found for your query."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        ctype  = chunk["content_type"].upper()
        pg     = f"p.{chunk['page_number']}" if chunk.get("page_number") else ""
        fignum = chunk.get("figure_number", "")
        ref    = f"[{ctype} {fignum} {pg}]".strip()
        context_parts.append(f"{ref}\n{chunk['content']}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a research assistant with access to retrieved content from academic papers.

Retrieved content:
{context}

Question: {question}

Instructions:
- Answer based strictly on the retrieved content
- Cite sources as [TEXT p.N], [FIGURE N p.N], [TABLE N p.N]
- If a figure or table is referenced, describe what it shows
- Be precise and academic in tone
- If content is insufficient, say so clearly"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()
