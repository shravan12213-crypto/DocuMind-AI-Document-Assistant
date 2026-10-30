from retrieve import retrieve
from generate import generate_answer


import time

from retrieve import retrieve
from generate import generate_answer


def answer_question(query):

    total_start = time.perf_counter()

    # STEP 1: Retrieval
    retrieval_start = time.perf_counter()

    print("\nSearching the document...")

    results = retrieve(query, top_k=5)

    retrieval_end = time.perf_counter()

    print(
        f"Retrieval time: "
        f"{retrieval_end - retrieval_start:.2f} seconds"
    )

    if not results:
        return "No relevant information found in the document."

    # STEP 2: Prepare context
    context = ""

    for result in results:

        context += f"""
Course: {result.get('course', 'Unknown')}
Unit: {result.get('unit', 'Unknown')}
Source Page: {result['page']}

Content:
{result['text']}

-------------------------
"""

    # STEP 3: Gemini generation
    generation_start = time.perf_counter()

    print("\nGenerating answer using Gemini...")

    answer = generate_answer(query, context)

    generation_end = time.perf_counter()

    print(
        f"Gemini generation time: "
        f"{generation_end - generation_start:.2f} seconds"
    )

    print(
        f"Total processing time: "
        f"{generation_end - total_start:.2f} seconds"
    )

    return answer


if __name__ == "__main__":

    print("\n========== DocuMind ==========")

    question = input("\nAsk a question about your document: ")

    answer = answer_question(question)

    print("\n========== ANSWER ==========")

    print(answer)



if __name__ == "__main__":

    print("\n========== DocuRAG ==========")

    question = input("\nAsk a question about your document: ")

    answer = answer_question(question)

    print("\n========== ANSWER ==========")
    print(answer)