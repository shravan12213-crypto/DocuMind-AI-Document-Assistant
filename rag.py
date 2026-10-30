from retrieve import retrieve
from generate import generate_answer


def answer_question(query):

    print("\nSearching the document...")

    # Step 1: Retrieve relevant chunks
    results = retrieve(query, top_k=3)

    if not results:
        return "No relevant information found in the document."

    # Step 2: Prepare context for Gemini
    context = ""

    for result in results:

        context += f"""
        Source Page: {result['page']}

        Content:
        {result['text']}

        -------------------------
        """

    # Step 3: Generate answer using Gemini
    print("Generating answer using Gemini...")

    answer = generate_answer(query, context)

    return answer


if __name__ == "__main__":

    print("\n========== DocuRAG ==========")

    question = input("\nAsk a question about your document: ")

    answer = answer_question(question)

    print("\n========== ANSWER ==========")
    print(answer)