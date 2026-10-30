import os

from dotenv import load_dotenv
from google import genai


# Load environment variables
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=api_key)


def generate_answer(query, context):

    prompt = f"""
    You are DocuRAG, an AI assistant that answers questions
    using the provided document context.

    Instructions:
    1. Answer using only the provided context.
    2. Do not make up information.
    3. If the answer is not available in the context,
       clearly say that the document does not contain the answer.
    4. Keep the answer clear and student-friendly.
    5. Mention the source page numbers.

    Context:
    {context}

    User Question:
    {query}

    Answer:
    """

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )

    return response.text


if __name__ == "__main__":

    sample_context = """
    Unit 4: Recurrence Relations and Generating Functions.

    Topics include Fibonacci numbers, Catalan numbers,
    Derangements, Tower of Hanoi, partitions,
    linear recurrence relations and generating functions.

    Source: Page 40.
    """

    question = "What topics are covered in recurrence relations?"

    answer = generate_answer(question, sample_context)

    print("\nDocuRAG Answer:\n")
    print(answer)