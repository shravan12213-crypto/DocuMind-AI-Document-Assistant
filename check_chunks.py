
import json

with open("vectorstore/chunks.json", "r", encoding="utf-8") as file:
    chunks = json.load(file)

print(f"Total chunks: {len(chunks)}")

print("\nSearching for Unit 6 content...\n")

for i, chunk in enumerate(chunks):

    text = chunk["text"].lower()

    if (
        "assembly language programming" in text
        or "instruction set of pentium" in text
        or "assembler directives" in text
    ):
        print(f"Chunk Index: {i}")
        print(f"Page: {chunk['page']}")
        print(f"Text:\n{chunk['text']}")
        print("-" * 70)
