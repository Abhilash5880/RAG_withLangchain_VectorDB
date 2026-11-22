from importlib.metadata import version
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print(f"langchain-core version: {version('langchain-core')}")
print(f"langgraph version: {version('langgraph')}")
print(f"langchain-google-genai version: {version('langchain-google-genai')}")


def main():
    print("Hello from langc-course!")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
    response = llm.invoke("Say 'setup complete!' in two words.")
    print(f"Response: {response.content}")
    print("Setup complete!")


if __name__ == "__main__":
    main()