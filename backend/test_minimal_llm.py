import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

def test_minimal():
    key = os.environ.get("GOOGLE_API_KEY")
    print(f"Key found: {'Yes' if key else 'No'}")
    
    # 尝试用户指定的模型名称格式
    models = ["gemini-2.5-flash", "models/gemini-2.5-flash"]
    
    for model_name in models:
        print(f"\nTesting model: {model_name}...")
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=key,
                temperature=0
            )
            response = llm.invoke([HumanMessage(content="Hello")])
            print(f"Success! Response: {response.content}")
            return
        except Exception as e:
            print(f"Failed with {model_name}: {e}")

if __name__ == "__main__":
    test_minimal()
