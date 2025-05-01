import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

my_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=2000)

role_template = "คุณคือ {role} โปรดตอบคำถามต่อไปนี้ตามบทบาทของคุณ:\n\n{question}"
prompt = PromptTemplate(
    input_variables=["role", "question"],
    template=role_template
)

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.invoke({"role": "นักการตลาดดิจิทัล", "question": "วิธีโปรโมทสินค้าใหม่อย่างไรดี", "xx": "วิธีโปรโมทสินค้าใหม่อย่างไรดี"})

print(response)
