from langchain_openai import ChatOpenAI

# สร้างโมเดล
llm = ChatOpenAI(
model="gpt-4o-mini",
api_key="sk-"
)
#เรียกใช้งาน Model
response = llm.invoke("นายกรัฐมนตรีคนล่าสุดของประเทศไทยชื่อว่าอะไร")

print(response.content)