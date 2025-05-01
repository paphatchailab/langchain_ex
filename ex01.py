from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
model="gpt-4o-mini",
api_key="sk-",
temperature= 0.8,
max_tokens=2000,
)

response = llm.invoke("ให้ไอเดียทำแคมเปญโฆษณาน้ำปลาแบบแหวกแนว")

print(response.content)