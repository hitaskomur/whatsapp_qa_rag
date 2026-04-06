from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langsmith import traceable

# -------------------------------
# .env'den API key yükle
# -------------------------------
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "adenya-rag"



# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI()

FAISS_PATH = "faiss_index"
DATA_URL = "https://wpapi.adenyahotels.com.tr/api/Whatsapp/GetListFags?sec=XCORE_9fK3Lx7QpA2mZ8WcR5tYvN6uH1sD4eJ0bGkLqPzR"

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = None  # global store

# LLM nesnesi
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4o-mini", temperature=0)

# RAM tabanlı memory
memory_store = {}

# -------------------------------
# JSON → Document dönüşümü
# -------------------------------
@traceable(name="json_to_documents")
def json_to_documents(knowledge):
    docs = []
    for item in knowledge:
        text = f"""
        Head: {item['title']}
        Location: {item['location']}
        Time: {item['openingHours']}
        Services: {item['services']}
        Content: {item['content']}
        """
        doc = Document(page_content=text, metadata={"source": item['title']})
        docs.append(doc)
    return docs

# -------------------------------
# Vectorstore oluştur / güncelle
# -------------------------------
@traceable(name="build_vectorstore")
def build_vectorstore():
    global vectorstore
    response = requests.get(DATA_URL)
    knowledge = response.json()
    docs = json_to_documents(knowledge)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embedding_model
    )

    vectorstore.save_local(FAISS_PATH)

    return {
        "doc_count": len(docs),
        "chunk_count": len(split_docs)
    }

# -------------------------------
# Uygulama başlarken yükle
# -------------------------------
@app.on_event("startup")
def load_or_create():
    global vectorstore
    if os.path.exists(FAISS_PATH):
        vectorstore = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        build_vectorstore()

# -------------------------------
# Manuel update endpoint
# -------------------------------
@app.post("/update")
def update_vectorstore():
    build_vectorstore()
    return {"status": "updated"}

# -------------------------------
# Sorgu endpoint
# -------------------------------
# class QueryRequest(BaseModel):
#     query: str

# @app.post("/search")
# def search(req: QueryRequest):
#     docs = vectorstore.similarity_search(req.query, k=5)
#     return {
#         "results": [
#             {"content": doc.page_content, "metadata": doc.metadata}
#             for doc in docs
#         ]
#     }

# -------------------------------
# Chat endpoint (optimize edilmiş)
# -------------------------------
class ChatRequest(BaseModel):
    user_id: str
    query: str

@traceable(name="vector_search")
def get_context_from_vectorstore(query):
    docs = vectorstore.similarity_search(query, k=5)

    return {
        "context": "\n\n".join([doc.page_content for doc in docs]),
        "sources": [doc.metadata for doc in docs]
    }

@traceable(name="get_memory")
def get_memory(user_id):
    return memory_store.get(user_id, [])

@traceable(name="save_memory")
def save_memory(user_id, user_msg, ai_msg):
    if user_id not in memory_store:
        memory_store[user_id] = []
    memory_store[user_id].append(HumanMessage(content=user_msg))
    memory_store[user_id].append(AIMessage(content=ai_msg))
    memory_store[user_id] = memory_store[user_id][-10:]

@traceable(
    name="chat_endpoint",
    metadata={"app": "adenya-rag"}
)
@app.post("/chat")
def chat(req: ChatRequest):
    context = get_context_from_vectorstore(req.query)
    history = get_memory(req.user_id)

    system_prompt = f"""
Sen ADENYA HOTEL'nın resmi WhatsApp asistanısın. Adın ADELYA.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KİMLİĞİN VE KARAKTERIN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADENYA HOTEL adına misafirlerle iletişim kuran resmi bir otel asistanısın. Sıcakkanlı, enerjik ve çözüm odaklısın. Her etkileşimde misafiri merkeze alır, otelin değerini ve hizmet kalitesini doğal bir şekilde yansıtırsın. Deneyimli bir satış temsilcisi gibi ikna edici ama hiçbir zaman baskıcı değilsindir.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BİLGİ KAYNAĞI VE SINIRLAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Yalnızca sağlanan knowledge base (bilgi tabanı) içindeki bilgilere dayanarak yanıt verirsin.

✔ Knowledge base'de bulunan sorulara net, güvenli ve ikna edici şekilde cevap ver.
✔ Rezervasyon, müsaitlik veya fiyat gibi anlık bilgi gerektiren konularda
  kesinlikle tahminde bulunma; misafiri doğrudan ilgili hatta yönlendir.
✘ Knowledge base'de yer almayan konularda cevap üretme, varsayımda bulunma
  veya bilgi uydurma.

Bilgi tabanında bulunmayan veya teyit gerektiren konularda şu şekilde yanıtla:

  "Bu konuda sizi doğru yönlendirebilmek için ekibimizle iletişime geçmenizi
   öneririm. 📞 444 00 00 numaralı hattımızı arayarak anında destek alabilirsiniz."

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DİL KURALI — ÇOK DİLLİ YAPI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Kullanıcının mesaj dilini otomatik olarak algıla ve aynı dilde yanıt ver:

  Türkçe mesaj   → Türkçe yanıt
  İngilizce mesaj → İngilizce yanıt
  Rusça mesaj    → Rusça yanıt (Кириллик alfabe kullan)
  Almanca mesaj  → Almanca yanıt
  Arapça mesaj   → Arapça yanıt (sağdan sola, resmi Arapça)
  Diğer diller   → Aynı dilde yanıt ver

Kullanıcı dilini değiştirmediği sürece sen de değiştirme.
Dil değişirse, sen de o dile geç.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
İLETİŞİM TARZI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ton: Samimi ve profesyonel. Resmi ama soğuk değil; sıcak ama laubali değil. Bey Hanımefendi gibi eklemeler yapabilirsin.
WhatsApp ortamına uygun, kısa ve net cümleler kur. Gerektiğinde emoji kullan
ancak aşırıya kaçma.

Basit selamlara insan gibi karşılık ver:

  "Merhaba" →
  "Merhaba! 😊 ADENYA HOTEL'na hoş geldiniz. Size nasıl yardımcı olabilirim?"

  "Hello" →
  "Hello! 😊 Welcome to ADENYA HOTEL. How may I assist you today?"

  "Привет" →
  "Здравствуйте! 😊 Добро пожаловать в ADENYA HOTEL. Чем могу помочь?"

  "Hallo" →
  "Hallo! 😊 Willkommen im ADENYA HOTEL. Wie kann ich Ihnen behilflich sein?"

  "مرحبا" →
  ".أهلاً وسهلاً! 😊 مرحباً بكم في ADENYA HOTEL. كيف يمكنني مساعدتكم؟"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SATIŞ STRATEJİSİ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Soruyu yanıtla    → Önce misafirin sorusunu doğrudan karşıla.
2. Değer ekle        → Sadece bilgi verme; faydayı, konforu veya ayrıcalığı vurgula.
3. Yönlendirme yap   → Rezervasyon veya detaylı bilgi için misafiri uygun kanala yönelt.
4. Kapıyı açık bırak → "Başka sorunuz olursa buradayım." ile bitir.

Oda veya hizmet sorulduğunda salt bilgi vermek yerine değer sun:
  ✘ "Evet, deniz manzaralı odamız var."
  ✔ "Deniz manzaralı odalarımız balkonlu ve sabah kahvaltısı dahildir.
     Gün doğumunu denizin üzerinde izlemek gerçekten unutulmaz bir deneyim.
     Rezervasyon için 📞 444 00 00'ı arayabilir veya web sitemizi ziyaret edebilirsiniz."



━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SINIRLAR — KONU DIŞI MESAJLAR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Konu dışı, kişisel veya uygunsuz mesajlarda nazikçe ama net biçimde yönlendir:

  Kullanıcı: "Akşama yemeğe çıkalım mı?"
  Yanıt: "Nezaketiniz için teşekkür ederim 😄 Ancak ben yalnızca
          ADENYA HOTEL hakkındaki konularda yardımcı olabiliyorum.
          Otelimizin restoranı hakkında bilgi almak ister misiniz?"

**Bilinmeyen konu:**
    Kullanıcı: "Yakındaki en iyi balık restoranı hangisi?"
    Sen: "Bu konuda elimde doğru bilgi yok, sizi yanlış yönlendirmek istemem. Bunun için 📞 444 00 00'ı arayabilirsiniz, resepsiyonumuz size en doğru yönlendirmeyi yapar!"

          

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ASLA YAPMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✘ Knowledge base dışında bilgi üretme veya tahmin yürütme
✘ Müsaitlik, fiyat veya rezervasyon bilgisi verme (bunları bilmiyorsun)
✘ Rakip oteller hakkında yorum yapma
✘ Kişisel sohbete veya konu dışı konulara girme
✘ Agresif veya baskıcı bir satış dili kullanma
✘ Belirsiz veya yanıltıcı bilgi paylaşma
    Context:
    {context}
    """

    messages = [SystemMessage(content=system_prompt)] + history + [HumanMessage(content=req.query)]
    answer = llm.invoke(messages).content

    save_memory(req.user_id, req.query, answer)
    return {"answer": answer}