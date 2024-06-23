from django.shortcuts import render, redirect
from django.urls import reverse
from django.contrib import messages

from .models import Chats, Prompts

from .RAGconfig import RAG
from langchain_core.messages import HumanMessage, AIMessage, ChatMessage
from langchain_core.prompts.chat import HumanMessagePromptTemplate, AIMessagePromptTemplate


import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize

from datetime import date

from django.http import JsonResponse

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


rag = RAG()


def get_last_three_chat(history):
    if len(history)>=3:
        return history[-3:]
    else :
        return history
def get_user_chat_content(request, chat_id):
    chat_history = []
    
    chat_object = Chats.objects.filter(id=chat_id, user=request.user.id).get()
    prompts = Prompts.objects.filter(chat_id=chat_object)
    for prompt in prompts:
        chat_history.append({"human":prompt.question, "ai":prompt.response})
    return chat_history

def get_chat_history(chat_id, user_id):
    chat_history = []
    if int(chat_id) != -1:
        chat_object = Chats.objects.filter(id=chat_id, user=user_id).get()
        prompts = Prompts.objects.filter(chat_id=chat_object)
        for prompt in prompts:
            human_message = HumanMessage(content=prompt.question)
            # ai_message = AIMessage(content=prompt.response)
            ai_message = ChatMessage(content=prompt.response, role="Assistant", type="chat")
            chat_history.extend([human_message, ai_message])

    return chat_history

def get_user_chats(user_id):
    chat_object = Chats.objects.filter(user=user_id)
    # chat_object = Chats.objects.filter(user=user_id).order_by('-updated_at')
    user_chats = []
    for chat in chat_object:
        chat_dict = {"title":chat.title, "updated_at":chat.updated_at, "id":chat.id}
        user_chats.append(chat_dict)

    user_chats.sort(key=lambda x: x['updated_at'], reverse=True)
    
    return user_chats

def home(request, chat_id=-1):
    if request.user.is_authenticated:
        chat_history = []
        
        if int(chat_id) != (-1):
            chat_history = get_user_chat_content(request, chat_id)
        user_chats = get_user_chats(request.user.id)
        

        context = {
            "chat_id" : chat_id,
            'user_chats': user_chats,
            'chat_history': chat_history,
        }
        
        return render(request, 'home.html', context= context) 
    else:
        return render(request, 'login_register.html')
    # return render(request, 'home.html') 


def deneme(request):
    context = {}
    return render(request, 'deneme.html', context) 


def generate_title(message):
    # Mesajı tokenize et
    words = word_tokenize(message)
    
    # Küçük harfe çevir ve durak kelimeleri çıkar
    words = [word.lower() for word in words if word.isalnum()]
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Kelimelerin frekans dağılımını hesapla
    freq_dist = FreqDist(filtered_words)
    
    # En sık kullanılan kelimeleri al
    most_common_words = [word for word, freq in freq_dist.most_common(3)]
    
    # Başlık oluştur
    title = ' '.join(most_common_words).capitalize()
    if title == None or title == " ":
        title = "New Chat"
    return title

def CreateChat(request):
    if request.method == 'POST':
        user = request.user
        chat = Chats.objects.create(user=user, title="New Chat")
        chat.save()
        chat_id = chat.id
        home_url = reverse("chatbot_app:home" ,kwargs={"chat_id":chat_id})
        return redirect(home_url)
    else:
        return render(request, 'login_register.html')

def handlePrompt(request, chat_id): 
    if request.user.is_authenticated:   
        if request.method == 'POST':  

            form_type = request.POST.get('user_prompt_form', None)


            if form_type == 'user_prompt_ready':
                prompt = request.POST.get('user_prompt', None)
                if prompt:

                    history = get_chat_history(chat_id, request.user.id)
                    rag.clear_cache()
                    history=get_last_three_chat(history)
                    # print("HISTORY LENGTH")
                    # print(len(history))
                    rag_rsp = rag.ragQA(prompt, history)
                    # print("KEYS")
                    # print("keys: ",rag_rsp.keys())
                    response = rag_rsp["answer"]
                    index = response.find("Assistant:") 
                    #if index == -1:
                        # index = response.find("AI:")
                    last_index = -1

                    while index != -1:
                        last_index = index
                        index = response.find("Assistant:", index + 1)

                    if last_index != -1:
                        ai_part = response[last_index + len("Assistant:"):]

                    else:
                        ai_part = "I am not an expert AI on this subject. I can only answer about 'Aircraft Materials."
                
                    # yeni chat oluşturulduysa
                    if int(chat_id) == -1:
                        user = request.user
                        chat = Chats.objects.create(user=user, title=generate_title(prompt))
                        chat.save()
                        chat_id = chat.id
                        Prompts.objects.create(chat_id=chat, user=request.user, question=prompt, response=ai_part)
                    else:
                    # yan chatten seçilenlerden seçilen chat                
                        chat_object = Chats.objects.filter(id=chat_id, user=request.user.id).get()
                        if (len(history))==0:
                            chat_object.title = generate_title(prompt)
                            chat_object.save()
                        Prompts.objects.create(chat_id=chat_object, user=request.user, question=prompt, response=ai_part)
                        chat_object.updated_at = date.today()
                        chat_object.save()
                else:
                    messages.error(request=request,message=f"prompt yok")
                          
                rag.clear_cache()
                response_data = {'ai_response': ai_part}
                return JsonResponse(response_data)
            else:
                return JsonResponse({'error': 'Geçersiz form türü'}, status=400)
        else:
            return JsonResponse({'error': 'Geçersiz istek metodu'}, status=400)
    else:
        return JsonResponse({'error': 'Yetkilendirme başarısız'}, status=401)
    
    
    
def DeleteChat(request, current_chat_id = -1, delete_chat_id=-1):
    try:
        chat_object = Chats.objects.get(id=delete_chat_id, user=request.user)
        # İlgili chat'e ait promptları sil
        Prompts.objects.filter(chat_id=delete_chat_id).delete()
        # Chat'i sil
        chat_object.delete()
        
        if current_chat_id == delete_chat_id:
            home_url = reverse("chatbot_app:home" ,kwargs={"chat_id":-1})
            return redirect(home_url)
        else:
            home_url = reverse("chatbot_app:home" ,kwargs={"chat_id":current_chat_id})
            return redirect(home_url)
    except Chats.DoesNotExist:
        home_url = reverse("chatbot_app:home" ,kwargs={"chat_id":-1})
        return redirect(home_url)