import streamlit as st
from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import AzureOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from llama_index.embeddings import AzureOpenAIEmbedding
import os
from dotenv import load_dotenv
load_dotenv(override=True)

# save keys in .env file
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
ENDPOINT = os.getenv('OPENAI_API_BASE')
OPENAI_VERSION = os.getenv('OPENAI_API_VERSION')

def init_page():
    st.set_page_config(page_title='Personal Chatbot', page_icon='mag_right')
    st.header('Knowledge Query Assistant')
    st.write("I'm here to help you get information from your file.")
    st.sidebar.title('Option')

def select_llm():
    return AzureOpenAI(
    model='gpt-35-turbo',
    deployment_name='bot-llm',
    api_key=OPENAI_KEY,
    azure_endpoint=ENDPOINT,
    api_version=OPENAI_VERSION
    )

def select_embedding():
    return AzureOpenAIEmbedding(
    model='text-embedding-ada-002',
    deployment_name='bot-embedding',
    api_key=OPENAI_KEY,
    azure_endpoint=ENDPOINT,
    api_version=OPENAI_VERSION
    )

def init_messages():
    clear_button = st.sidebar.button('Clear Conversation', key='clear')
    if clear_button or 'messages' not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content='You are a helpful AI assistant. Reply your answer in markdown format.'
            )
        ]

def get_answer(query_engine, messages):
    response = query_engine.query(messages)
    return response.response, response.metadata

def main():
    init_page()
    file = st.file_uploader('Upload file: ', type=['pdf', 'txt', 'docx'])
    if file is not None:
        with open(os.path.join('data', file.name), 'wb') as f: 
            f.write(file.getbuffer())        
        st.success('Saved file!')
        documents = SimpleDirectoryReader('./data').load_data()
        file_names = []
        for doc in documents:
            file_names.append(doc.metadata['file_name'])
        st.write('Current documents in folder:', ', '.join(file_names))
        keywords = []
        for i in file_names:
            keywords.append(i.split('.')[0])
        keywords = keywords + ['Amazon', 'AWS', 'statistics', 'collection', 'machine learning', 'AI']
        
        # Load llm & embed model
        llm = select_llm()
        embed = select_embedding()
        service_context = ServiceContext.from_defaults(
            llm=llm,
            embed_model=embed,
        )

        # Setup query engine
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        # chunks = index.as_retriever(similarity_top_k=2)
        query_engine = index.as_query_engine()

        init_messages()

        # Get user input -> Generate the answer
        greetings = ['Hey', 'Hello', 'hi', 'hello', 'hey', 'helloo', 'hellooo', 'g morning', 'gmorning', 'good morning', 'morning',
                    'good day', 'good afternoon', 'good evening', 'greetings', 'greeting', 'good to see you',
                    'its good seeing you', 'how are you', "how're you", 'how are you doing', "how ya doin'", 'how ya doin',
                    'how is everything', 'how is everything going', "how's everything going", 'how is you', "how's you",
                    'how are things', "how're things", 'how is it going', "how's it going", "how's it goin'", "how's it goin",
                    'how is life been treating you', "how's life been treating you", 'how have you been', "how've you been",
                    'what is up', "what's up", 'what is cracking', "what's cracking", 'what is good', "what's good",
                    'what is happening', "what's happening", 'what is new', "what's new", 'what is neww', "g’day", 'howdy']
        compliment = ['thank you', 'thanks', 'thanks a lot', 'thanks a bunch', 'great', 'awesome', 'nice']
        if user_input := st.chat_input('Input your question!'):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner('Bot is typing ...'):
                answer, meta_data = get_answer(query_engine, user_input)
                print(answer)
            if user_input.lower() in greetings:
                answer = 'Hi, how are you? I am here to help you get information from your file. How can I assist you?'
                st.session_state.messages.append(AIMessage(content=answer))
            elif user_input.lower() in compliment:
                answer = 'My pleasure! If you have any more questions, feel free to ask.'
                st.session_state.messages.append(AIMessage(content=answer))
            elif all(i not in answer for i in keywords):
                answer = 'This is outside of scope of the provided knowledge base.'
                st.session_state.messages.append(AIMessage(content=answer))
            else:
                st.session_state.messages.append(AIMessage(content=f"**Source**: {list(meta_data.values())[0]['file_name']}  \n**Answer**: {answer}"))
                    

        # Show all the messages of the conversation
        messages = st.session_state.get('messages', [])
        for message in messages:
            if isinstance(message, AIMessage):
                with st.chat_message('assistant'):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message('user'):
                    st.markdown(message.content)
    else:
        if not os.listdir('./data'):
            st.write('No file is saved yet.')
    
if __name__ == '__main__':
    main()
