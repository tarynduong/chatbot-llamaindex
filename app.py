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
    return response.response

def main():
    init_page()
    file = st.file_uploader('Upload file: ', type=['pdf', 'txt', 'docs'])
    if file is not None:
        with open(os.path.join('data', file.name), 'wb') as f: 
            f.write(file.getbuffer())        
        st.success('Saved File')
        documents = SimpleDirectoryReader('./data').load_data()
        file_names = []
        for doc in documents:
            file_names.append(doc.metadata['file_name'])
        st.write('Current documents in folder:', ', '.join(file_names))
        
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
        query_engine = index.as_query_engine()

        init_messages()

        # Get user input -> Generate the answer
        if user_input := st.chat_input('Input your question!'):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner('Bot is typing ...'):
                # answer = get_answer(llm, user_input)
                answer = get_answer(query_engine, user_input)
                print(f'Answer:\n```\n{answer}\n```')
            st.session_state.messages.append(AIMessage(content=answer))

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
