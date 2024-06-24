import os
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import warnings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.schema import HumanMessage, AIMessage
import pandas as pd
from sqlalchemy import create_engine
from itertools import product





openai.api_key = os.getenv("OPENAI_API_KEY")

warnings.filterwarnings("ignore")


def get_similary_vars(var):

    embedding = OpenAIEmbeddings()
    vectordb_node = Chroma(persist_directory='vec_nodes', embedding_function=embedding)

    docs = vectordb_node.similarity_search_with_score(var, k=20)

    # print(type(docs))
    result = []

    for doc, score in docs:
        if score < 0.45:
            content_lines = doc.page_content.split('\n')
            data = {}
            for line in content_lines:
                if line.startswith('name:'):
                    data['name'] = line.split('name: ')[1]
                elif line.startswith('define:'):
                    data['define'] = line.split('define: ')[1]
                elif line.startswith('source:'):
                    data['source'] = line.split('source: ')[1]
            result.append(data)

    return result




def get_respond_1(var_1, var_2):
    # 相似的变量集合
    vars_1 = get_similary_vars(var_1)
    vars_2 = get_similary_vars(var_2)
    set_1 = {item['name'] for item in vars_1}
    set_2 = {item['name'] for item in vars_2}
    vars_list = []
    vars_list.append({var_1:set_1})
    vars_list.append({var_2:set_2})

    user_input = str(vars_list)

    # RAG
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    persist_directory = 'vec_edges'
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

    retriever = vectordb.as_retriever(k=10)

    qa_system_prompt = """ The user's input consists of a list with two groups of variables. 
        Variables within a group are similar, and the user would like to know if there are any relationships between the variables from different groups.
        Your task is to utilize the retrieved context below to answer the question regarding the relationships between the variables from the two groups, and for each relationship stated, indicate the source of the literature that investigated such a relationship.
        Here is an example of your output.And if you do not find any relationship, you should answer that there is currently no literature studying the relationship between these two groups of variables.

        The variables you've provided can be grouped into two main categories: 'VAR_1' and 'VAR_2'. 

        Under 'VAR_1', we have 'a', 'b', 'c', 'd', 'e', 'f', and 'g'. These variables are related to the importance of a node in a network. 

        Under 'VAR_2', we have 'h', 'i', 'j', 'k', 'l', 'm', and 'n'. These variables are related to the production of patents. 
        
        'a'  have been found to have a positive effect respectively on 'j' (source: The impact of multilevel networks on innovation).
        'd'  have been found to have a xxxxxxx effect on 'm' (source: xxxxxxxxxxxxxxx).


    {context}"""

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    msg = rag_chain.invoke({"input": user_input})


    return  msg['answer']




def get_respond_2(var_1, var_2):
    # 获取相似变量列表
    vars_1 = get_similary_vars(var_1)
    vars_2 = get_similary_vars(var_2)
    list_1 = [item['name'] for item in vars_1]
    list_2 = [item['name'] for item in vars_2]
    vars_list = []
    vars_list.append({var_1:list_1})
    vars_list.append({var_2:list_2})

    user_input = str(vars_list)

    #获取子表
    file_path = 'edges.csv'  
    data = pd.read_csv(file_path)

    list_1 = [item['name'] for item in vars_1]
    list_2 = [item['name'] for item in vars_2]

    combinations_1 = list(product(list_1, list_2))
    combinations_df_1 = pd.DataFrame(combinations_1, columns=['list_1', 'list_2'])

    combinations_2 = list(product(list_2, list_1))
    combinations_df_2 = pd.DataFrame(combinations_2, columns=['list_2', 'list_1'])

    merged_df_1 = pd.merge(combinations_df_1, data, left_on=['list_1', 'list_2'], right_on=['node_1', 'node_2'], how='inner')
    merged_df_2 = pd.merge(combinations_df_2, data, left_on=['list_2', 'list_1'], right_on=['node_1', 'node_2'], how='inner')


    merged_df_2 = merged_df_2.rename(columns={'list_2': 'list_1', 'list_1': 'list_2'})


    final_df = pd.concat([merged_df_1, merged_df_2], ignore_index=True)[['node_1', 'node_2', 'edge', 'source']]



    # 获得答复
    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY")
    )

    qa_system_prompt = """ The user's input consists of a list with two groups of variables. 
        Variables within a group are similar, and the user would like to know if there are any relationships between the variables from different groups.
        Your task is to utilize the user's input content to answer the question regarding the relationships between the variables from the two groups, and for each relationship stated, indicate the source of the literature that investigated such a relationship.
        Here is an example of your output.You only need to follow this format, and do not need to pay attention to the content of this example.
        And if you do not find any relationship, you should answer that there is currently no literature studying the relationship between these two groups of variables.

        The variables you've provided can be grouped into two main categories: 'VAR_1' and 'VAR_2'. 

        Under 'VAR_1', we have 'a', 'b', 'c', 'd', 'e', 'f', and 'g'. These variables are related to the importance of a node in a network. 

        Under 'VAR_2', we have 'h', 'i', 'j', 'k', 'l', 'm', and 'n'. These variables are related to the production of patents. 
        
        'a'  have been found to have a positive effect respectively on 'j' (source: The impact of multilevel networks on innovation).
        'd'  have been found to have a xxxxxxx effect on 'm' (source: xxxxxxxxxxxxxxx)."""
    
    content = str(final_df)

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": qa_system_prompt},
            {"role": "user", "content": user_input},
            {"role": "user", "content": content}
        ],
        model="gpt-4o",
    )

    return chat_completion.choices[0].message.content




def get_respond_3(var_1,var_2):
    # 获取相似变量列表
    vars_1 = get_similary_vars(var_1)
    vars_2 = get_similary_vars(var_2)
    list_1 = [item['name'] for item in vars_1]
    list_2 = [item['name'] for item in vars_2]

    if list_1 and list_2:

        # 将列表转换为字符串格式，并确保每个项用单引号括起来，并正确处理单引号
        list_1_str = ', '.join(["'{}'".format(item.replace("'", "''")) for item in list_1])
        list_2_str = ', '.join(["'{}'".format(item.replace("'", "''")) for item in list_2])
    

        # 数据库连接参数
        db_user = 'root'
        db_password = '123456'
        db_host = 'localhost'
        db_port = '3306'  # 默认是3306
        db_name = 'MySQL'
        table_name = 'relation'

        # 创建数据库连接
        engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')

        # 构建SQL查询，包含两种情况，并使用DISTINCT确保结果唯一
        query = """
        SELECT DISTINCT * FROM {}
        WHERE (node_1 IN ({}) AND node_2 IN ({}))
            OR (node_1 IN ({}) AND node_2 IN ({}));
        """.format(table_name, list_1_str, list_2_str, list_2_str, list_1_str)

        # 执行SQL查询并将结果读取到DataFrame
        matching_rows = pd.read_sql(query, con=engine)

        return matching_rows
    else:
        print("null")
        return




# test
while True:
    user_input = input("welcome to using our tool(input 1 to exit)!\n")
    if user_input == "1":
        break
    else:
        var_1 = input("input the first variable: ")
        var_2 = input("input the second variable: ")
        respond = get_respond_3(var_1, var_2)
        print(respond)

# respond = get_respond(var_1, var_2)

# print(respond)



