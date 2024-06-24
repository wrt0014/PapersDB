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
from itertools import product
from sqlalchemy import create_engine




# 数据库连接参数
db_user = 'root'
db_password = '123456'
db_host = 'localhost'
db_port = '3306'  # 默认是3306
db_name = 'MySQL'
table_name = 'relation'

# 读取CSV文件
file_path = 'edges.csv'
edges_df = pd.read_csv(file_path)

# 创建数据库连接
engine = create_engine(f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')


'''
# 将DataFrame导入到MySQL
edges_df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)

print(f"表 '{table_name}' 已成功导入到数据库 '{db_name}' 中。")
'''


# 假设你已经有了两个变量列表
list_1 = ['City Centrality', 'Country Centrality']
list_2 = ['Patent Output', 'City Centrality']

# 将列表转换为字符串格式，以便在SQL查询中使用
list_1_str = ', '.join([f"'{item}'" for item in list_1])
list_2_str = ', '.join([f"'{item}'" for item in list_2])

# 构建SQL查询，包含两种情况，并使用DISTINCT确保结果唯一
query = f"""
SELECT DISTINCT * FROM {table_name}
WHERE (node_1 IN ({list_1_str}) AND node_2 IN ({list_2_str}))
   OR (node_1 IN ({list_2_str}) AND node_2 IN ({list_1_str}));
"""

# 执行SQL查询并将结果读取到DataFrame
matching_rows = pd.read_sql(query, con=engine)

# 显示匹配的行
print(matching_rows)