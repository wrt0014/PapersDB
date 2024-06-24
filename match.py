import pandas as pd

# 读取CSV文件
edges_df = pd.read_csv('edges.csv')
nodes_df = pd.read_csv('nodes.csv')

# Merge edges表的node_1和node_2与nodes表的name
merged_df_1 = edges_df.merge(nodes_df, left_on='node_1', right_on='name', how='left', suffixes=('', '_node_1'))
merged_df_2 = merged_df_1.merge(nodes_df, left_on='node_2', right_on='name', how='left', suffixes=('', '_node_2'))

# 查找node_1或node_2没有在nodes表中找到匹配的记录
invalid_nodes = merged_df_2[(merged_df_2['name'].isnull()) | (merged_df_2['name_node_2'].isnull())]

output_df = pd.DataFrame(columns=['source', 'unmatched_node'])

for _, row in invalid_nodes.iterrows():
    if pd.isnull(row['name']):
        output_df = output_df._append({'source': row['source'], 'unmatched_node': row['node_1']}, ignore_index=True)
    if pd.isnull(row['name_node_2']):
        output_df = output_df._append({'source': row['source'], 'unmatched_node': row['node_2']}, ignore_index=True)

# 导出到CSV文件
output_df.to_csv('unmatched_nodes.csv', index=False)

print("未匹配的节点和对应的source已导出到'unmatched_nodes.csv'")