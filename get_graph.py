from pathlib import Path
from openai import OpenAI
import json
import pandas as pd
import os
import csv

#从文献中提取信息构造知识图谱，参数filename为文献文件地址
def create_KGs(filepath):
    client = OpenAI(
    api_key = os.getenv("MOONSHOT_API_KEY"),
    base_url = "https://api.moonshot.cn/v1",
    )

    file_object = client.files.create(file=Path(filepath), purpose="file-extract")
 
    # 获取结果
    file_content = client.files.content(file_id=file_object.id).text
 
    # 把它放进请求中
    messages = [
        {
            "role": "system",
            "content": '''You are Kimi, an AI assistant powered by Moonshot AI. You are an excellent expert in reading papers and are adept at extracting the variables and relationships between variables in research papers.
                        The content of these papers focuses on empirical research related to patents. Typically, the papers discuss variables and the relationships between them. These relationships often manifest as hypotheses that are later tested and validated within the text. Your role is to identify the relationships between variables in the validated conclusions.
                        Additionally, the data presented in tables or figures within the papers may also represent the relationships between these variables. When necessary, you need to analyze the tables or figures to understand the relationships between the variables. While extracting the relationships between variables, it's crucial to only analyze the variables that have been identified and their relationships, maintaining consistency in the variable names when presenting your output.
                        Your task is to extract from the given literature the variables studied in the paper and the relationships between these variables.
                        Your output should be formatted and in English. First, output the title of the paper, then output the variables that may be included in the study, including independent variables, dependent variables, and control variables. The number of variables should be limited to 10. Finally, output the relationships between the variables studied in the text. These relationships include how variable 1 affects variable 2, and it is hoped that they can be as detailed and specific as possible, down to how variable 1 affects variable 2. These variables must be the ones you extracted before. If there is no specific description of the relationship in the text, there is no need to output it.
                        For example, if you extract variables X and Y, but in the relationship output, you mention X and Z, this is not allowed. If you want to output the relationship between X and Z, please include Z when you output the variables.
                        The output format should strictly follow the format of a JSON string.
                        The following is merely an example for you to refer to the output format. Please only refer to the format below and disregard the content.
                        {
                            "paper": "Exploring the use of patents in a weak institutional environment: The effects of innovation partnerships, firm ownership, and new management practices",
                            "variables": [
                            {
                                "Innovation Partnerships": "A dummy variable representing whether firms cooperated with other organizations on innovation."
                            },
                            {
                                "Firm Ownership": "A categorical variable with three categories: domestic, foreign, and domestic and foreign ownership."
                            },
                            {
                                "New Management Practices": "A binary variable indicating whether a firm had implemented new or significant changes in management methods."
                            },
                            {
                                "Propensity to Patent": "A binary variable extracted from a survey question asking whether firms patented during the 2001–2003 period."
                            }
                            ],
                        "relationships": [
                            {
                                "node_1": "Innovation Partnerships",
                                "node_2": "Propensity to Patent",
                                "edge": "Firms engaged in innovation-oriented collaborations are more inclined to patent than firms not involved in these partnerships, even in weak institutional environments."
                            },
                            {
                                "node_1": "Firm Ownership",
                                "node_2": "Propensity to Patent",
                                "edge": "Domestic and foreign firms in a weak institutional environment show no significant difference in their inclination to patent."
                            },
                            {
                                "node_1": "New Management Practices",
                                "node_2": "Propensity to Patent",
                                "edge": "The adoption of new management practices is negatively related to the use of patents, suggesting that they may serve as substitutes for patents in weak patent systems."
                            }  
                            ]
                        }'''
        },
        {
            "role": "system",
            "content": file_content,
        },
    ]

    # 调用 chat-completion, 获取 Kimi 的回答
    completion = client.chat.completions.create(
        model="moonshot-v1-128k",
        messages=messages,
        temperature=0.0,
        max_tokens=40000
    )

    # print(completion.choices[0].message.content)
 
    # json字符串：completion.choices[0].message.content
    data = json.loads(completion.choices[0].message.content)

    # print(completion.choices[0].message.content)

    new_nodes = []
    new_edges = []

    for variable in data['variables']:
        for name, definition in variable.items():
            new_nodes.append({
                'name': name,
                'define': definition,
                'source': data['paper']
            })

    for relationship in data['relationships']:
        new_edges.append({
            'node_1': relationship['node_1'],
            'node_2': relationship['node_2'],
            'edge': relationship['edge'],
            'source': data['paper']
        })
    

    nodes_file = 'nodes.csv'
    edges_file = 'edges.csv'


    def append_to_csv(file, fieldnames, new_data):
        file_exists = os.path.isfile(file)
    
        with open(file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
            if not file_exists:
                writer.writeheader()
        
            for row in new_data:
                writer.writerow(row)


    # 追加新的节点数据到 nodes.csv
    append_to_csv(nodes_file, ['name', 'define', 'source'], new_nodes)

    # 追加新的边数据到 edges.csv
    append_to_csv(edges_file, ['node_1', 'node_2', 'edge', 'source'], new_edges)

    print("添加成功")

    return



# 获取文件相对地址
def get_all_file_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            # 规范化文件路径并替换反斜杠为斜杠
            file_path = os.path.join(root, file).replace("\\", "/")
            file_paths.append(file_path)
    return file_paths



file_paths = get_all_file_paths("data")

for file_path in file_paths:
    create_KGs(file_path)



# create_KGs("data/1-s2.0-S0166497215000449-main.pdf")


'''

temp = """
{
    "paper": "The impact of technological relatedness, prior ties, and geographical distance on university–industry collaborations: A joint-patent analysis",
    "variables": [
        {
            "InnValue": "The value of university-industry joint innovations, measured by the total number of citations a joint patent received within five years of the issue date."
        },
        {
            "TechRel": "The degree of overlapping between the technology profile of the university and the firm jointly developing a patent."
        },
        {
            "PriorTies": "A binary variable indicating whether the university and the firm have previously registered joint patents in the five years prior to the joint patent under analysis."
        },
        {
            "GeoDist": "The geographical distance between the university and the firm, measured as the spatial distance in kilometers between their location sites."
        },
        {
            "FirmSize": "The size of the firm, measured by the number of full-time employers."
        },
        {
            "FirmPatents": "The technological capital of the firm, measured by the number of patents registered by the firm in the five years prior to the collaboration."
        },
        {
            "UnivSize": "The size of the university, measured by the number of full-time researchers."
        },
        {
            "UnivPatents": "The patenting propensity of the university, measured by the number of patents registered by the university in the five years prior to the collaboration."
        },
        {
            "SpinOff": "A binary variable indicating whether the university has at least one spin-off."
        },
        {
            "TTO": "A binary variable indicating whether the university has a technology transfer office."
        },
        {
            "UnivReputation": "The reputation of the university, measured by its ranking in the Academic Ranking of World Universities."
        }
    ],
    "relationships": [
        {
            "node_1": "TechRel",
            "node_2": "InnValue",
            "edge": "Technological relatedness has an inverted U-shaped relationship with the value of joint innovations, indicating that both too little and too much technological similarity can be detrimental to innovation value."
        },
        {
            "node_1": "PriorTies",
            "node_2": "InnValue",
            "edge": "The existence of prior ties between universities and firms has a positive effect on the value of joint innovations, suggesting that previous collaborations can facilitate the development of more valuable innovations."
        },
        {
            "node_1": "GeoDist",
            "node_2": "InnValue",
            "edge": "Geographical distance between universities and firms is positively related to the achievement of higher innovative outcomes, which contradicts the common assumption that proximity is beneficial for knowledge transfer and innovation."
        },
        {
            "node_1": "FirmSize",
            "node_2": "InnValue",
            "edge": "Firm size has a positive and significant effect on the joint innovation value, indicating that larger firms may be more capable of developing valuable joint innovations with universities."
        },
        {
            "node_1": "FirmPatents",
            "node_2": "InnValue",
            "edge": "A firm's technological capital, as measured by the number of patents, positively affects the value of joint innovations, suggesting that a firm's existing technological competencies contribute to the creation of new technology in collaboration with universities."
        },
        {
            "node_1": "UnivSize",
            "node_2": "InnValue",
            "edge": "The size of the university, in terms of the number of full-time researchers, does not show a significant relationship with the value of joint innovations."
        } 
    ]
}
"""

data = json.loads(temp)

# print(completion.choices[0].message.content)

new_nodes = []
new_edges = []

for variable in data['variables']:
    for name, definition in variable.items():
        new_nodes.append({
            'name': name,
            'define': definition,
            'source': data['paper']
        })

for relationship in data['relationships']:
    new_edges.append({
        'node_1': relationship['node_1'],
        'node_2': relationship['node_2'],
        'edge': relationship['edge'],
        'source': data['paper']
    })
    

nodes_file = 'nodes.csv'
edges_file = 'edges.csv'


def append_to_csv(file, fieldnames, new_data):
    file_exists = os.path.isfile(file)
    
    with open(file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for row in new_data:
            writer.writerow(row)


# 追加新的节点数据到 nodes.csv
append_to_csv(nodes_file, ['name', 'define', 'source'], new_nodes)

# 追加新的边数据到 edges.csv
append_to_csv(edges_file, ['node_1', 'node_2', 'edge', 'source'], new_edges)

print("添加成功")
'''