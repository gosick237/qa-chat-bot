import random
import pandas as pd
import json
import re

def open_pkl(file_path):
    try:
        data = pd.read_pickle(file_path)
        print("파일을 성공적으로 불러왔습니다.")
        return data
    except Exception as e:
        print(f"파일을 읽는 중 에러가 발생했습니다: {e}")

if __name__ == "__main__":
    data = open_pkl("faq.pkl")
    processed_data = []
    related_count = 0
    category_count = 0

    for key, value in data.items():
        # special case: given file has BOM
        value = value.replace('\ufeff', '')

        # remove "\xa0"
        key = re.sub(r'\xa0+', '', key).strip()
        value = re.sub(r'\xa0+', '', value).strip()

        # process for 'key'
        # Extract 'category' and 'question'
        matches = re.findall(r'\[(.*?)\]', key) # category pattern
        category = [match.strip() for match in matches] # category
        if len(category):
            category_count += 1

        question = re.sub(r'\[.*?\]', '', key).strip()  # remove category info

        # process for 'value'
        if value:
            # Extract 'answer' and 'related key,answer'
            splitted = re.split(r"\n{3,}", value)

            # 'answer'
            answer = splitted[0].strip()
            
            # 'related'
            related = []
            for item in splitted[1:]:
                if "관련 도움말" in item:
                    related_item = re.split(r"\n\n", item)[-1].split("\n")
                    related = [i.strip() for i in related_item]
                    related_count += 1
                    break

            # Output
            json_object = {
                "category": category,
                "question": question,
                "answer": answer,
                "related": related
            }
            processed_data.append(json_object)

    # save with JSONL
    jsonl_file_path = f'processed_data_{len(processed_data)}.jsonl'
    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for item in processed_data:
            jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"saved in {jsonl_file_path}.")
    print(f"A number of question include related question: {related_count}")
    print(f"A number of question include category: {category_count}")

    print("-" * 50)
    print("example")
    ex_num = 5
    for i in random.sample(range(0, len(processed_data)), ex_num):
        print(processed_data[i])