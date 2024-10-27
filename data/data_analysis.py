import pandas as pd

def open_pkl(file_path):
    try:
        data = pd.read_pickle(file_path)
        print("파일을 성공적으로 불러왔습니다.")
        return data
    except Exception as e:
        print(f"파일을 읽는 중 에러가 발생했습니다: {e}")

def check_shape(data):
    if isinstance(data, pd.DataFrame):
        print("데이터는 DataFrame 형태입니다.:", data.info())
        print("예시:", data.head())
    elif isinstance(data, dict):
        print("데이터는 딕셔너리 형태입니다.", len(data))
        key, val = list(data.items())[0]
        print(f"key: {type(key)}, value: {type(val)}")
        for i, (k, v) in enumerate(data.items()):
            if i< 10:
                print("Raw data: ")
                print(" Key:", repr(k))
                print(" Value:", repr(v))
                """print("예시:")
                print(" Key:", k)
                print(" Value:", v)"""
    elif isinstance(data, list):
        print("데이터는 리스트 형태입니다.")
        print("데이터 개수:", len(data))
        for i, element in enumerate(data):
            if i < 3:
                print(f"[{i}]Key: {element} ({type(element)})")
            else:
                break
    else:
        print("알 수 없는 데이터 구조입니다:")

if __name__ == "__main__":
    data = open_pkl("faq.pkl")
    check_shape(data)