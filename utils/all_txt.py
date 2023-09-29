import json
import os

dir_path = "/home/ml/qa_system_llm/copus/txt_files"
list_dir = os.listdir(dir_path)
result_list = []
for file_name in list_dir:
    if not file_name.endswith("copy.txt"):
        with open("/home/ml/qa_system_llm/copus/txt_files/" + file_name, "r") as rf:
            lines = rf.readlines()
            len_lines = len(lines)
            for i in range(len_lines):
                result_list.append(lines[i])
            rf.close()
with open("/home/ml/qa_system_llm/copus/txt_files/all.txt", "w") as wf:
    result = "".join(result_list)
    json.dump(result, wf, ensure_ascii=False)
    wf.close()