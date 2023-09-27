import json
import os

dir_path = "/home/ml/qa_system_llm/copus/txt_files"
list_dir = os.listdir(dir_path)
result_list = []
for file_name in list_dir:
    if file_name.endswith("copy.txt"):
        with open("/home/ml/qa_system_llm/copus/txt_files/" + file_name, "r") as rf:
            lines = rf.readlines()
            len_lines = len(lines)
            i = 0
            while i < len_lines:
                if lines[i] == "\n":
                    i += 1
                else:
                    if "问：" in lines[i]:
                        question = lines[i]
                        j = 1
                        while not lines[i + j] == "\n":
                            j += 1
                            if i + j >= len_lines:
                                break
                            else:
                                lines[i + j - 1] = lines[i + j - 1].replace("\t", "")
                        answer = ("").join(lines[i + 1: i + j])
                        # answer = lines[i + 1: i + j]
                        i += j
                        input = {
                            "instruction": question,
                            "input": "",
                            "output": answer
                        }
                        result_list.append(input)
            rf.close()

with open("/home/ml/qa_system_llm/copus/json_files/regulation_qa.json", "w") as wf:
    json.dump(result_list, wf, ensure_ascii=False)
    wf.close()