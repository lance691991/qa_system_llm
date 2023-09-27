import docx2txt
import os

dir_path = "/home/ml/qa_system_llm/copus/doc_files"
list_dir = os.listdir(dir_path)

for file_name in list_dir:
    if file_name.endswith(".docx"):
        corpus_path = "/home/ml/qa_system_llm/copus/doc_files/" + file_name
        text = docx2txt.process(corpus_path)
        text_lines = text.split("\n")

        with open("/home/ml/qa_system_llm/copus/txt_files/" + file_name[0: -5] + ".txt", "w") as f:
            for l in text_lines:
                l = l.replace("\n", "").replace("\t", "").replace(" ", "")
                if not l:
                    continue
                if l == "—" or l == "……" or l == "●" or l == "　" or l == "　●":
                    continue
                f.write(f"{l}\n")
            f.close()