import os
import langchain

root = os.path.abspath(os.path.join(os.path.dirname(langchain.__file__), '..'))
print('root', root)
for dirpath, dirnames, filenames in os.walk(root):
    for fname in filenames:
        if not fname.endswith('.py'):
            continue
        path = os.path.join(dirpath, fname)
        try:
            text = open(path, 'r', encoding='utf-8').read()
        except Exception:
            continue
        if any(x in text for x in ['RetrievalQA', 'FAISS', 'PyPDFLoader', 'RecursiveCharacterTextSplitter']):
            print(path)
