from langchain_core.prompts import PromptTemplate
print('has_format', hasattr(PromptTemplate, 'format'))
print('has_format_prompt', hasattr(PromptTemplate, 'format_prompt'))
print('methods', [n for n in dir(PromptTemplate) if 'format' in n.lower()])
