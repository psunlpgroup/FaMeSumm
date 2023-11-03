from questeval.questeval_metric import QuestEval
import os
import json

questeval = QuestEval(no_cuda=False)
hypothesis, sources, list_references = [], [], []

# read txt files 
with open("test.txt", 'r') as input:
	idx = 0
	for jsonObj in input:
		ref = []
		patientDict = json.loads(jsonObj)
		sources.append(patientDict["text"])
		ref.append(patientDict["summary"])
		list_references.append(ref)

		dec_file_name = "pega_dec_dir/" + str(idx) + "_decoded.txt"

		with open(dec_file_name,'r') as f:
			contents = f.read()
			hypothesis.append(contents)

		idx += 1

# score = questeval.corpus_questeval(hypothesis=hypothesis, sources=sources, list_references=list_references)
score = questeval.corpus_questeval(hypothesis=hypothesis, sources=sources)

print(score)

