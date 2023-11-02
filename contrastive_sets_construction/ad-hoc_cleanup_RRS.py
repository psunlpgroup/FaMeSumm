from tqdm import tqdm
import glob
import json
import os
import spacy
import scispacy
from scispacy.linking import EntityLinker
nlp = spacy.load("en_core_web_sm")
from pattern.text.en import singularize
import nlpaug.augmenter.word as naw
aug = naw.BackTranslationAug(device='cuda')


path_train, json_list = "train.txt", []
with open(path_train, 'r') as input:
	for jsonObj in input:
		patientDict = json.loads(jsonObj)
		json_list.append(patientDict)
		

idx_list = []
for i in range(91544):
	for name in glob.iglob("Positive/" + str(i) + "/*.txt"):
		if name.split("/")[-1] == "positive2.txt":
			with open(name) as f:
				source = f.read()
				if 'TECHNIQUE:' == source or "CT" == source:
					idx_list.append(i)

for i in idx_list:
	os.remove("Positive/" + str(i) + "/positive4.txt")
	os.remove("Positive/" + str(i) + "/positive2.txt")

	written_str = None

	if len(json_list[i]["2_medical"]) > 1:
		target_term = json_list[i]["2_medical"][-2].lower()
		doc = nlp(json_list[i]["text"])
		sent_list = list(doc.sents)
		for j in range(len(sent_list)-1, -1, -1):
			if target_term in str(sent_list[j]).lower() or singularize(target_term) in str(sent_list[j]).lower():
				written_str = str(sent_list[j]).replace("[SEP]", "").strip()
				with open("Positive/" + str(i) + "/positive2.txt", 'w') as test_file:
					test_file.write(written_str)
				break
	else:
		doc = nlp(json_list[i]["text"].split("[SEP]")[0])
		sent_list = list(doc.sents)
		max_len, max_idx = 0, -1
		for j in range(len(sent_list)-1, -1, -1):
			if len(str(sent_list[j]).strip()) >= max_len:
				max_idx = j
				max_len = len(str(sent_list[j]).strip())
		written_str = str(sent_list[max_idx]).strip()
		with open("Positive/" + str(i) + "/positive3.txt", 'w') as test_file:
			test_file.write(written_str)

	augmented_data = aug.augment(written_str)[0]
	with open("Positive/" + str(i) + "/positive4.txt", 'w') as test_file:
		test_file.write(augmented_data)