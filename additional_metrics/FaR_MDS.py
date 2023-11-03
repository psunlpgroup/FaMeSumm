import regex
import glob
import os
import json


medical_term_collection = set()

with open('ALL_medical_term_file_train.txt', 'r', encoding='utf8') as f:
    custom_noun = f.readlines()
    for i in range(len(custom_noun)):
        medical_term_collection.add(custom_noun[i].replace('\n', ''))

# We downloaded Tsinghua open chinese lexicon (medical domain) to collect a large collection of medical terms in addition to 'ALL_medical_term_file_train.txt'. 
with open('THUOCL_medical.txt', 'r', encoding='utf8') as f:
    custom_noun = f.readlines()
    for i in range(len(custom_noun)):
        medical_term_collection.add(regex.findall(r'\p{Han}+', custom_noun[i])[0])

total_score = 0
path = "mt5_dec_dir"
for output_filename in glob.glob(os.path.join(path, '*.txt')):
    dia_id = int(output_filename.split("/")[-1].split("_")[0])
    dia_id_str = str(dia_id)
    
    with open(output_filename, 'r', encoding='utf8') as f:
        output = f.readlines()[0]
        output_set, output_set_med = set(output.split(" ")), set()
        for e in output_set:
            if e in medical_term_collection: output_set_med.add(e)
        
    with open("mt5_ref_dir/" + dia_id_str + "_reference.txt", 'r', encoding='utf8') as f:
        reference = f.readlines()[0]
        reference_set, reference_set_med = set(reference.split(" ")), set()
        for e in reference_set:
            if e in medical_term_collection: reference_set_med.add(e)

    # tokenize the source texts of the test set and place them into 'source_test_tokenized'
    with open("source_test_tokenized/" + dia_id_str + "_source.txt", 'r', encoding='utf8') as f:
        source = f.readlines()[0]
        source_set, source_set_med = set(source.split(" ")), set()
        for e in source_set:
            if e in medical_term_collection: source_set_med.add(e)

    B_C = reference_set_med.intersection(source_set_med)
    C = reference_set_med.intersection(source_set_med, output_set_med)
    if len(B_C) == 0:
        total_score += 0
    else:
        total_score += len(C) / len(B_C)

print("FaR:", total_score / 288)
