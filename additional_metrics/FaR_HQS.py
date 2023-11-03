import regex
import glob
import os
import json


total_2medical = []
with open("test.txt", 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        total_2medical.append(patientDict["2_medical"])

total_l = []
total_score = 0
path = "pega_dec_dir"
for output_filename in glob.glob(os.path.join(path, '*.txt')):
    idx = int(output_filename.split("/")[-1].split("_")[0])
    B_C, C = len(total_2medical[idx]), 0
    if B_C == 0: 
        total_l.append(0)
        continue
    
    with open(output_filename, 'r', encoding='utf8') as f:
        output = f.readlines()[0]
        temp_l = sorted(total_2medical[idx], key=lambda x: (-len(x), x))
        
        for e in temp_l:
            if e in output:
                C += 1
                output = output.replace(e, "")
    
    total_score += C / B_C
    total_l.append(C / B_C)

print("FaR:", total_score / 100)
print(total_l)
