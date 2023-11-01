from tqdm import tqdm
import numpy as np
import glob
import os
import json
import random
import re

import spacy
import scispacy

from scispacy.linking import EntityLinker

nlp = spacy.load("en_core_web_sm")

import nlpaug.augmenter.word as naw
aug = naw.BackTranslationAug(device='cuda')

from pattern.text.en import singularize



s = set()
with open("train_RRS.txt", 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        for m in patientDict["1_medical"]:
            s.add(m)
        for m in patientDict["2_medical"]:
            s.add(m)

with open("validation_RRS.txt", 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        for m in patientDict["1_medical"]:
            s.add(m)
        for m in patientDict["2_medical"]:
            s.add(m)
            
with open("test_RRS.txt", 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        for m in patientDict["1_medical"]:
            s.add(m)
        for m in patientDict["2_medical"]:
            s.add(m)

ALL_medical_term = list(s)


# positive set
path_train, path_base, count, no_type1, no_type2 = "train_RRS.txt", "Positive/", 0, set(), set()
with open(path_train, 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        path_id = path_base + str(count) + "/"
        idx = count
        count += 1
        if not os.path.exists(path_id): os.makedirs(path_id)
            
        #1: the reference summary itself is truthful
        if len(patientDict["1_medical"]) == 0:
            with open(path_id + "positive1.txt", 'w') as test_file:
                test_file.write(patientDict["summary"].strip())
        else:
            no_type1.add(idx)
        
        #2: extract another utterance with using the last medical term (the last one in "2_medical")
        if len(patientDict["2_medical"]) > 0:
            target_term = patientDict["2_medical"][-1].lower()
            doc = nlp(patientDict["text"])
            sent_list = list(doc.sents)
            for i in range(len(sent_list)-1, -1, -1):
                if target_term in str(sent_list[i]).lower() or singularize(target_term) in str(sent_list[i]).lower():
                    with open(path_id + "positive2.txt", 'w') as test_file:
                        test_file.write(str(sent_list[i]).replace("[SEP]", "").strip())
                    break
                #no_type2.add(idx)
        else:
            no_type2.add(idx)

        
#3: extract the longest sentence in findings for any training instances without positive2
count = 0
with open(path_train, 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        idx = count
        path_id = path_base + str(count) + "/"
        count += 1
        if idx in no_type2:
            doc = nlp(patientDict["text"].split("[SEP]")[0])
            sent_list = list(doc.sents)
            max_len, max_idx = 0, -1
            for i in range(len(sent_list)-1, -1, -1):
                # if "[sep]" in str(sent_list[i]).lower():
                #     sent_list[i] = str(sent_list[i]).replace("[SEP]", "")
                if len(str(sent_list[i]).strip()) >= max_len:
                    max_idx = i
                    max_len = len(str(sent_list[i]).strip())
            with open(path_id + "positive3.txt", 'w') as test_file:
                test_file.write(str(sent_list[max_idx]).strip())
        
        #4: if reference summary is unfaithful, machine translation to perform data augmentation on #2 or #3
        if idx in no_type1:
            target_eg = ""
            if idx not in no_type2:
                with open(path_id + "positive2.txt", 'r', encoding='utf8') as f:
                    target_eg = f.readlines()[0]
            else:
                with open(path_id + "positive3.txt", 'r', encoding='utf8') as f:
                    target_eg = f.readlines()[0]
            if target_eg == "":
                print("empty target:", idx)
                
            augmented_data = aug.augment(target_eg)[0]
            with open(path_id + "positive4.txt", 'w') as test_file:
                test_file.write(augmented_data.strip())
        else:
            #5: if reference summary is faithful, machine translation to perform data augmentation on #1
            target_eg = patientDict["summary"].strip()
            
            augmented_data = aug.augment(target_eg)[0]
            with open(path_id + "positive5.txt", 'w') as test_file:
                test_file.write(augmented_data.strip())


# negative set
path_train, path_base, count = "train_RRS.txt", "Negative/", 0
with open(path_train, 'r') as input:
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        path_id = path_base + str(count) + "/"
        if not os.path.exists(path_id): os.makedirs(path_id)
        
        no_type1, no_type2 = False, False
        
        #1: the reference summary itself is untruthful
        if len(patientDict["1_medical"]) > 0:
            with open(path_id + "negative1.txt", 'w') as test_file:
                test_file.write(patientDict["summary"].strip())
        else:
            no_type1 = True
                
        #2: change 1 truthful medical term (the 1st one in "2_medical") to another untruthful one
        if len(patientDict["2_medical"]) + len(patientDict["1_medical"]) > 0:
            truthful_medical_list, random_idx = patientDict["2_medical"] + patientDict["1_medical"], random.randint(0, 5194)
            while ALL_medical_term[random_idx] in truthful_medical_list:
                random_idx = random.randint(0, 5194)

            modified_summary = patientDict["summary"].replace(truthful_medical_list[0], ALL_medical_term[random_idx])
            with open(path_id + "negative2.txt", 'w') as test_file:
                test_file.write(modified_summary)
        else:
            no_type2 = True
        
        #3: append an untruthful medical term to the start of the reference summary
        truthful_medical_list, random_idx = patientDict["2_medical"] + patientDict["1_medical"], random.randint(0, 5194)
        while ALL_medical_term[random_idx] in truthful_medical_list:
            random_idx = random.randint(0, 5194)
        modified_summary = ALL_medical_term[random_idx] + " " + patientDict["summary"]
        with open(path_id + "negative3.txt", 'w') as test_file:
            test_file.write(modified_summary)
            
        #4: append an untruthful medical term to the END of the reference summary
        if no_type1 and no_type2:
            prev_idx = random_idx
            random_idx = random.randint(0, 5194)
            while prev_idx == random_idx or ALL_medical_term[random_idx] in truthful_medical_list:
                random_idx = random.randint(0, 5194)
            modified_summary = patientDict["summary"] + " " + ALL_medical_term[random_idx]
            with open(path_id + "negative4.txt", 'w') as test_file:
                test_file.write(modified_summary)
            
        count += 1
                