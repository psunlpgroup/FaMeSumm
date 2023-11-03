from summac.model_summac import SummaCZS, SummaCConv
import json

model_zs = SummaCZS(granularity="document", model_name="vitc", device="cuda") # If you have a GPU: switch to: device="cuda"


final_score = []
with open("test.txt", 'r') as input:
    idx = 0
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        source_document = patientDict["question"]

        dec_file_name = "pega_dec_dir/" + str(idx) + "_decoded.txt"

        with open(dec_file_name,'r') as f:
            contents = f.read()
            summary = contents

        score = model_zs.score([source_document], [summary])
        final_score.append(score["scores"][0])

        idx += 1

print(final_score)
print("avg:", sum(final_score) / len(final_score))
