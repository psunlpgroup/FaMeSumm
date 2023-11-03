from summac.model_summac import SummaCZS
import json

model = SummaCZS(granularity="document", model_name="vitc", device="cuda")



final_score = []
with open("test.txt", 'r') as input:
    idx = 0
    for jsonObj in input:
        patientDict = json.loads(jsonObj)
        source_document = patientDict["text"]

        # RR_outputs/indiana_base_dec
        dec_file_name = "pega_dec_dir/" + str(idx) + "_decoded.txt"

        with open(dec_file_name,'r') as f:
            contents = f.read()
            summary = contents

        score = model.score([source_document], [summary])
        final_score.append(score["scores"][0])

        idx += 1

print(final_score)
print("avg:", sum(final_score) / len(final_score))
