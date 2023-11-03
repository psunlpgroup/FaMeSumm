import glob
import os
from evaluate import load
bertscore = load("bertscore")

predictions = []
references = []


path = "dialogue_output/mt5_ref_dir"
for ref_filename in glob.glob(os.path.join(path, '*.txt')):
    dia_id = int(ref_filename.split("/")[-1].split("_")[0])
    dia_id_str = str(dia_id)
    # length = len(str(dia_id))
    # dia_id_str = "0" * (6 - length) + str(dia_id)
    output_file = "dialogue_output/mt5_dec_dir/" + dia_id_str + "_decoded.txt"

    with open(ref_filename, 'r', encoding='utf8') as f:
        see = f.readlines()[0].replace(" ", "")
        references.append(see)

        with open(output_file, 'r', encoding='utf8') as f2:
            whole = f2.readlines()
            for i in range(1, len(whole)):
                whole[0] = whole[0] + " " + whole[i]
            
            output_str = whole[0].replace(" ", "")
            predictions.append(output_str)

results = bertscore.compute(predictions=predictions, references=references, lang="ch")
final_list = results['f1']
print(sum(final_list) / len(final_list))