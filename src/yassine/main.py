from tqdm import tqdm

filename_in = 'properties06191148_modified.csv'
filename_out = 'test_file_cleaned.csv'

print("starting cleaning dataset...")

with open(filename_in) as f_in, open(filename_out, "w") as f_out:
    for line in tqdm(f_in.readlines()):
        ls_line = line.split(",")
        last_ten = ls_line[-11:-1]
        for i in range(len(last_ten)):
            if last_ten[i] == 'False':
                last_ten[i] = 0
            elif last_ten[i] == 'True':
                last_ten[i] = 1
        f_out.write(",".join([str(e) for e in last_ten]) + "," + ls_line[-1])
f_out.close()

print("Done !")
