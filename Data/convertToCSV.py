

outputfile = open("ethylene_methane.csv", "w")

with open("ethylene_methane.txt", "r") as f:
    for line in f:
        words = line.split()
        for i in range(len(words)):
            outputfile.write(words[i])
            if i < len(words) - 1:
                outputfile.write(",")
        outputfile.write("\n")

outputfile.close()
