    
import csv

def createSubmission(path, ids, predictions):
    with open(path, 'w', newline='') as csvfile:
        for index in range(len(ids)):
            pred = str(predictions[index])
            csvfile.write(str(ids[index] + "\t" + pred + "\n"))

ids = ['111111','222222','3333333','4444444']
labels = ['NON', 'OFF', 'NON', 'OFF']
createSubmission("test_submission.csv", ids, labels)