""" def read_corpus(corpus_file, binary=True):
    '''Reading in data from corpus file'''

    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        fi = fi.readlines()
        line = 0
        while line < (len(fi)):

            print("--------------")
            print("line is ")
            print(fi[line])
            print("index ")
            print(line)
            if fi[line] in ['\n', '\r\n']:
                line += 1
                continue
            
            d = fi[line].strip().split(',')
            for element in range(len(d)-1, -1, -1):
                if d[element] == '':
                    d.pop(element)

            print("d is")
            print(d)
            if d[1][0] != "\"":
                print("one-liner")
            else:
                if d[-2][-1] == "\"" and len(d[2]) == 3:
                    print("quted oneliner")
                else:
                    print("several lines")
                    lastChar = ''
                    line += 1
                    while True:
                        newD = fi[line].strip().split(',')
                        lastBlock =  list(newD[len(newD) - 2])
                        print("last block is")
                        print(lastBlock)
                        if (len(lastBlock) > 0):
                            lastChar = lastBlock[len(lastBlock) - 1]
                        d.append(" ")
                        d += newD
                        if lastChar == "\"":
                            break
                        else:
                            line += 1
            print("index at the end")
            print(line)
            line += 1
                        
                

            data = [d[0], "".join(d[1:len(d)-1]) ,d[len(d)-1]]
            print("data is")
            print(data)

            # making sure no missing labels
            if len(data) != 3:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            tweets.append(data[1])
            if binary:
                if data[2] == 'NAG':
                    labels.append('NON')
                else:
                    #labels.append('OFF')
                    labels.append(data[2])
            else:
                labels.append(data[2])
    print(labels)
    return tweets, labels """


def read_corpus(corpus_file, binary=True):
    '''Reading in data from corpus file'''
    ids = []
    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 5:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            ids.append(data[0])


            tweets.append(data[1])
            labels.append(data[2])
    #print(ids[1:20])
    #print(tweets[1:20])
    #print(labels[1:20])
    return ids[1:], tweets[1:], labels[1:]


read_corpus('/Users/balinthompot/RUG/Honours/HateSpeech/offenseval-rug-master/Data/train/offenseval-training-v1.tsv')