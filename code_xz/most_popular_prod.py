from csv import DictReader
from collections import defaultdict
import operator

train_path = "../data/train.csv"
sample_submission = "../data/sample_submission.csv"
output_path = "../data/most_popular_prod.csv"

# get product names
with open(train_path, "rb") as infile:
    header = infile.readline().rstrip().replace('"','').split(",")
products = header[24:]

customer_existing_products = defaultdict(set)
products_total_counts = defaultdict(int)
with open(train_path,"rb") as infile:
    reader = DictReader(infile)
    for e, row in enumerate(reader):
        if e % 1000000 == 0:
            print "processing line...", e
        for p in products:
            if row[p] and int(row[p]) == 1:
                products_total_counts[p] += int(row[p])
                customer_existing_products[row['ncodpers']].add(p)

ranked_products = [x[0] for x in sorted(products_total_counts.items(), key=operator.itemgetter(1), reverse=True)]

k = 7 # number of products to include

with open(output_path, "wb") as outfile:
    outfile.write("ncodpers,added_products\n")
    with open(sample_submission, "rb") as infile:
        reader = DictReader(infile)
        for e, row in enumerate(reader):
            if e % 100000 == 0:
                print "writing line ...", e
            #if row['ncodpers'] in customer_existing_products:
            #    products_left = [x for x in ranked_products if x not in customer_existing_products[row['ncodpers']]]
            #else:
            #    products_left = ranked_products
            products_left = ranked_products
            outfile.write("%s,%s\n"%(str(row['ncodpers']), ' '.join(products_left[:k])))





