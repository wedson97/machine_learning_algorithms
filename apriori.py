import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

with open('Transacoes.txt', 'r') as f:
    transactions = [line.strip().split(",") for line in f.readlines()]


te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

frequente_itemsets = apriori(df, min_support=0.5, use_colnames=True)

rules = association_rules(frequente_itemsets, metric='confidence', min_threshold=0.5)
print(rules)