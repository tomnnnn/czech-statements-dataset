import json

with open("../datasets/with_evidence/bing/statements.json") as f:
    statements = json.load(f)

true_count = len([s for s in statements if s['assessment'].lower() == "pravda"])
false_count = len([s for s in statements if s['assessment'].lower() == "nepravda"])
unverifiable_count = len([s for s in statements if s['assessment'].lower() == "neověřitelné"])
misleading_count = len([s for s in statements if s['assessment'].lower() not in ["pravda", "nepravda", "neověřitelné"]])

print(f"""Distribution:
True: {true_count} ({round(true_count/len(statements),2)*100} %)
False: {false_count} ({round(false_count/len(statements),2)*100} %)
Unverifiable: {unverifiable_count} ({round(unverifiable_count/len(statements),2)*100} %)
Misleading: {misleading_count} ({round(misleading_count/len(statements),2)*100} %)
Total: {len(statements)}""")
