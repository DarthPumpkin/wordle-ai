import csv
import json
import string


SAOL_CSV = "saol2018.csv"
OUTFILE = "words_sv.json"
WORD_LEN = 5
PERMITTED_CHARS = string.ascii_lowercase + "åäö"


with open(SAOL_CSV, 'r', encoding='utf-8') as fp:
    r = csv.reader(fp)
    all_words = [l[1].lower() for l in r]

words_l5 = [w for w in all_words if len(w) == WORD_LEN]
words_permitted = [w for w in words_l5 if all(c in PERMITTED_CHARS for c in w)]
words_permitted = sorted(set(words_permitted))
word_lists = {'guesses': [], 'solutions': words_permitted}
with open(OUTFILE, 'w', encoding='utf-8') as fp:
    json.dump(word_lists, fp, ensure_ascii=False, indent=2)

print(f"Extracted {len(words_permitted)} words of length {WORD_LEN} and saved to {OUTFILE}.")
