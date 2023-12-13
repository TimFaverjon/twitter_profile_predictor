print("loading the module")

import sys
sys.path.append('/sps/humanum/user/tfaverjo/twitter_profile_predictor/src/twitter_profile_predictor')

import bios_analyzer as tba

example_bios_1 = "je suis president du monde et journaliste chez sciencespo"

print('starting class')

extractor = tba.bios_analyzer(example_bios_1)

print('tokenization')

extractor.full_tokenize()
print(extractor.full_tokens)

print("profession extraction")

print(extractor.get_professions())

print("statuses extraction")

print(extractor.get_statuses())