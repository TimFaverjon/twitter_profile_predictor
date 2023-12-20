print("importing the module")

import twitter_profile_predictor

print('starting class')

extractor = twitter_profile_predictor.bios_analyzer("je suis president du monde et journaliste chez sciencespo")

print('tokenization')

extractor.full_tokenize()
print(extractor.full_tokens)

print("profession extraction")

print(extractor.get_professions())

print("statuses extraction")

print(extractor.get_statuses())

print("language recognition")

print(extractor.get_lang())