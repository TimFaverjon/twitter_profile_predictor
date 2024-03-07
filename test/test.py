import unittest
from twitter_profile_predictor import bios_analyzer

#### Test 1 : one bios loaded

class TestBiosAnalyzer(unittest.TestCase):
    def setUp(self):
        self.example_bios_1 = "je suis president du monde et journaliste chez sciencespo"
        self.extractor = bios_analyzer.bios_analyzer(self.example_bios_1)

    def test_full_tokenize(self):
        self.extractor.full_tokenize()
        self.assertEqual(self.extractor.full_tokens, ['je', 'suis', 'president', 'du', 'monde', 'et', 'journaliste', 'chez', 'sciencespo'])

    def test_get_professions(self):
        professions = self.extractor.get_professions()
        self.assertEqual(professions, ['president', 'journaliste'])

    def test_get_statuses(self):
        statuses = self.extractor.get_statuses()
        self.assertEqual(statuses, ['monde'])

    def test_get_lang(self):
        lang = self.extractor.get_lang()
        self.assertEqual(lang, 'fr')

if __name__ == '__main__':
    unittest.main()