import unittest
from tasks import *

# lambda string: "".join([x for x in string.lower() if x in '\n\t abcdefghijklmnopqrstuvwxyz0123456789']).split()


class TestTokenizeFunction(unittest.TestCase):

    def test_basic_sentence(self):
        self.assertEqual(tokenize("Hello, world!"), ["hello", "world"])

    def test_mixed_case(self):
        self.assertEqual(tokenize("HeLLo WoRLd"), ["hello", "world"])

    def test_punctuation_removal(self):
        self.assertEqual(tokenize("This, is a test!"),
                         ["this", "is", "a", "test"])

    def test_extra_spaces(self):
        self.assertEqual(tokenize("  Lots   of   spaces   "),
                         ["lots", "of", "spaces"])

    def test_empty_string(self):
        self.assertEqual(tokenize(""), [])

    def test_only_punctuation(self):
        self.assertEqual(tokenize("!@#$%^&*()"), [])

    def test_single_word(self):
        self.assertEqual(tokenize("Word"), ["word"])

    def test_special_characters(self):
        self.assertEqual(tokenize("Hello @#%& world!"), ["hello", "world"])

    def test_newlines_and_tabs(self):
        self.assertEqual(tokenize("Hello\nworld\tthis\tis\na test"), [
                         "hello", "world", "this", "is", "a", "test"])


unittest.main()
