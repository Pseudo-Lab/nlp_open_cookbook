import pytest

from sklearn.feature_extraction.text import CountVectorizer

@pytest.mark.skip(reason="n of vocab = 10224")
def test_bow_without_parser(bn_test_data):
    bow = CountVectorizer()
    bow.fit(bn_test_data['text'])
    print(len(bow.vocabulary_))
    assert len(bow.vocabulary_) > 100